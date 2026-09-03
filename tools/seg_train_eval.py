# -*- coding: utf-8 -*-
"""Track C: 학습형 인스턴스 세그멘테이션(YOLO11n-seg) vs 기하 검출기 — 학습·평가·열화 강건성 비교.

학습 데이터: explore/seg_yolo_cotrain (BlenderProc noisy + Isaac noisy 합성 + 실측 6장, seg_dataset.py)
평가:
  (1) explore/seg_yolo_holdout 실측 24장: box mAP50 / mask mAP50 (pseudo-GT = 기하 검출기 rect_px 폴리곤)
  (2) robustness_bench.py 와 동일한 열화 그리드(speckle/gauss/contrast/blobs, 동일 seed·순서)를
      실측 30세션에 주입 → sdg_dataset.depth_to_img 렌더 → 기하 / seg / det(cotrain) 3종 검출
      → 무열화 기하 pseudo-GT 와 IoU>=0.5 매칭 recall·precision (30장 전체 + 학습 미사용 24장)
  (3) 추론 지연 ms/frame: GPU(RTX3080) / CPU, 기하(CPU) 재측정
산출: explore/seg/seg_results.json, explore/seg/seg_vs_geometric.png

실행: CUDA_VISIBLE_DEVICES=1 python tools/seg_train_eval.py --stage train|eval|chart
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from binpick_topface import detect_boxes  # noqa: E402
from mim_loader import load_session, valid_mask  # noqa: E402
from robustness_bench import perturb  # noqa: E402
from sdg_dataset import depth_to_img  # noqa: E402

try:
    from local_paths import BINPICK_DIR
except ImportError:
    import os
    BINPICK_DIR = os.environ["BINPICK_DIR"]

ROOT = Path(r"E:\Robot_Sim\explore")
OUT = ROOT / "seg"
OUT.mkdir(parents=True, exist_ok=True)
CO = ROOT / "seg_yolo_cotrain"
HOLD = ROOT / "seg_yolo_holdout"
DET_HOLD = ROOT / "real_holdout"          # 검출 모델용 bbox holdout (동일 24장)
DET_W = ROOT / "sim2real/train_cotrain/weights/best.pt"
SEG_W = OUT / "train_seg/weights/best.pt"

GRID = {"speckle": [0.05, 0.15, 0.30], "gauss": [5, 10, 20],
        "contrast": [0.5, 0.25, 0.1], "blobs": [5, 15, 30]}
CONF, IOU_MATCH = 0.25, 0.5


# ----------------------------------------------------------------------------- train
def train(epochs=60):
    from ultralytics import YOLO
    m = YOLO("yolo11n-seg.pt")
    m.train(data=str(CO / "data.yaml"), epochs=epochs, imgsz=640, batch=16, device=0,
            workers=0, project=str(OUT), name="train_seg", exist_ok=True,
            verbose=False, plots=False, seed=0)
    r = m.val(data=str(HOLD / "data.yaml"), split="val", device=0, verbose=False, plots=False)
    print(f"holdout box mAP50={r.box.map50:.4f} mask mAP50={r.seg.map50:.4f}")


# ----------------------------------------------------------------------------- helpers
def rect_to_xyxy(rect_px):
    p = cv2.boxPoints(rect_px)
    return np.array([*p.min(axis=0), *p.max(axis=0)], np.float32)


def geometric_xyxy(sess):
    _, _, boxes = detect_boxes(sess)
    return np.array([rect_to_xyxy(b["rect_px"]) for b in boxes], np.float32).reshape(-1, 4)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=-1)

    def area(x):
        return (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])

    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match_count(gt, det, thr=IOU_MATCH):
    """greedy 1:1 매칭 TP 수 (IoU 내림차순)."""
    M = iou_matrix(gt, det)
    if M.size == 0:
        return 0
    tp, used_g, used_d = 0, set(), set()
    for i, j in zip(*np.unravel_index(np.argsort(-M, axis=None), M.shape)):
        if M[i, j] < thr:
            break
        if i in used_g or j in used_d:
            continue
        used_g.add(i)
        used_d.add(j)
        tp += 1
    return tp


def predict_xyxy(model, imgs, device):
    rs = model.predict(imgs, imgsz=640, conf=CONF, device=device, verbose=False)
    return [r.boxes.xyxy.cpu().numpy().astype(np.float32).reshape(-1, 4) for r in rs]


def render(sess):
    return depth_to_img(sess["D"], valid_mask(sess))


# ----------------------------------------------------------------------------- eval
def eval_holdout(seg, det, device):
    out = {}
    r = seg.val(data=str(HOLD / "data.yaml"), split="val", device=device, verbose=False, plots=False)
    out["seg_holdout"] = {"box_map50": round(float(r.box.map50), 4), "box_map5095": round(float(r.box.map), 4),
                          "mask_map50": round(float(r.seg.map50), 4), "mask_map5095": round(float(r.seg.map), 4)}
    r = seg.val(data=str(CO / "data.yaml"), split="val", device=device, verbose=False, plots=False)
    out["seg_synth_val"] = {"box_map50": round(float(r.box.map50), 4), "mask_map50": round(float(r.seg.map50), 4)}
    r = det.val(data=str(DET_HOLD / "data.yaml"), split="val", device=device, verbose=False, plots=False)
    out["det_cotrain_holdout"] = {"box_map50": round(float(r.box.map50), 4), "box_map5095": round(float(r.box.map), 4)}
    print("holdout:", json.dumps(out))
    return out


def score(gt, dets, n_gt, hold_idx):
    """프레임별 검출 목록 → pseudo-GT 매칭 recall/precision (30장 전체, 학습 미사용 24장)."""
    tp = np.array([match_count(g, x) for g, x in zip(gt, dets)])
    nd = np.array([len(x) for x in dets])

    def agg(idx):
        return {"recall": round(float(tp[idx].sum() / n_gt[idx].sum()), 3),
                "precision": round(float(tp[idx].sum() / max(nd[idx].sum(), 1)), 3),
                "dets": int(nd[idx].sum())}

    return {"all30": agg(np.arange(len(gt))), "holdout24": agg(hold_idx)}


def eval_degradation(seg, det, loaded, hold_idx, device):
    """robustness_bench.main() 과 동일한 rng(0)·kind→level→session 순서로 교란 → 3종 검출기."""
    gt = [geometric_xyxy(s) for s in loaded]                       # 무열화 기하 pseudo-GT (148)
    n_gt = np.array([len(g) for g in gt])
    base = {"geometric": gt,
            "seg": predict_xyxy(seg, [render(s) for s in loaded], device),
            "det": predict_xyxy(det, [render(s) for s in loaded], device)}
    print(f"baseline dets: geometric={n_gt.sum()} seg={sum(map(len, base['seg']))} det={sum(map(len, base['det']))}")

    res = {"n_gt_boxes_30": int(n_gt.sum()), "n_gt_boxes_holdout24": int(n_gt[hold_idx].sum()),
           "baseline": {k: score(gt, d, n_gt, hold_idx) for k, d in base.items()}, "degradation": {}}

    rng = np.random.default_rng(0)
    for kind, levels in GRID.items():
        res["degradation"][kind] = []
        for lv in levels:
            pert = [perturb(s, kind, lv, rng) for s in loaded]
            dets = {"geometric": [geometric_xyxy(p) for p in pert],
                    "seg": predict_xyxy(seg, [render(p) for p in pert], device),
                    "det": predict_xyxy(det, [render(p) for p in pert], device)}
            rec = {"level": lv}
            for k, d in dets.items():
                rec[k] = score(gt, d, n_gt, hold_idx)
            rec["geometric"]["count_ratio_vs_base"] = round(rec["geometric"]["all30"]["dets"] / int(n_gt.sum()), 3)
            res["degradation"][kind].append(rec)
            print(f"  {kind}={lv}: " + " | ".join(
                f"{k} R={rec[k]['holdout24']['recall']:.3f} P={rec[k]['holdout24']['precision']:.3f}" for k in dets))
    return res


def speckle_prefilter(seg, det, loaded, hold_idx, device, ksize=5):
    """스펙클 붕괴 진단: 렌더 후 median(ksize) 전처리(고립 무효점 메움)만 넣었을 때 학습 모델 recall.
    GRID 첫 항목이 speckle 이므로 rng(0) 재시작 → eval_degradation 과 동일한 교란 재현."""
    gt = [geometric_xyxy(s) for s in loaded]
    n_gt = np.array([len(g) for g in gt])
    out = {"ksize": ksize, "levels": []}
    rng = np.random.default_rng(0)
    for lv in [0.0] + GRID["speckle"]:
        pert = loaded if lv == 0.0 else [perturb(s, "speckle", lv, rng) for s in loaded]
        imgs = [cv2.medianBlur(render(p), ksize) for p in pert]
        rec = {"level": lv, "seg": score(gt, predict_xyxy(seg, imgs, device), n_gt, hold_idx),
               "det": score(gt, predict_xyxy(det, imgs, device), n_gt, hold_idx)}
        out["levels"].append(rec)
        print(f"  speckle={lv} +median{ksize}: seg R={rec['seg']['holdout24']['recall']:.3f} "
              f"P={rec['seg']['holdout24']['precision']:.3f} | det R={rec['det']['holdout24']['recall']:.3f}")
    return out


def eval_latency(seg, loaded, device):
    import torch
    imgs = [render(s) for s in loaded]
    out = {}
    for dev in [device, "cpu"]:
        for im in imgs[:10]:                                        # warm-up
            seg.predict(im, imgsz=640, conf=CONF, device=dev, verbose=False)
        if dev != "cpu":
            torch.cuda.synchronize()
        loops = 3 if dev != "cpu" else 1
        t0, sp = time.perf_counter(), []
        for _ in range(loops):
            for im in imgs:
                r = seg.predict(im, imgsz=640, conf=CONF, device=dev, verbose=False)[0]
                sp.append([r.speed["preprocess"], r.speed["inference"], r.speed["postprocess"]])
        if dev != "cpu":
            torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / (loops * len(imgs)) * 1000
        sp = np.mean(sp, axis=0)
        key = "seg_gpu" if dev != "cpu" else "seg_cpu"
        out[key] = {"wall_ms_per_frame": round(wall, 1), "preprocess_ms": round(float(sp[0]), 1),
                    "inference_ms": round(float(sp[1]), 1), "postprocess_ms": round(float(sp[2]), 1)}
        print(f"latency {key}: wall={wall:.1f} ms/frame (pre/inf/post = {sp.round(1).tolist()})")
    t0 = time.perf_counter()
    for s in loaded:
        detect_boxes(s)
    g = (time.perf_counter() - t0) / len(loaded) * 1000
    out["geometric_cpu"] = {"wall_ms_per_frame": round(g, 1)}
    out["cpu_threads_torch"] = torch.get_num_threads()
    print(f"latency geometric_cpu: {g:.1f} ms/frame")
    return out


# ----------------------------------------------------------------------------- chart
def chart(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = {"geometric": "#eb6834", "seg": "#2a78d6", "det": "#1baf7a"}
    L = {"geometric": "geometric detector (no learning)",
         "seg": "YOLO11n-seg (synth BP+Isaac + 6 real)",
         "det": "YOLO11n det (synth BP + 6 real)"}
    titles = {"speckle": "extra invalid px (%)", "gauss": "depth noise sigma (mm)",
              "contrast": "intensity contrast (x)", "blobs": "invalid blobs (#)"}
    deg, base = res["degradation"], res["degradation_baseline"]
    pre = res.get("speckle_prefilter")
    fig = plt.figure(figsize=(14, 10.5), dpi=130)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.15], hspace=0.55, wspace=0.32, top=0.86)
    handles = {}
    for row, metric in enumerate(["recall", "precision"]):
        for col, (kind, recs) in enumerate(deg.items()):
            ax = fig.add_subplot(gs[row, col])
            if kind == "contrast":
                xs = [1.0] + [r["level"] for r in recs]
            else:
                xs = [0] + [r["level"] * (100 if kind == "speckle" else 1) for r in recs]
            for k in ["geometric", "seg", "det"]:
                ys = [base[k]["holdout24"][metric] * 100] + [r[k]["holdout24"][metric] * 100 for r in recs]
                handles[L[k]], = ax.plot(xs, ys, "o-", color=C[k], lw=2, ms=5)
            if kind == "speckle" and pre:                         # 진단: median 전처리 추가 시
                for k in ["seg", "det"]:
                    ys = [r[k]["holdout24"][metric] * 100 for r in pre["levels"]]
                    xs2 = [r["level"] * 100 for r in pre["levels"]]
                    lab = f"{L[k].split(' (')[0]} + median{pre['ksize']} pre-filter"
                    handles[lab], = ax.plot(xs2, ys, "s--", color=C[k], lw=1.5, ms=5, alpha=0.8)
            if kind == "contrast":
                ax.invert_xaxis()
            ax.set_ylim(0, 105)
            ax.set_xlabel(titles[kind], fontsize=9)
            ax.set_ylabel(f"{metric} vs pseudo-GT (%)", fontsize=9)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=8)
    fig.legend(handles.values(), handles.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.915),
               ncol=3, fontsize=8, frameon=False)

    # mAP bars
    ax = fig.add_subplot(gs[2, :2])
    h = res["holdout"]
    names = ["det box mAP50", "seg box mAP50", "seg mask mAP50", "seg box mAP50-95", "seg mask mAP50-95"]
    vals = [h["det_cotrain_holdout"]["box_map50"], h["seg_holdout"]["box_map50"], h["seg_holdout"]["mask_map50"],
            h["seg_holdout"]["box_map5095"], h["seg_holdout"]["mask_map5095"]]
    cols = [C["det"], C["seg"], C["seg"], C["seg"], C["seg"]]
    bars = ax.bar(names, vals, color=cols, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title(f"real holdout ({res['degradation_meta']['n_gt_boxes_holdout24']} pseudo-GT boxes, 24 frames)", fontsize=10)
    ax.tick_params(axis="x", labelsize=8, rotation=12)
    ax.grid(alpha=0.25, axis="y")

    # latency bars
    ax = fig.add_subplot(gs[2, 2:])
    lat = res["latency"]
    names = ["geometric CPU", "seg GPU (RTX 3080)", "seg CPU"]
    vals = [lat["geometric_cpu"]["wall_ms_per_frame"], lat["seg_gpu"]["wall_ms_per_frame"],
            lat["seg_cpu"]["wall_ms_per_frame"]]
    bars = ax.bar(names, vals, color=[C["geometric"], C["seg"], C["seg"]], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.1f} ms", ha="center", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_ylabel("ms / frame (640x480, batch 1)", fontsize=9)
    ax.set_title("inference latency", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Learned instance segmentation vs geometric detector on real ToF (holdout 24 frames)\n"
                 "top: matched recall / precision under the robustness_bench degradation grid "
                 "(IoU>=0.5 vs un-degraded geometric pseudo-GT)", fontsize=11, y=0.97)
    fig.savefig(OUT / "seg_vs_geometric.png", bbox_inches="tight")
    print("saved:", OUT / "seg_vs_geometric.png")


# ----------------------------------------------------------------------------- main
def load_real():
    sessions = sorted(d for d in Path(BINPICK_DIR).iterdir() if d.is_dir())
    names = [s.name for s in sessions]
    loaded = [load_session(s) for s in sessions]
    train_stems = {p.stem.replace("real_", "") for p in (CO / "images/train").glob("real_*.png")}
    hold_idx = np.array([i for i, n in enumerate(names) if n not in train_stems])
    assert len(hold_idx) == 24, len(hold_idx)
    return loaded, hold_idx


def evaluate(device=0):
    from ultralytics import YOLO
    seg, det = YOLO(str(SEG_W)), YOLO(str(DET_W))
    loaded, hold_idx = load_real()
    res = {"train": {"data": "seg_yolo_cotrain (BlenderProc noisy 276 + Isaac noisy 276 + real 6; val synth 48)",
                     "model": "yolo11n-seg", "epochs": 60, "imgsz": 640, "batch": 16},
           "conf": CONF, "iou_match": IOU_MATCH}
    res["holdout"] = eval_holdout(seg, det, device)
    d = eval_degradation(seg, det, loaded, hold_idx, device)
    res["degradation_baseline"] = d.pop("baseline")
    res["degradation"] = d.pop("degradation")
    res["degradation_meta"] = d
    res["latency"] = eval_latency(seg, loaded, device)
    (OUT / "seg_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    chart(res)


def diagnose_speckle(device=0):
    """eval 이후 실행: seg_results.json 에 speckle_prefilter 항목 추가 + 차트 갱신."""
    from ultralytics import YOLO
    res = json.loads((OUT / "seg_results.json").read_text(encoding="utf-8"))
    seg, det = YOLO(str(SEG_W)), YOLO(str(DET_W))
    loaded, hold_idx = load_real()
    res["speckle_prefilter"] = speckle_prefilter(seg, det, loaded, hold_idx, device)
    (OUT / "seg_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    chart(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "eval", "speckle", "chart"], required=True)
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    if a.stage == "train":
        train(a.epochs)
    elif a.stage == "eval":
        evaluate()
    elif a.stage == "speckle":
        diagnose_speckle()
    else:
        chart(json.loads((OUT / "seg_results.json").read_text(encoding="utf-8")))
