# -*- coding: utf-8 -*-
"""실용성 검증 1-b: 기하 검출기 v1(detect_boxes) vs v2(detect_boxes_v2, 격자 결손 보완) 강건성 비교.

robustness_bench.py 와 같은 교란 격자(스펙클/가우시안/대비/무효 블롭, rng seed 0, 같은 소비 순서)를
실측 30프레임에 주입하고, 동일한 교란 프레임에 v1·v2 를 모두 돌려 비교한다.

기준(정답) = 클린 프레임의 v2 출력. 30프레임 전부 RGB(로컬 전용)와 대조해 검증: v1 이 놓친 박스(2층 4x3 격자
중앙 2개의 병합 실패, 층 말미 1~2개 잔여 박스의 피크 전환 누락)를 v2 가 전부 회수하고 빈 크레이트·빈 팔레트·
티어시트 프레임에서 오검출 0. 클린 v1 148 은 이 기준 대비 리콜로 보고한다.
  - count ratio    : 검출 박스 수 / 클린 v2 기준선
  - matched recall : 클린 v2 박스(중심 80mm·각 25도 이내 1:1 매칭) 대비 회수율 — 위치까지 맞는지
  - unmatched      : 어느 클린 박스에도 매칭되지 않은 박스 수 (오검출; det=v1 조각, inf=v2 추론 셀)
  - empty-crate FP : 빈 크레이트 3세션에서 나온 박스 수 (0 이어야 함)
산출: explore/robustness/robustness_v2.json + degradation_v2.png (v1/v2 곡선)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from binpick_topface import detect_boxes, detect_boxes_v2  # noqa: E402
from mim_loader import load_session  # noqa: E402
from robustness_bench import perturb  # noqa: E402

try:
    from local_paths import BINPICK_DIR
except ImportError:
    import os
    BINPICK_DIR = os.environ["BINPICK_DIR"]

OUT = Path(r"E:\Robot_Sim\explore\robustness")
OUT.mkdir(parents=True, exist_ok=True)
EMPTY = {"126013011432055", "126013011443804", "126013011460258"}   # 빈 크레이트 세션
GRID = {"speckle": [0.05, 0.15, 0.30], "gauss": [5, 10, 20],
        "contrast": [0.5, 0.25, 0.1], "blobs": [5, 15, 30]}


def match(gt, boxes, tol_mm=80.0, tol_deg=25.0):
    """boxes 를 gt 에 1:1 탐욕 매칭 (중심 거리·장축 각도). 반환 (matched, unmatched_boxes)."""
    used = [False] * len(gt)
    m, un = 0, []
    for b in boxes:
        c = np.array(b["center_mm"])
        best, bd = None, 1e9
        for i, g in enumerate(gt):
            if used[i]:
                continue
            d = float(np.hypot(*(c - np.array(g["center_mm"]))))
            da = abs((b["ang_deg"] - g["ang_deg"] + 90) % 180 - 90)
            if d < tol_mm and da < tol_deg and d < bd:
                bd, best = d, i
        if best is None:
            un.append(b)
        else:
            used[best] = True
            m += 1
    return m, un


def dims_stat(boxes):
    if not boxes:
        return None
    d = np.array([b["dims_mm"] for b in boxes], float)
    return {"L_mm": round(float(d[:, 0].mean()), 1), "L_std": round(float(d[:, 0].std()), 1),
            "W_mm": round(float(d[:, 1].mean()), 1), "W_std": round(float(d[:, 1].std()), 1)}


def run_v1(sess):
    # v1 박스에 중심/각도만 붙임 (매칭용) — 검출 결과 자체는 detect_boxes 와 동일
    return detect_boxes_v2(sess, max_rounds=0, layer_fallback=False)[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-seeds", default="", help="추가 시드(쉼표) — blobs 평균 보고용")
    args = ap.parse_args()

    sessions = sorted(d for d in Path(BINPICK_DIR).iterdir() if d.is_dir())
    names = [s.name for s in sessions]
    loaded = [load_session(s) for s in sessions]

    # (a) 클린 기준선 + 지연시간
    t0 = time.perf_counter()
    for sess in loaded:
        detect_boxes(sess)
    lat1 = (time.perf_counter() - t0) / len(loaded) * 1000
    t0 = time.perf_counter()
    clean_v2 = [detect_boxes_v2(sess)[2] for sess in loaded]
    lat2 = (time.perf_counter() - t0) / len(loaded) * 1000
    clean_v1 = [run_v1(sess) for sess in loaded]
    n1 = sum(len(b) for b in clean_v1)
    n2 = sum(len(b) for b in clean_v2)
    all1 = [b for bx in clean_v1 for b in bx]
    all2 = [b for bx in clean_v2 for b in bx]
    inf2 = [b for b in all2 if b["source"] == "inferred"]
    fb2 = sorted({n for n, bx in zip(names, clean_v2) if any(b.get("layer_fallback") for b in bx)})
    conf = np.array([b["confidence"] for b in all2])
    print(f"latency: v1 {lat1:.1f} ms/frame, v2 {lat2:.1f} ms/frame (CPU)")
    print(f"clean v1 recall vs validated v2 count: {n1}/{n2} = {n1 / n2:.1%}")
    print(f"clean v1: {n1} boxes {dims_stat(all1)} | clean v2: {n2} boxes ({len(inf2)} inferred, "
          f"{len(all2) - len(inf2)} detected; layer-fallback sessions={fb2}) {dims_stat(all2)}")
    print(f"clean v2 confidence p5/50/95 = {np.percentile(conf, [5, 50, 95]).round(3)}")
    per_session = []
    for n, b1, b2 in zip(names, clean_v1, clean_v2):
        per_session.append({"session": n, "v1": len(b1), "v2": len(b2),
                            "v2_inferred": sum(b["source"] == "inferred" for b in b2),
                            "v2_layer_fallback": any(b.get("layer_fallback") for b in b2),
                            "empty_crate": n in EMPTY})
        if n in EMPTY:
            print(f"  empty crate {n}: v1={len(b1)} v2={len(b2)}")
        elif len(b1) != len(b2):
            print(f"  {n}: v1={len(b1)} v2={len(b2)} inferred={per_session[-1]['v2_inferred']} "
                  f"fallback={per_session[-1]['v2_layer_fallback']}")

    res = {"latency_ms_per_frame": {"v1": round(lat1, 1), "v2": round(lat2, 1)},
           "baseline": {"v1_boxes": n1, "v1_dims": dims_stat(all1), "v2_boxes": n2, "v2_inferred": len(inf2),
                        "v1_clean_recall_vs_v2": round(n1 / n2, 3),
                        "v2_dims": dims_stat(all2),
                        "v2_detected_dims": dims_stat([b for b in all2 if b["source"] == "detected"]),
                        "v2_layer_fallback_sessions": fb2,
                        "empty_crate_boxes": {"v1": sum(len(b1) for n, b1 in zip(names, clean_v1) if n in EMPTY),
                                              "v2": sum(len(b2) for n, b2 in zip(names, clean_v2) if n in EMPTY)}},
           "per_session_clean": per_session,
           "matching": {"gt": "clean v2 boxes", "center_tol_mm": 80, "angle_tol_deg": 25},
           "degradation": {}}

    # (b) 열화: 같은 교란 프레임에 v1/v2 동시 적용 (rng 소비 순서 = robustness_bench 와 동일)
    def sweep(seed, kinds=None, verbose=True):
        rng = np.random.default_rng(seed)
        out = {}
        for kind, levels in GRID.items():
            out[kind] = []
            for lv in levels:
                c1 = c2 = m1 = m2 = fp1 = fp2 = ninf = 0
                un_d = un_i = un1 = 0
                bx1_all, bx2_all = [], []
                for n, sess, gt in zip(names, loaded, clean_v2):
                    p = perturb(sess, kind, lv, rng)
                    if kinds is not None and kind not in kinds:
                        continue
                    b1 = run_v1(p)
                    b2 = detect_boxes_v2(p)[2]
                    c1 += len(b1)
                    c2 += len(b2)
                    ninf += sum(b["source"] == "inferred" for b in b2)
                    mm1, u1 = match(gt, b1)
                    mm2, u2 = match(gt, b2)
                    m1 += mm1
                    m2 += mm2
                    un1 += len(u1)
                    un_d += sum(b["source"] != "inferred" for b in u2)
                    un_i += sum(b["source"] == "inferred" for b in u2)
                    if n in EMPTY:
                        fp1 += len(b1)
                        fp2 += len(b2)
                    bx1_all += b1
                    bx2_all += b2
                if kinds is not None and kind not in kinds:
                    continue
                rec = {"level": lv,
                       "v1": {"boxes": c1, "recall_vs_base": round(c1 / n2, 3), "matched": m1,
                              "matched_recall": round(m1 / n2, 3), "unmatched": un1,
                              "empty_crate_boxes": fp1, "dims": dims_stat(bx1_all)},
                       "v2": {"boxes": c2, "recall_vs_base": round(c2 / n2, 3), "matched": m2,
                              "matched_recall": round(m2 / n2, 3), "inferred": ninf,
                              "unmatched_detected": un_d, "unmatched_inferred": un_i,
                              "empty_crate_boxes": fp2, "dims": dims_stat(bx2_all)}}
                out[kind].append(rec)
                if verbose:
                    print(f"  {kind}={lv:g}: v1 boxes={c1} ({c1 / n2:.0%}) matched={m1} ({m1 / n2:.0%}) | "
                          f"v2 boxes={c2} ({c2 / n2:.0%}) matched={m2} ({m2 / n2:.0%}) inferred={ninf} "
                          f"unmatched det/inf={un_d}/{un_i} | emptyFP v1={fp1} v2={fp2} | "
                          f"L={rec['v1']['dims']['L_mm'] if rec['v1']['dims'] else None}/"
                          f"{rec['v2']['dims']['L_mm'] if rec['v2']['dims'] else None}")
        return out

    print(f"seed {args.seed}:")
    res["degradation"] = sweep(args.seed)
    if args.extra_seeds:
        seeds = [int(s) for s in args.extra_seeds.split(",")]
        multi = {}
        for s in seeds:
            print(f"extra seed {s} (blobs only):")
            multi[s] = sweep(s, kinds={"blobs"})
        agg = []
        for i, lv in enumerate(GRID["blobs"]):
            recs = [res["degradation"]["blobs"][i]] + [multi[s]["blobs"][i] for s in seeds]
            agg.append({"level": lv, "n_seeds": len(recs),
                        "v1_recall_mean": round(float(np.mean([r["v1"]["recall_vs_base"] for r in recs])), 3),
                        "v2_recall_mean": round(float(np.mean([r["v2"]["recall_vs_base"] for r in recs])), 3),
                        "v1_matched_mean": round(float(np.mean([r["v1"]["matched_recall"] for r in recs])), 3),
                        "v2_matched_mean": round(float(np.mean([r["v2"]["matched_recall"] for r in recs])), 3),
                        "v2_unmatched_inferred_mean": round(float(np.mean([r["v2"]["unmatched_inferred"] for r in recs])), 1),
                        "empty_crate_boxes_sum": {"v1": int(sum(r["v1"]["empty_crate_boxes"] for r in recs)),
                                                  "v2": int(sum(r["v2"]["empty_crate_boxes"] for r in recs))}})
        res["blobs_multi_seed"] = {"seeds": [args.seed] + seeds, "levels": agg}
        for a in agg:
            print(f"  blobs={a['level']:g} mean over {a['n_seeds']} seeds: count recall v1 {a['v1_recall_mean']:.1%} -> v2 "
                  f"{a['v2_recall_mean']:.1%}; matched v1 {a['v1_matched_mean']:.1%} -> v2 {a['v2_matched_mean']:.1%}; "
                  f"unmatched inferred/seed {a['v2_unmatched_inferred_mean']}; emptyFP {a['empty_crate_boxes_sum']}")

    (OUT / "robustness_v2.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), dpi=130)
    titles = {"speckle": "extra invalid px (%)", "gauss": "depth noise sigma (mm)",
              "contrast": "intensity contrast (x)", "blobs": "invalid blobs (#)"}
    for j, (kind, recs) in enumerate(res["degradation"].items()):
        xs = [r["level"] * (100 if kind == "speckle" else 1) for r in recs]
        ax = axes[0, j]
        ax.plot(xs, [r["v1"]["recall_vs_base"] * 100 for r in recs], "o-", color="#9ca3af", lw=2, label="v1 detect_boxes")
        ax.plot(xs, [r["v2"]["recall_vs_base"] * 100 for r in recs], "s-", color="#d97706", lw=2, label="v2 lattice completion")
        ax.axhline(100, color="#9ca3af", ls="--", lw=1)
        ax.set_ylim(0, 120)
        ax.set_xlabel(titles[kind])
        ax.set_ylabel(f"box count vs validated clean {n2} (%)")
        ax.grid(alpha=0.3)
        ax = axes[1, j]
        ax.plot(xs, [r["v1"]["matched_recall"] * 100 for r in recs], "o-", color="#9ca3af", lw=2, label="v1")
        ax.plot(xs, [r["v2"]["matched_recall"] * 100 for r in recs], "s-", color="#d97706", lw=2, label="v2")
        ax.axhline(100, color="#9ca3af", ls="--", lw=1)
        ax.set_ylim(0, 110)
        ax.set_xlabel(titles[kind])
        ax.set_ylabel(f"position-matched recall vs {n2} (%)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Geometric detector v1 vs v2 on 30 real frames (v1 {lat1:.0f} ms, v2 {lat2:.0f} ms/frame CPU; "
                 f"clean: v1 {n1} / v2 {n2} boxes, RGB-validated)")
    fig.tight_layout()
    fig.savefig(OUT / "degradation_v2.png")
    print("saved", OUT / "robustness_v2.json", OUT / "degradation_v2.png")


if __name__ == "__main__":
    main()
