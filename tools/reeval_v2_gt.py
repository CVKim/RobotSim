# -*- coding: utf-8 -*-
"""학습 모델(검출 3종·세그 1종)을 v2 기하 검출 pseudo-GT 로 재평가.

기존 sim2real/seg 평가의 실측 정답은 v1 검출기(detect_boxes) 출력 148박스였다. v2(detect_boxes_v2, RGB 대조 검증)로
확인된 실제 박스는 167개 → 19개(11%)가 배경으로 라벨돼 있었고, 학습 모델이 그 박스를 잡으면 FP 로 계산됐다(정밀도·mAP 과소).
같은 holdout 24장(sdg_cotrain 분리 그대로)에서 v1-GT 와 v2-GT 로 각 모델의 mAP50 을 나란히 계산한다.
산출: explore/sim2real/reeval_v2gt.json
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from binpick_topface import detect_boxes, detect_boxes_v2  # noqa: E402
from mim_loader import load_session  # noqa: E402

try:
    from local_paths import BINPICK_DIR
except ImportError:
    import os
    BINPICK_DIR = os.environ["BINPICK_DIR"]

ROOT = Path(r"E:\Robot_Sim\explore")
HOLD_DET = ROOT / "real_holdout"          # v1-GT bbox holdout (24장)
HOLD_SEG = ROOT / "seg_yolo_holdout"      # v1-GT polygon holdout (같은 24장)
MODELS = {
    "det_bp_noisy": ROOT / "sim2real/train_noisy/weights/best.pt",
    "det_isaac_noisy": ROOT / "sim2real/train_isaac_noisy/weights/best.pt",
    "det_cotrain_6real": ROOT / "sim2real/train_cotrain/weights/best.pt",
    "seg_cotrain_6real": ROOT / "seg/train_seg/weights/best.pt",
}


def write_split(src, dst, names, det_fn, poly):
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (dst / sub).mkdir(parents=True, exist_ok=True)
    n = 0
    for s in names:
        shutil.copy(src / "images/val" / f"{s}.png", dst / "images/val" / f"{s}.png")
        sess = load_session(Path(BINPICK_DIR) / s)
        h, w = sess["D"].shape
        _, _, bx = det_fn(sess)
        lines = []
        for b in bx:
            pts = cv2.boxPoints(b["rect_px"])
            if poly:
                lines.append("0 " + " ".join(f"{np.clip(x / w, 0, 1):.5f} {np.clip(y / h, 0, 1):.5f}" for x, y in pts))
            else:
                x0, y0 = pts.min(0)
                x1, y1 = pts.max(0)
                lines.append(f"0 {(x0 + x1) / 2 / w:.5f} {(y0 + y1) / 2 / h:.5f} {(x1 - x0) / w:.5f} {(y1 - y0) / h:.5f}")
        n += len(lines)
        (dst / "labels/val" / f"{s}.txt").write_text("\n".join(lines), encoding="utf-8")
    (dst / "data.yaml").write_text(f"path: {dst}\ntrain: images/train\nval: images/val\nnames:\n  0: box\n",
                                   encoding="utf-8")
    return n


def main():
    from ultralytics import YOLO
    names = sorted(p.stem for p in (HOLD_DET / "images/val").glob("*.png"))
    assert len(names) == 24, names
    n_v2_det = write_split(HOLD_DET, ROOT / "real_holdout_v2", names, detect_boxes_v2, poly=False)
    n_v2_seg = write_split(HOLD_SEG, ROOT / "seg_yolo_holdout_v2", names, detect_boxes_v2, poly=True)
    n_v1 = sum(len((HOLD_DET / "labels/val" / f"{s}.txt").read_text().split("\n")) for s in names
               if (HOLD_DET / "labels/val" / f"{s}.txt").read_text().strip())
    print(f"holdout 24: v1-GT {n_v1} boxes, v2-GT {n_v2_det} boxes (seg polygons {n_v2_seg})")

    res = {"holdout_sessions": names, "gt_boxes": {"v1": n_v1, "v2": n_v2_det}, "models": {}}
    for name, w in MODELS.items():
        seg = name.startswith("seg")
        model = YOLO(str(w))
        out = {}
        for gt, data in (("v1", (HOLD_SEG if seg else HOLD_DET) / "data.yaml"),
                         ("v2", (ROOT / ("seg_yolo_holdout_v2" if seg else "real_holdout_v2")) / "data.yaml")):
            r = model.val(data=str(data), split="val", device=0, verbose=False, plots=False, workers=0)
            out[gt] = {"box_map50": round(float(r.box.map50), 4), "box_map5095": round(float(r.box.map), 4),
                       "box_precision": round(float(r.box.mp), 4), "box_recall": round(float(r.box.mr), 4)}
            if seg:
                out[gt].update({"mask_map50": round(float(r.seg.map50), 4), "mask_map5095": round(float(r.seg.map), 4)})
        out["delta_map50"] = round(out["v2"]["box_map50"] - out["v1"]["box_map50"], 4)
        res["models"][name] = out
        print(f"{name:18s} mAP50 v1-GT {out['v1']['box_map50']:.3f} -> v2-GT {out['v2']['box_map50']:.3f} "
              f"(P {out['v1']['box_precision']:.3f}->{out['v2']['box_precision']:.3f}, "
              f"R {out['v1']['box_recall']:.3f}->{out['v2']['box_recall']:.3f})")
    (ROOT / "sim2real/reeval_v2gt.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved", ROOT / "sim2real/reeval_v2gt.json")


if __name__ == "__main__":
    main()
