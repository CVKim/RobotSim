# -*- coding: utf-8 -*-
"""Isaac Replicator BasicWriter 출력 → YOLO 데이터셋 (BlenderProc 파이프라인과 동일 전처리).

입력: distance_to_image_plane_XXXX.npy(z-depth, m), semantic_segmentation_XXXX.png(uint32 id)
      + semantic_segmentation_labels_XXXX.json (id -> {"class": "boxK"|...})
전처리: sdg_dataset.depth_to_img / add_tof_noise 재사용 (동일 노이즈·정규화 → 공정 비교)
"""
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sdg_dataset import add_tof_noise, depth_to_img, write_yaml  # noqa: E402

ROOT = Path(r"E:\Robot_Sim\explore")


def read_seg(png_path, json_path):
    seg = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if seg.ndim == 3:  # RGBA-packed uint32
        seg = seg[..., 0].astype(np.uint32) | (seg[..., 1].astype(np.uint32) << 8) | \
              (seg[..., 2].astype(np.uint32) << 16) | (seg[..., 3].astype(np.uint32) << 24)
    labels = json.loads(Path(json_path).read_text(encoding="utf-8"))
    box_ids = {int(k): int(v["class"][3:]) for k, v in labels.items()
               if str(v.get("class", "")).startswith("box")}
    cat = np.zeros(seg.shape, np.int32)
    for pix_id, cid in box_ids.items():
        cat[seg == pix_id] = cid
    return cat


def convert(src_dir, dst_dir, noisy, val_frac=0.1, seed=0, min_area=150):
    rng = np.random.default_rng(seed)
    dst = ROOT / dst_dir
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (dst / sub).mkdir(parents=True, exist_ok=True)
    deps = sorted(glob.glob(str(ROOT / src_dir / "distance_to_image_plane_*.npy")))
    n_val, n_boxes = 0, 0
    for dp in deps:
        idx = Path(dp).stem.split("_")[-1]
        D = np.load(dp).astype(np.float64) * 1000.0
        valid = np.isfinite(D) & (D < 1e6) & (D > 0)
        seg = read_seg(ROOT / src_dir / f"semantic_segmentation_{idx}.png",
                       ROOT / src_dir / f"semantic_segmentation_labels_{idx}.json")
        if noisy:
            D, valid = add_tof_noise(D, valid, rng)
        img = depth_to_img(D, valid)
        h, w = seg.shape
        lines = []
        for i in np.unique(seg):
            if i == 0:
                continue
            m = seg == i
            if m.sum() < min_area:
                continue
            ys, xs = np.nonzero(m)
            lines.append(f"0 {(xs.min()+xs.max())/2/w:.5f} {(ys.min()+ys.max())/2/h:.5f} "
                         f"{(xs.max()-xs.min())/w:.5f} {(ys.max()-ys.min())/h:.5f}")
        n_boxes += len(lines)
        split = "val" if rng.random() < val_frac else "train"
        n_val += split == "val"
        cv2.imwrite(str(dst / f"images/{split}/isaac_{idx}.png"), img)
        (dst / f"labels/{split}/isaac_{idx}.txt").write_text("\n".join(lines), encoding="utf-8")
    write_yaml(dst)
    print(f"{dst_dir}: {len(deps)} frames ({n_val} val), boxes={n_boxes}, noisy={noisy}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="isaac_data")
    args = ap.parse_args()
    convert(args.src, "isaac_yolo_noisy", noisy=True)
