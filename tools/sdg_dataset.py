# -*- coding: utf-8 -*-
"""T5 데이터셋 빌더: 합성(hdf5)→YOLO 변환(+노이즈 ablation), 실측 평가셋 생성.

도메인 정렬 원칙: 합성/실측 모두 depth를 고정 범위(2600–3500mm) 그레이스케일로 렌더링.
노이즈 ablation: clean vs noisy(가우시안 σ + 에지 인접 드롭아웃 + 스펙클) — 렌더는 1회.
실측 GT: binpick_topface 검출(수동 검증됨 148박스)을 pseudo-GT로 사용.
"""
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from binpick_topface import detect_boxes  # noqa: E402
from mim_loader import load_session, valid_mask  # noqa: E402

try:
    from local_paths import BINPICK_DIR
except ImportError:
    import os
    BINPICK_DIR = os.environ["BINPICK_DIR"]

D_LO, D_HI = 2600.0, 4300.0
ROOT = Path(r"E:\Robot_Sim\explore")


def depth_to_img(D_mm, valid):
    """고정 범위 깊이 → 3ch 그레이 이미지. 무효 픽셀 = 0(검정)."""
    n = np.clip((D_mm - D_LO) / (D_HI - D_LO), 0, 1)
    g = (n * 254 + 1).astype(np.uint8)  # 1..255 (0은 무효 전용)
    g[~valid] = 0
    return cv2.merge([g, g, g])


def add_tof_noise(D_mm, valid, rng, sigma=3.0, edge_drop=0.55, speckle=0.04, n_blobs=(6, 18)):
    """실측 특성 기반 노이즈: 가우시안 σ + 에지 드롭아웃 + 스펙클 + 대면적 블롭 무효
    (실측 프레임의 저반사/기구부 대면적 결손 재현 — 유효율 ~60% 수준). """
    h, w = D_mm.shape
    D = D_mm + rng.normal(0, sigma, D_mm.shape)
    g = cv2.morphologyEx(D_mm.astype(np.float32), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    drop = (rng.random(D.shape) < edge_drop) & (g > 30)
    drop |= rng.random(D.shape) < speckle
    blob = np.zeros((h, w), np.uint8)
    for _ in range(rng.integers(*n_blobs)):
        cv2.ellipse(blob, (rng.integers(0, w), rng.integers(0, h)),
                    (int(rng.integers(8, 80)), int(rng.integers(8, 80))),
                    float(rng.uniform(0, 180)), 0, 360, 1, -1)
    drop |= blob > 0
    return D, valid & ~drop


def masks_to_yolo_lines(seg, min_area=150):
    h, w = seg.shape
    lines = []
    for i in np.unique(seg):
        if i == 0:
            continue
        m = seg == i
        if m.sum() < min_area:
            continue
        ys, xs = np.nonzero(m)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        bw, bh = (x1 - x0) / w, (y1 - y0) / h
        lines.append(f"0 {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}")
    return lines


def write_yaml(dst, val_only=False):
    (dst / "data.yaml").write_text(
        f"path: {dst}\ntrain: images/train\nval: images/val\nnames:\n  0: box\n",
        encoding="utf-8")


def convert_synth(src_dir, dst_dir, noisy, val_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    dst = ROOT / dst_dir
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (dst / sub).mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / src_dir).glob("*.hdf5"))
    n_val = 0
    for f in files:
        with h5py.File(f, "r") as h:
            D = h["depth"][:] * 1000.0
            seg = h["category_id_segmaps"][:]
        valid = D < 1e6
        if noisy:
            D, valid = add_tof_noise(D, valid, rng)
        img = depth_to_img(D, valid)
        lines = masks_to_yolo_lines(seg)
        split = "val" if rng.random() < val_frac else "train"
        n_val += split == "val"
        stem = f.stem
        cv2.imwrite(str(dst / f"images/{split}/{stem}.png"), img)
        (dst / f"labels/{split}/{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    write_yaml(dst)
    print(f"{dst_dir}: {len(files)} scenes ({n_val} val), noisy={noisy}")


def build_real_eval(dst_dir="real_eval"):
    """실측 30세션 → 동일 렌더링 + 검출 pseudo-GT (val 전용 데이터셋)."""
    dst = ROOT / dst_dir
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (dst / sub).mkdir(parents=True, exist_ok=True)
    n_boxes = 0
    for s in sorted(d for d in Path(BINPICK_DIR).iterdir() if d.is_dir()):
        sess = load_session(s)
        v = valid_mask(sess)
        img = depth_to_img(sess["D"], v)
        _, _, boxes = detect_boxes(sess)
        h, w = sess["D"].shape
        lines = []
        for b in boxes:
            pts = cv2.boxPoints(b["rect_px"])
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            lines.append(f"0 {(x0+x1)/2/w:.5f} {(y0+y1)/2/h:.5f} {(x1-x0)/w:.5f} {(y1-y0)/h:.5f}")
        n_boxes += len(lines)
        cv2.imwrite(str(dst / f"images/val/{s.name}.png"), img)
        (dst / f"labels/val/{s.name}.txt").write_text("\n".join(lines), encoding="utf-8")
    write_yaml(dst)
    print(f"real_eval: 30 sessions, {n_boxes} pseudo-GT boxes")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "synth"], required=True)
    ap.add_argument("--src", default="sdg_data")
    args = ap.parse_args()
    if args.mode == "real":
        build_real_eval()
    else:
        convert_synth(args.src, "sdg_yolo_clean", noisy=False)
        convert_synth(args.src, "sdg_yolo_noisy", noisy=True)
