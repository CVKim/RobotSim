# -*- coding: utf-8 -*-
"""Track C 데이터셋 빌더: 합성 인스턴스 마스크 → YOLO-seg 폴리곤 라벨.

입력
  BlenderProc: explore/sdg_data/*.hdf5  ('category_id_segmaps' = 박스별 id)
  Isaac      : explore/isaac_data/semantic_segmentation_XXXX.png + _labels_XXXX.json (isaac_to_yolo.read_seg)
  실측       : BINPICK_DIR 30세션 → 기하 검출기 rect_px → cv2.boxPoints 4점 폴리곤 (pseudo-GT)

이미지는 이미 렌더된 noisy 검출 데이터셋(explore/sdg_yolo_noisy, explore/isaac_yolo_noisy)의 png를
그대로 재사용한다 → 검출 모델과 동일한 입력·동일한 train/val 분할 (공정 비교).

산출
  explore/seg_yolo_cotrain : train = BlenderProc(noisy) + Isaac(noisy) + 실측 6장, val = 합성 val
  explore/seg_yolo_holdout : val   = 실측 24장 (sdg_cotrain.py 와 동일 분할: sorted(real)[::5][:6] 제외)

폴리곤 규칙: 인스턴스 마스크 → 최대 외곽 컨투어 → approxPolyDP(1px) → [0,1] 정규화.
"""
import shutil
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from binpick_topface import detect_boxes  # noqa: E402
from isaac_to_yolo import read_seg  # noqa: E402
from mim_loader import load_session, valid_mask  # noqa: E402
from sdg_dataset import depth_to_img, write_yaml  # noqa: E402

try:
    from local_paths import BINPICK_DIR
except ImportError:
    import os
    BINPICK_DIR = os.environ["BINPICK_DIR"]

ROOT = Path(r"E:\Robot_Sim\explore")
CO = ROOT / "seg_yolo_cotrain"
HOLD = ROOT / "seg_yolo_holdout"
MIN_AREA = 150  # 검출 변환기(sdg_dataset/isaac_to_yolo)와 동일


def mask_to_polygon(mask, eps_px=1.0):
    """단일 인스턴스 이진 마스크 → 최대 외곽 컨투어 → (N,2) 픽셀 폴리곤. 실패 시 None."""
    m8 = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if eps_px > 0:
        c = cv2.approxPolyDP(c, eps_px, True)
    c = c.reshape(-1, 2)
    return c if len(c) >= 3 else None


def polygon_line(poly_px, w, h, cls=0):
    pts = np.clip(np.asarray(poly_px, np.float64) / [w, h], 0.0, 1.0)
    return f"{cls} " + " ".join(f"{x:.5f} {y:.5f}" for x, y in pts)


def seg_to_lines(seg, min_area=MIN_AREA):
    """id 맵(0=배경) → YOLO-seg 라인 목록."""
    h, w = seg.shape
    lines = []
    for i in np.unique(seg):
        if i == 0:
            continue
        m = seg == i
        if m.sum() < min_area:
            continue
        poly = mask_to_polygon(m)
        if poly is None:
            continue
        lines.append(polygon_line(poly, w, h))
    return lines


def _find_split(img_root, stem):
    for split in ("train", "val"):
        p = img_root / "images" / split / f"{stem}.png"
        if p.exists():
            return split, p
    raise FileNotFoundError(f"{stem}.png not in {img_root}")


def _mkdirs(dst):
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (dst / sub).mkdir(parents=True, exist_ok=True)


def convert_blenderproc(dst, src="sdg_data", img_src="sdg_yolo_noisy"):
    n_img, n_inst, n_pts = 0, 0, []
    for f in sorted((ROOT / src).glob("*.hdf5")):
        with h5py.File(f, "r") as h:
            seg = h["category_id_segmaps"][:]
        split, img = _find_split(ROOT / img_src, f.stem)
        lines = seg_to_lines(seg)
        shutil.copy(img, dst / "images" / split / f"bp_{f.stem}.png")
        (dst / "labels" / split / f"bp_{f.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_img += 1
        n_inst += len(lines)
        n_pts += [len(l.split()) // 2 for l in lines]
    print(f"blenderproc: {n_img} imgs, {n_inst} instances, mean poly pts={np.mean(n_pts):.1f}")
    return n_img, n_inst


def convert_isaac(dst, src="isaac_data", img_src="isaac_yolo_noisy"):
    n_img, n_inst, n_pts = 0, 0, []
    for png in sorted((ROOT / src).glob("semantic_segmentation_*.png")):
        idx = png.stem.split("_")[-1]
        seg = read_seg(png, ROOT / src / f"semantic_segmentation_labels_{idx}.json")
        split, img = _find_split(ROOT / img_src, f"isaac_{idx}")
        lines = seg_to_lines(seg)
        shutil.copy(img, dst / "images" / split / f"isaac_{idx}.png")
        (dst / "labels" / split / f"isaac_{idx}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_img += 1
        n_inst += len(lines)
        n_pts += [len(l.split()) // 2 for l in lines]
    print(f"isaac: {n_img} imgs, {n_inst} instances, mean poly pts={np.mean(n_pts):.1f}")
    return n_img, n_inst


def real_polygon_lines(sess):
    """실측 세션 → 기하 검출기 rect_px(minAreaRect) → 4점 폴리곤 pseudo-GT."""
    _, _, boxes = detect_boxes(sess)
    h, w = sess["D"].shape
    return [polygon_line(cv2.boxPoints(b["rect_px"]), w, h) for b in boxes]


def build_real(co, hold):
    """실측 30세션: 6장은 co-train train에, 24장은 holdout val에 (sdg_cotrain.py 와 동일 분할)."""
    reals = sorted((ROOT / "real_eval" / "images" / "val").glob("*.png"))
    train_stems = {r.stem for r in reals[::5][:6]}
    n_train_box = n_hold_box = 0
    for s in sorted(d for d in Path(BINPICK_DIR).iterdir() if d.is_dir()):
        sess = load_session(s)
        img = depth_to_img(sess["D"], valid_mask(sess))
        lines = real_polygon_lines(sess)
        # 검출 pseudo-GT 와 박스 수 일치 확인
        n_det = len((ROOT / "real_eval/labels/val" / f"{s.name}.txt").read_text(encoding="utf-8").splitlines())
        assert n_det == len(lines), (s.name, n_det, len(lines))
        if s.name in train_stems:
            cv2.imwrite(str(co / "images/train" / f"real_{s.name}.png"), img)
            (co / "labels/train" / f"real_{s.name}.txt").write_text("\n".join(lines), encoding="utf-8")
            n_train_box += len(lines)
        else:
            cv2.imwrite(str(hold / "images/val" / f"{s.name}.png"), img)
            (hold / "labels/val" / f"{s.name}.txt").write_text("\n".join(lines), encoding="utf-8")
            n_hold_box += len(lines)
    print(f"real: train 6 imgs/{n_train_box} boxes, holdout {30 - len(train_stems)} imgs/{n_hold_box} boxes")


def main():
    for d in (CO, HOLD):
        if d.exists():
            shutil.rmtree(d)
        _mkdirs(d)
    convert_blenderproc(CO)
    convert_isaac(CO)
    build_real(CO, HOLD)
    write_yaml(CO)
    write_yaml(HOLD)
    for d in (CO, HOLD):
        for split in ("train", "val"):
            print(f"{d.name}/{split}: {len(list((d / 'images' / split).glob('*.png')))} imgs")


if __name__ == "__main__":
    main()
