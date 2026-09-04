# -*- coding: utf-8 -*-
"""빈피킹 상면 검출 — 알고리즘은 robotsim_perception.geometry 로 이동했다.

이 모듈은 기존 12개 tools 스크립트의 import 를 그대로 유지하기 위한 재수출 + 시각화 계층이다.
알고리즘 수정은 robotsim_perception/geometry.py 에서 한다 (중복 구현으로 인한 버그 재발 방지).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from mim_loader import load_session  # noqa: E402,F401
from robotsim_perception.geometry import (  # noqa: E402,F401
    SKU_PRIOR_MM, SKU_TOL, SENTINEL_D, SENTINEL_XY, valid_mask,
    _affine_px_from_mm, _annotate_layer, _box_geometry, _cell_poly_px, _detect_boxes_at,
    _dims_score, _estimate_lattice, _intensity_edges, _LatticeGrid, _orient_class,
    _plane_rms, _plane_score, _refine_peak, _roi_depth_hist,
    detect_boxes, detect_boxes_v2, find_top_layer, top_layer_candidates,
)


def render_overlay(sess, top_d, mask, boxes, out_path):
    D = sess["D"]
    valid = valid_mask(sess)
    lo, hi = np.percentile(D[valid], [2, 98])
    norm = np.clip((D - lo) / max(hi - lo, 1), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_BONE)
    img[~valid] = (20, 20, 20)
    img[mask] = (0.55 * img[mask] + 0.45 * np.array([0, 140, 255])).astype(np.uint8)
    for i, b in enumerate(boxes):
        pts = cv2.boxPoints(b["rect_px"]).astype(np.int32)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(img, f'{i+1}', (cx - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f'top layer @ {top_d:.0f}mm, {len(boxes)} boxes',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(out_path), img)


def render_overlay_v2(sess, top_d, mask, boxes, out_path):
    """v2 오버레이: detected=초록, inferred=주황, 박스별 신뢰도 표기."""
    D = sess["D"]
    valid = valid_mask(sess)
    lo, hi = np.percentile(D[valid], [2, 98]) if valid.any() else (0, 1)
    norm = np.clip((D - lo) / max(hi - lo, 1), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_BONE)
    img[~valid] = (20, 20, 20)
    img[mask] = (0.55 * img[mask] + 0.45 * np.array([0, 140, 255])).astype(np.uint8)
    n_inf = 0
    for i, b in enumerate(boxes):
        pts = cv2.boxPoints(b["rect_px"]).astype(np.int32)
        inferred = b.get("source") == "inferred"
        n_inf += int(inferred)
        color = (0, 165, 255) if inferred else (0, 255, 0)
        cv2.polylines(img, [pts], True, color, 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(img, f'{i+1}', (cx - 8, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(img, f'{b.get("confidence", 0):.2f}', (cx - 14, cy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
    cv2.putText(img, f'top @ {top_d:.0f}mm, {len(boxes)} boxes ({n_inf} inferred)',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(out_path), img)


if __name__ == "__main__":
    import os
    try:
        from local_paths import BINPICK_DIR
    except ImportError:
        BINPICK_DIR = os.environ["BINPICK_DIR"]
    root = Path(BINPICK_DIR)
    out = Path(r"E:\Robot_Sim\explore\topface")
    out.mkdir(parents=True, exist_ok=True)
    sessions = sorted([d for d in root.iterdir() if d.is_dir()])
    for s in sessions:
        sess = load_session(s)
        top_d, mask, boxes = detect_boxes(sess)
        render_overlay(sess, top_d, mask, boxes, out / f"{s.name}_topface.png")
        dims = ", ".join(f'{b["dims_mm"][0]:.0f}x{b["dims_mm"][1]:.0f}' for b in boxes[:6])
        print(f"{s.name}: top@{top_d:.0f}mm boxes={len(boxes)} dims(mm)=[{dims}{'...' if len(boxes)>6 else ''}]")
