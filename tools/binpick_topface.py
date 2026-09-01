# -*- coding: utf-8 -*-
"""T1 프로토타입: 빈피킹 현장 ToF에서 박스 상면 검출 + mm 치수 측정.

원리 (딥러닝 없이 기하만):
  1. D 히스토그램에서 최상층(카메라에 가장 가까운) 평면 깊이 피크 탐색
  2. 피크 ±tol 픽셀 → 상면 마스크 → 모폴로지 정리 → 연결요소 = 박스 후보
  3. 각 박스: X/Y 좌표맵(mm)에서 min-area rect → 상면 L×W (mm)
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mim_loader import load_session, valid_mask


def find_top_layer(D, valid, bin_mm=10, roi_frac=0.25):
    """중앙 ROI(팔레트 영역)의 깊이 히스토그램에서 최대 피크 = 박스 상면 깊이."""
    h, w = D.shape
    roi = np.zeros_like(valid)
    roi[int(h * roi_frac):int(h * (1 - roi_frac)), int(w * roi_frac):int(w * (1 - roi_frac))] = True
    d = D[valid & roi]
    lo, hi = np.percentile(d, [1, 99])
    bins = np.arange(lo, hi + bin_mm, bin_mm)
    hist, edges = np.histogram(d, bins=bins)
    i = int(np.argmax(hist))
    # 피크 주변 ±2bin 가중 평균으로 서브빈 정밀화
    j0, j1 = max(0, i - 2), min(len(hist), i + 3)
    centers = 0.5 * (edges[j0:j1] + edges[j0 + 1:j1 + 1])
    return float(np.average(centers, weights=hist[j0:j1] + 1e-9))


def detect_boxes(sess, tol_mm=40, min_area_px=700):
    D, X, Y = sess["D"], sess["X"], sess["Y"]
    valid = valid_mask(sess)
    top_d = find_top_layer(D, valid)

    mask = (np.abs(D - top_d) < tol_mm) & valid
    # 박스 간 이음새 = 깊이 그래디언트 에지 → 마스크에서 제거해 분리 유도
    Ds = cv2.medianBlur(D.astype(np.float32), 5)
    gx = cv2.Sobel(Ds, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Ds, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    mask &= grad < 70.0  # mm/px (3m ToF 노이즈 감안)

    m8 = (mask * 255).astype(np.uint8)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, kernel3, iterations=2)  # 점 구멍 메움
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, kernel3, iterations=1)

    # ToF 강도(I) 채널 — 깊이와 픽셀 정합 완벽 → 박스 이음새를 에지로 절단
    I = sess["I"]
    In = np.log1p(np.clip(I, 0, None))
    In = cv2.normalize(In, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    In = cv2.GaussianBlur(In, (3, 3), 0)
    edges = cv2.Canny(In, 40, 100)
    edges = cv2.dilate(edges, kernel3, iterations=1)

    sep = m8.copy()
    sep[edges > 0] = 0
    n, labels = cv2.connectedComponents(sep)

    boxes = []
    for i in range(1, n):
        comp = (labels == i)
        if comp.sum() < min_area_px:
            continue
        # 에지로 깎인 영역 복원(살짝 팽창) 후 top-layer 마스크로 제한
        comp = cv2.dilate((comp * 255).astype(np.uint8), kernel3, iterations=2) > 0
        comp &= (m8 > 0)
        ys, xs = np.nonzero(comp)
        # 미터릭 치수: X/Y 좌표맵(mm)에서 min-area rect
        pts_mm = np.stack([X[comp], Y[comp]], axis=-1).astype(np.float32)
        rect_mm = cv2.minAreaRect(pts_mm)
        (w_mm, h_mm) = rect_mm[1]
        # 픽셀 상 표시용 rect
        rect_px = cv2.minAreaRect(np.stack([xs, ys], axis=-1).astype(np.float32))
        # 직사각형성/종횡비 필터 (바닥 줄무늬 등 오검출 제거)
        rect_area = max(rect_px[1][0] * rect_px[1][1], 1)
        fill = comp.sum() / rect_area
        aspect = max(w_mm, h_mm) / max(min(w_mm, h_mm), 1)
        if fill < 0.6 or aspect > 3.5:
            continue
        boxes.append({
            "area_px": int(comp.sum()),
            "dims_mm": (round(max(w_mm, h_mm), 1), round(min(w_mm, h_mm), 1)),
            "depth_mm": round(float(np.median(D[comp])), 1),
            "rect_px": rect_px,
        })
    return top_d, mask, boxes


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
