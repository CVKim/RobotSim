# -*- coding: utf-8 -*-
"""최상층 깊이 탐색 + 박스 상면 검출 (딥러닝 없이 기하만).

tools/binpick_topface.py 의 find_top_layer / detect_boxes 와 수치적으로 동일한 절차를
복제하고 (tests/test_real_integration.py 의 파리티 테스트로 보증), 박스마다
평면 피팅(tools/binpick_pickpoints.fit_plane)으로 중심·법선·기울기와 신뢰도를 추가한다.

원리:
  1. 중앙 ROI 깊이 히스토그램 최대 피크 = 최상층(카메라 최근접 평면) 깊이
  2. 피크 ±tol_mm 마스크 → 깊이 그래디언트·강도(I) 에지로 이음새 절단 → 연결요소
  3. 각 요소: X/Y 좌표맵(mm) minAreaRect → L×W, 평면 피팅 → 중심/법선/기울기
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import cv2
import numpy as np

from .frame import Frame

DEFAULT_SKU = (293.0, 219.0, 283.0)  # (L, W, H) mm — 30세션 실측 평균 (293.0 / 218.8 / 283)
MIN_VALID_PX = 200                   # 이보다 유효 픽셀이 적으면 최상층 탐색 불가 → 박스 0개


@dataclass
class TopLayer:
    depth_mm: float          # 최상층 깊이 (NaN 이면 탐색 실패)
    mask: np.ndarray         # bool (H,W): |D - depth| < tol 이고 유효한 픽셀
    tol_mm: float
    n_px: int

    def __float__(self):
        return float(self.depth_mm)

    @property
    def ok(self) -> bool:
        return bool(np.isfinite(self.depth_mm))


@dataclass
class Box:
    id: int
    center_mm: tuple          # (X, Y, Z) 상면 평면 피팅 중심, 카메라 좌표 mm
    dims_mm: tuple            # (L, W) mm, L >= W
    tilt_deg: float           # 법선의 카메라 광축 대비 기울기
    normal: tuple             # 단위 법선, 카메라 쪽(-Z) 을 향함. 석션 접근 벡터 = -normal
    rect_px: tuple            # cv2.minAreaRect 형식 ((cx,cy),(w,h),angle) — 픽셀
    confidence: float         # 0..1 휴리스틱 (SKU 치수 일치·채움률·평면 RMS 가중합)
    depth_mm: float           # 상면 픽셀 깊이 중앙값
    area_px: int
    center_px: tuple          # (u, v) int
    plane_rms_mm: float
    fill: float               # 요소 픽셀 수 / minAreaRect 면적

    def corners_px(self) -> np.ndarray:
        return cv2.boxPoints(self.rect_px)

    def to_dict(self) -> dict:
        d = asdict(self)
        (cx, cy), (w, h), ang = self.rect_px
        d["rect_px"] = {"center": [float(cx), float(cy)], "size": [float(w), float(h)],
                        "angle": float(ang)}
        d["center_mm"] = [float(v) for v in self.center_mm]
        d["dims_mm"] = [float(v) for v in self.dims_mm]
        d["normal"] = [float(v) for v in self.normal]
        d["center_px"] = [int(v) for v in self.center_px]
        return d


# ---------------------------------------------------------------- top layer

def find_top_layer(D: np.ndarray, valid: np.ndarray, bin_mm: float = 10, roi_frac: float = 0.25) -> float:
    """중앙 ROI 깊이 히스토그램 최대 피크 (tools/binpick_topface.find_top_layer 동일).

    ROI 안에 유효 픽셀이 없으면 전체 유효 픽셀로 폴백, 그래도 없으면 NaN.
    """
    h, w = D.shape
    roi = np.zeros_like(valid)
    roi[int(h * roi_frac):int(h * (1 - roi_frac)), int(w * roi_frac):int(w * (1 - roi_frac))] = True
    d = D[valid & roi]
    if d.size < MIN_VALID_PX:
        d = D[valid]
    if d.size < MIN_VALID_PX:
        return float("nan")
    lo, hi = np.percentile(d, [1, 99])
    bins = np.arange(lo, hi + bin_mm, bin_mm)
    if len(bins) < 2:
        return float(np.median(d))
    hist, edges = np.histogram(d, bins=bins)
    i = int(np.argmax(hist))
    j0, j1 = max(0, i - 2), min(len(hist), i + 3)
    centers = 0.5 * (edges[j0:j1] + edges[j0 + 1:j1 + 1])
    return float(np.average(centers, weights=hist[j0:j1] + 1e-9))


def detect_top_layer(frame: Frame, tol_mm: float = 40, bin_mm: float = 10, roi_frac: float = 0.25) -> TopLayer:
    """최상층 깊이 + 최상층 마스크."""
    depth = find_top_layer(frame.D, frame.valid, bin_mm=bin_mm, roi_frac=roi_frac)
    if not math.isfinite(depth):
        mask = np.zeros(frame.shape, dtype=bool)
    else:
        mask = (np.abs(frame.D - depth) < tol_mm) & frame.valid
    return TopLayer(depth_mm=depth, mask=mask, tol_mm=float(tol_mm), n_px=int(mask.sum()))


# ---------------------------------------------------------------- plane fit

def fit_plane(pts: np.ndarray):
    """pts (N,3) mm → (centroid, unit normal(-Z 향), rms residual mm). binpick_pickpoints.fit_plane 동일."""
    c = pts.mean(axis=0)
    q = pts - c
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    n = vt[2]
    if n[2] > 0:
        n = -n
    rms = float(np.sqrt(np.mean((q @ n) ** 2)))
    return c, n, rms


def _confidence(dims_mm, fill, plane_rms, sku) -> float:
    """0..1 휴리스틱. SKU 치수 오차 15% 에서 0, 채움률 0.5→0 / 1.0→1, 평면 RMS 25mm 에서 0."""
    if sku is not None:
        L0, W0 = float(sku[0]), float(sku[1])
        rel = max(abs(dims_mm[0] - L0) / L0, abs(dims_mm[1] - W0) / W0)
        s_dim = float(np.clip(1.0 - rel / 0.15, 0.0, 1.0))
        w_dim = 0.5
    else:
        s_dim, w_dim = 0.0, 0.0
    s_fill = float(np.clip((fill - 0.5) / 0.5, 0.0, 1.0))
    s_plane = float(np.clip(1.0 - plane_rms / 25.0, 0.0, 1.0))
    w_rest = 1.0 - w_dim
    return round(w_dim * s_dim + w_rest * (0.5 * s_fill + 0.5 * s_plane), 4)


# ---------------------------------------------------------------- boxes

def detect_boxes(frame: Frame, sku: Optional[Sequence[float]] = DEFAULT_SKU, tol_mm: float = 40,
                 min_area_px: int = 700, sku_tol: Optional[float] = None,
                 top_layer: Optional[TopLayer] = None) -> list:
    """최상층 박스 상면 검출 → list[Box].

    sku      : (L, W, H) mm. 신뢰도 계산 및 sku_tol 필터에 사용. None 이면 치수 항 제외.
    sku_tol  : None 이면 필터 없음(tools/binpick_topface 와 동일 검출 집합). 0.5 등 값을 주면
               L,W 가 SKU 대비 ±sku_tol 비율 밖인 후보(병합·부분 박스) 를 제거.
    top_layer: 미리 계산한 TopLayer 를 재사용할 때.
    """
    D, X, Y, I, valid = frame.D, frame.X, frame.Y, frame.I, frame.valid
    if top_layer is None:
        top_layer = detect_top_layer(frame, tol_mm=tol_mm)
    if not top_layer.ok:
        return []
    top_d = top_layer.depth_mm
    mask = top_layer.mask.copy()

    # 박스 간 이음새 = 깊이 그래디언트 에지 → 마스크에서 제거해 분리 유도
    Ds = cv2.medianBlur(D.astype(np.float32), 5)
    gx = cv2.Sobel(Ds, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Ds, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    mask &= grad < 70.0  # mm/px

    m8 = (mask * 255).astype(np.uint8)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, kernel3, iterations=2)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, kernel3, iterations=1)

    # ToF 강도(I) 채널 에지로 박스 이음새 절단
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
        comp = cv2.dilate((comp * 255).astype(np.uint8), kernel3, iterations=2) > 0
        comp &= (m8 > 0)
        # m8 은 MORPH_CLOSE 로 무효 픽셀 구멍이 메워져 있어, 그 픽셀의 X/Y 센티넬(8191.75)이
        # minAreaRect 에 섞이면 13m 짜리 사각형이 나와 종횡비 필터에서 정상 박스가 탈락한다.
        # (셀 트윈에서 발견 — 실측 30프레임에서도 4개 박스가 이 경로로 누락되고 있었다.)
        comp &= frame.valid
        if comp.sum() < min_area_px:
            continue
        ys, xs = np.nonzero(comp)
        pts_mm = np.stack([X[comp], Y[comp]], axis=-1).astype(np.float32)
        rect_mm = cv2.minAreaRect(pts_mm)
        (w_mm, h_mm) = rect_mm[1]
        rect_px = cv2.minAreaRect(np.stack([xs, ys], axis=-1).astype(np.float32))
        rect_area = max(rect_px[1][0] * rect_px[1][1], 1)
        fill = comp.sum() / rect_area
        aspect = max(w_mm, h_mm) / max(min(w_mm, h_mm), 1)
        if fill < 0.6 or aspect > 3.5:
            continue
        dims = (round(float(max(w_mm, h_mm)), 1), round(float(min(w_mm, h_mm)), 1))
        if sku is not None and sku_tol is not None:
            if not (abs(dims[0] - sku[0]) <= sku_tol * sku[0] and abs(dims[1] - sku[1]) <= sku_tol * sku[1]):
                continue

        pts3 = np.stack([X[comp], Y[comp], D[comp]], axis=-1).astype(np.float64)
        c, nrm, rms = fit_plane(pts3)
        tilt = float(np.degrees(np.arccos(min(abs(float(nrm[2])), 1.0))))
        (cx, cy), (rw, rh), ang = rect_px
        boxes.append(Box(
            id=len(boxes),
            center_mm=(round(float(c[0]), 1), round(float(c[1]), 1), round(float(c[2]), 1)),
            dims_mm=dims,
            tilt_deg=round(tilt, 2),
            normal=(round(float(nrm[0]), 4), round(float(nrm[1]), 4), round(float(nrm[2]), 4)),
            rect_px=((float(cx), float(cy)), (float(rw), float(rh)), float(ang)),
            confidence=_confidence(dims, float(fill), rms, sku),
            depth_mm=round(float(np.median(D[comp])), 1),
            area_px=int(comp.sum()),
            center_px=(int(round(xs.mean())), int(round(ys.mean()))),
            plane_rms_mm=round(rms, 2),
            fill=round(float(fill), 3),
        ))
    return boxes
