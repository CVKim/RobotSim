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

from . import geometry
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
    source: str = "detected"  # 'detected' | 'inferred' (lattice=True 에서 격자로 보완된 셀)

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

def _sess_view(frame: Frame) -> dict:
    """geometry 모듈이 받는 딕셔너리 형태로 Frame 을 노출 (복사 없음)."""
    return {"X": frame.X, "Y": frame.Y, "D": frame.D, "I": frame.I}


def _to_box(frame: Frame, b: dict, sku, box_id: int) -> Box:
    """geometry 가 낸 박스 딕셔너리 -> Box (평면 피팅으로 중심·법선·기울기 추가)."""
    D, X, Y = frame.D, frame.X, frame.Y
    pts = cv2.boxPoints(b["rect_px"]).astype(np.int32)
    poly = np.zeros(D.shape, np.uint8)
    cv2.fillPoly(poly, [pts], 1)
    comp = (poly > 0) & frame.valid & (np.abs(D - b["depth_mm"]) < 60.0)
    if comp.sum() >= 30:
        pts3 = np.stack([X[comp], Y[comp], D[comp]], axis=-1).astype(np.float64)
        c, nrm, rms = fit_plane(pts3)
    else:                       # 상면 픽셀이 거의 없는 추론 셀 — rect 중심으로 대체
        cxy = pts.mean(axis=0)
        c = np.array([float(np.median(X[comp])) if comp.any() else 0.0,
                      float(np.median(Y[comp])) if comp.any() else 0.0,
                      float(b["depth_mm"])])
        nrm, rms = np.array([0.0, 0.0, -1.0]), 0.0
    tilt = float(np.degrees(np.arccos(min(abs(float(nrm[2])), 1.0))))
    (cx, cy), (rw, rh), ang = b["rect_px"]
    dims = (float(b["dims_mm"][0]), float(b["dims_mm"][1]))
    rect_area = max(rw * rh, 1.0)
    fill = float(min(b["area_px"] / rect_area, 1.0))
    conf = b.get("confidence")
    return Box(
        id=box_id,
        center_mm=(round(float(c[0]), 1), round(float(c[1]), 1), round(float(c[2]), 1)),
        dims_mm=(round(dims[0], 1), round(dims[1], 1)),
        tilt_deg=round(tilt, 2),
        normal=(round(float(nrm[0]), 4), round(float(nrm[1]), 4), round(float(nrm[2]), 4)),
        rect_px=((float(cx), float(cy)), (float(rw), float(rh)), float(ang)),
        confidence=float(conf) if conf is not None else _confidence(dims, fill, rms, sku),
        depth_mm=round(float(b["depth_mm"]), 1),
        area_px=int(b["area_px"]),
        center_px=(int(round(cx)), int(round(cy))),
        plane_rms_mm=round(float(rms), 2),
        fill=round(fill, 3),
        source=str(b.get("source", "detected")),
    )


def detect_boxes(frame: Frame, sku: Optional[Sequence[float]] = DEFAULT_SKU, tol_mm: float = 40,
                 min_area_px: int = 700, sku_tol: Optional[float] = None,
                 top_layer: Optional[TopLayer] = None, lattice: bool = False,
                 min_confidence: float = 0.0) -> list:
    """최상층 박스 상면 검출 -> list[Box].

    알고리즘은 robotsim_perception.geometry 한 곳에만 있다 (예전에는 여기와 tools 에 복제돼
    있어 센티넬 오염 버그를 두 번 고쳐야 했다).

    lattice  : True 면 v2 — 박스별 신뢰도 + 격자 기반 결손 보완('inferred') + 층 선택 규칙.
               실측 30프레임 기준 v1 152 박스 vs **v2 167 박스**(RGB 대조로 검증된 실제 개수).
               대가는 지연(52 -> 약 340 ms/프레임).
    sku      : (L, W, H) mm. 신뢰도 계산 및 sku_tol 필터에 사용. None 이면 치수 항 제외.
    sku_tol  : None 이면 필터 없음. 0.5 등을 주면 L,W 가 SKU 대비 ±sku_tol 밖인 후보를 제거.
    top_layer: 미리 계산한 TopLayer 를 재사용할 때 (v1 경로에서만 사용).
    """
    sess = _sess_view(frame)
    if lattice:
        _, _, raw = geometry.detect_boxes_v2(sess, tol_mm=tol_mm, min_area_px=min_area_px,
                                             conf_min=min_confidence or None)
    else:
        if top_layer is None:
            top_layer = detect_top_layer(frame, tol_mm=tol_mm)
        if not top_layer.ok:
            return []
        _, _, raw = geometry._detect_boxes_at(sess, top_layer.depth_mm,
                                              tol_mm=tol_mm, min_area_px=min_area_px)
    boxes = []
    for b in raw:
        dims = (float(b["dims_mm"][0]), float(b["dims_mm"][1]))
        if sku is not None and sku_tol is not None:
            if not (abs(dims[0] - sku[0]) <= sku_tol * sku[0]
                    and abs(dims[1] - sku[1]) <= sku_tol * sku[1]):
                continue
        boxes.append(_to_box(frame, b, sku, len(boxes)))
    if min_confidence > 0:
        boxes = [b for b in boxes if b.confidence >= min_confidence]
        for i, b in enumerate(boxes):
            b.id = i
    return boxes
