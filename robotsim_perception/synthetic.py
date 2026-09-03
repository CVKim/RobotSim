# -*- coding: utf-8 -*-
"""합성 ToF 프레임 생성기 (테스트·데모용). 실제 카메라 모델(fx≈fy≈500px, cx≈320, cy≈240)을 따른다.

바닥은 기본적으로 기울여서(깊이가 여러 히스토그램 빈에 분산) 박스 상면이 최상층 피크가 되게 한다.
평평한 바닥이 ROI 를 지배하면 실제 알고리즘도 바닥을 최상층으로 잡는다(원 검출기의 가정).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .frame import Frame, SENTINEL_D_VALUE, SENTINEL_XY_VALUE

DEFAULT_INTRINSICS = dict(fx=500.0, fy=500.0, cx=320.0, cy=240.0)


@dataclass
class SynthBox:
    center_xy_mm: tuple          # 상면 중심 (X, Y) 카메라 좌표 mm
    size_mm: tuple = (293.0, 219.0)   # (L, W)
    depth_mm: float = 2970.0     # 상면 중심 깊이
    yaw_deg: float = 0.0         # 이미지 평면 내 회전
    slope_x: float = 0.0         # 깊이 기울기 dZ/dX (tan(tilt)) — X 방향
    slope_y: float = 0.0         # dZ/dY
    intensity: float = 900.0


def make_frame(boxes: Sequence[SynthBox], shape=(480, 640), floor_depth_mm: float = 3253.0,
               floor_tilt_mm=(200.0, 500.0), floor: str = "tilted", floor_intensity: float = 250.0,
               noise_mm: float = 0.0, seed: int = 0, intrinsics: Optional[dict] = None,
               invalid_rect_px=None) -> Frame:
    """합성 Frame.

    floor: 'tilted' (기본, 깊이가 floor_tilt_mm 범위로 선형 변화) | 'flat' | 'invalid' (센티넬)
    noise_mm: 깊이 가우시안 노이즈 σ (X/Y 는 노이즈 깊이로 재투영)
    invalid_rect_px: (u0, v0, u1, v1) 영역을 센티넬 무효 픽셀로 덮어씀
    """
    K = dict(DEFAULT_INTRINSICS, **(intrinsics or {}))
    h, w = shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    ru = (uu - K["cx"]) / K["fx"]   # 정규화 광선 방향 x/z
    rv = (vv - K["cy"]) / K["fy"]

    if floor == "invalid":
        D = np.full(shape, SENTINEL_D_VALUE, dtype=np.float64)
        valid_floor = np.zeros(shape, dtype=bool)
    else:
        D = np.full(shape, float(floor_depth_mm), dtype=np.float64)
        if floor == "tilted":
            D += floor_tilt_mm[0] * (uu - K["cx"]) / w + floor_tilt_mm[1] * (vv - K["cy"]) / h
        valid_floor = np.ones(shape, dtype=bool)
    I = np.full(shape, float(floor_intensity), dtype=np.float64)

    for b in boxes:
        # 상면 평면: Z = z0 + sx*(X - x0) + sy*(Y - y0), X = ru*Z, Y = rv*Z  → Z = (z0 - sx*x0 - sy*y0) / (1 - sx*ru - sy*rv)
        x0, y0 = b.center_xy_mm
        z0 = b.depth_mm
        denom = 1.0 - b.slope_x * ru - b.slope_y * rv
        Z = (z0 - b.slope_x * x0 - b.slope_y * y0) / denom
        Xp, Yp = ru * Z, rv * Z
        th = math.radians(b.yaw_deg)
        dx, dy = Xp - x0, Yp - y0
        lx = dx * math.cos(th) + dy * math.sin(th)
        ly = -dx * math.sin(th) + dy * math.cos(th)
        L, W = b.size_mm
        inside = (np.abs(lx) <= L / 2) & (np.abs(ly) <= W / 2) & (Z > 0)
        D = np.where(inside, Z, D)
        I = np.where(inside, float(b.intensity), I)
        valid_floor |= inside

    if noise_mm > 0:
        rng = np.random.default_rng(seed)
        D = D + rng.normal(0.0, noise_mm, size=shape) * valid_floor
    X = ru * D
    Y = rv * D
    inval = ~valid_floor
    if invalid_rect_px is not None:
        u0, v0, u1, v1 = invalid_rect_px
        inval[v0:v1, u0:u1] = True
    D[inval] = SENTINEL_D_VALUE
    X[inval] = SENTINEL_XY_VALUE
    Y[inval] = SENTINEL_XY_VALUE
    I[inval] = 0.0
    return Frame(X.astype(np.float32), Y.astype(np.float32), D.astype(np.float32), I.astype(np.float32),
                 source="synthetic")


def write_session(frame: Frame, session_dir) -> None:
    """Frame 을 세션 폴더 규약(2_tof_X.mim …)으로 저장 (tifffile). load_frame 으로 재로드 가능."""
    import tifffile
    from pathlib import Path
    from .frame import CHANNEL_FILES
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    for key, name in CHANNEL_FILES.items():
        tifffile.imwrite(str(session_dir / name), getattr(frame, key))
