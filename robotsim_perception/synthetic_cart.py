# -*- coding: utf-8 -*-
"""합성 대차 장면 (테스트·ROS 데모용, 회사 데이터 불필요) — 레이캐스팅.

실측 대차 세션의 기하를 따른다: 카메라가 데크 플레이트를 법선 대비 약 40° 비스듬히 본다
(plane normal toward camera ≈ (0.03, 0.64, −0.77), 카메라 높이 400~460 mm). 장면은 **평면 좌표 (u, v, h)** 로
정의하고 축정렬 박스들로 만든다:

  플레이트  h∈[−20,0], v < rim_v  (앞쪽 수직면 = 림)        밝음 (I≈10k)
  고리      u∈hook_u±15, v∈[rim_v−40, rim_v], h∈[0,130]      카메라를 향한 벽 밝음(8k), 윗면 어두움(1.5k)
  레일 2개  안쪽 벽 u = rail_u (왼쪽), rail_u + rail_gap (오른쪽), h∈[0,60]     어두움(2.5k)
  앞쪽 몸체 박스 3개 (림 앞, 플레이트보다 낮음), 바닥 h=−700

절대 정답: 고리 wall_top = (hook_u, rim_v, 130) 평면좌표, 대차 프레임에서 (hook_u − rail_u, 0, 130).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .frame import Frame, SENTINEL_D_VALUE, SENTINEL_XY_VALUE
from .cart import plane_frame, plane_to_cam
from .synthetic import DEFAULT_INTRINSICS

HOOK_H_MM = 130.0
HOOK_HALF_U_MM = 15.0
HOOK_DEPTH_MM = 40.0
RAIL_H_MM = 60.0
RAIL_W_MM = 40.0


def make_cart_frame(hook_u_mm: float = -80.0, rim_v_mm: float = -285.0, rail_u_mm: float = -116.0,
                    rail_gap_mm: float = 281.0, cam_height_mm: float = 420.0,
                    normal_cam=(0.03, 0.64, -0.77), noise_mm: float = 2.5, seed: int = 0,
                    shape=(480, 640), intrinsics: Optional[dict] = None, cart_yaw_deg: float = 0.0):
    """합성 Frame + 정답 dict.

    cart_yaw_deg: 대차 구조물(림·레일·고리)을 평면 안에서 회전 — 대차 프레임 yaw 추정 검사용.
    반환 gt: hook_cam_mm, hook_plane_mm, hook_cart_mm, rail_u_mm, rim_v_mm, rail_gap_mm,
             normal_cam, d_mm, cam_height_mm, frame(origin,u,v,n)
    """
    K = dict(DEFAULT_INTRINSICS, **(intrinsics or {}))
    H, W = shape
    n = np.asarray(normal_cam, np.float64)
    n = n / np.linalg.norm(n)
    d = -float(cam_height_mm)
    origin, u_ax, v_ax = plane_frame(n, d)
    M = np.stack([u_ax, v_ax, n])                     # p_plane = M @ (p_cam - origin)
    cam_p = -M @ origin                               # = (0, 0, cam_height)

    # 픽셀 광선 (깊이 스케일: p_cam = t * (ru, rv, 1) 이면 t = D)
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    ru = (uu - K["cx"]) / K["fx"]
    rv = (vv - K["cy"]) / K["fy"]
    r_cam = np.stack([ru, rv, np.ones_like(ru)], axis=-1).reshape(-1, 3)
    r_p = r_cam @ M.T                                  # 평면 좌표 광선 방향

    # 대차 구조물 (yaw 회전은 박스를 회전시키는 대신 광선/카메라를 역회전해 처리)
    th = math.radians(cart_yaw_deg)
    c, s = math.cos(th), math.sin(th)
    Rz_inv = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])   # 장면 좌표 = Rz(-yaw) @ 평면 좌표
    cam_s = Rz_inv @ cam_p
    r_s = r_p @ Rz_inv.T

    rail_r = rail_u_mm + rail_gap_mm
    boxes = [   # (lo, hi, I_side, I_top)
        ((-450.0, -1100.0, -20.0), (450.0, rim_v_mm, 0.0), 6000.0, 10000.0),                 # 플레이트
        ((hook_u_mm - HOOK_HALF_U_MM, rim_v_mm - HOOK_DEPTH_MM, 0.0),
         (hook_u_mm + HOOK_HALF_U_MM, rim_v_mm, HOOK_H_MM), 8000.0, 1500.0),                 # 고리
        ((rail_u_mm - RAIL_W_MM, -1100.0, 0.0), (rail_u_mm, rim_v_mm - 50.0, RAIL_H_MM), 2500.0, 2000.0),   # 왼쪽 레일
        ((rail_r, -1100.0, 0.0), (rail_r + RAIL_W_MM, rim_v_mm - 50.0, RAIL_H_MM), 2500.0, 2000.0),        # 오른쪽 레일
        ((-300.0, rim_v_mm + 10.0, -320.0), (-60.0, rim_v_mm + 160.0, -150.0), 3000.0, 2600.0),  # 앞쪽 몸체 박스
        ((-40.0, rim_v_mm + 5.0, -320.0), (140.0, rim_v_mm + 130.0, -190.0), 2800.0, 2400.0),
        ((160.0, rim_v_mm + 20.0, -320.0), (380.0, rim_v_mm + 180.0, -230.0), 3200.0, 2200.0),
        ((-3000.0, -3000.0, -720.0), (3000.0, 3000.0, -700.0), 800.0, 800.0),                # 바닥
    ]
    N = r_s.shape[0]
    t_best = np.full(N, np.inf)
    I_best = np.zeros(N)
    eps = 1e-9
    for lo, hi, i_side, i_top in boxes:
        lo, hi = np.asarray(lo), np.asarray(hi)
        t_enter = np.full(N, -np.inf)
        t_exit = np.full(N, np.inf)
        axis_enter = np.zeros(N, np.int64)
        for k in range(3):
            rk = r_s[:, k]
            safe = np.where(np.abs(rk) < eps, eps, rk)
            t1 = (lo[k] - cam_s[k]) / safe
            t2 = (hi[k] - cam_s[k]) / safe
            tmin, tmax = np.minimum(t1, t2), np.maximum(t1, t2)
            parallel = np.abs(rk) < eps
            inside = (cam_s[k] >= lo[k]) & (cam_s[k] <= hi[k])
            tmin = np.where(parallel, np.where(inside, -np.inf, np.inf), tmin)
            tmax = np.where(parallel, np.where(inside, np.inf, -np.inf), tmax)
            upd = tmin > t_enter
            axis_enter = np.where(upd, k, axis_enter)
            t_enter = np.maximum(t_enter, tmin)
            t_exit = np.minimum(t_exit, tmax)
        hit = (t_exit > t_enter) & (t_enter > 1.0) & (t_enter < t_best)
        top = (axis_enter == 2) & (r_s[:, 2] < 0)
        I_best = np.where(hit, np.where(top, i_top, i_side), I_best)
        t_best = np.where(hit, t_enter, t_best)

    valid = np.isfinite(t_best)
    D = np.where(valid, t_best, 0.0)
    if noise_mm > 0:
        rng = np.random.default_rng(seed)
        D = D + rng.normal(0.0, noise_mm, size=N) * valid
    D = D.reshape(H, W)
    valid = valid.reshape(H, W)
    I = I_best.reshape(H, W)
    X, Y = ru * D, rv * D
    D[~valid], X[~valid], Y[~valid], I[~valid] = SENTINEL_D_VALUE, SENTINEL_XY_VALUE, SENTINEL_XY_VALUE, 0.0
    frame = Frame(X.astype(np.float32), Y.astype(np.float32), D.astype(np.float32), I.astype(np.float32),
                  source="synthetic_cart")

    # 정답 (장면 좌표 -> 평면 좌표 -> 카메라 좌표)
    Rz = Rz_inv.T
    hook_s = np.array([hook_u_mm, rim_v_mm, HOOK_H_MM])
    hook_p = Rz @ hook_s
    hook_cam = plane_to_cam((origin, u_ax, v_ax, n), *hook_p)
    gt = dict(hook_cam_mm=tuple(float(x) for x in hook_cam), hook_plane_mm=tuple(float(x) for x in hook_p),
              hook_cart_mm=(hook_u_mm - rail_u_mm, 0.0, HOOK_H_MM), rail_u_mm=rail_u_mm, rim_v_mm=rim_v_mm,
              rail_gap_mm=rail_gap_mm, cart_yaw_deg=cart_yaw_deg, normal_cam=tuple(float(x) for x in n),
              d_mm=d, cam_height_mm=float(cam_height_mm), frame=(origin, u_ax, v_ax, n))
    return frame, gt
