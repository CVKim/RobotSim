# -*- coding: utf-8 -*-
"""대차 견인 고리(hook) 검출 v2 - 높이 밴드 + 강도(I) 채널 분할 + 강건 중심 추정.

v1(hook_analysis.py)의 문제:
  * 12-100mm 높이 밴드만으로 클러스터링 -> 고리와 레일이 플라잉 픽셀로 연결되면
    (세션 126020310141923) 강한 오프닝으로 분리 -> 고리 하부가 깎여 중심이 ~25mm 편향.
  * 중심 = 클러스터 픽셀 평균 -> 어느 부분(벽/윗면)이 얼마나 포함되느냐에 따라 흔들림.

v2 관찰 (4세션 컬럼 프로파일 분석):
  * 고리 = 플레이트 위 수직 벽(카메라를 향한 면, v 일정, h 12->~125mm) + 상단 크라운
    + 뒤쪽(카메라 반대쪽)의 어두운 윗면(h~90-115mm, I~1000-1400).
  * 강도: 플레이트 I~7k-19k, 고리 벽 I~5k-12k(금속이지만 카메라 정면이라 밝음),
    레일/고리 윗면/틈새 I~1k-3k. -> 밴드 안에서 "밝은 것" = 고리 벽, 레일/윗면과 분리됨.
  * 어두운 윗면은 저강도라 깊이가 세션마다 20-50mm씩 튐(485477에서 47mm 갭)
    -> 중심 추정에서 제외.

파이프라인 (세션당):
  1. v1과 동일한 평면 RANSAC + 2차면 정련 (함수 재사용)
  2. 밴드 h in [12,150] & I > ratio*median(I_plate_central) -> 오프닝 1회 -> 연결성분
     -> 점수(면적, 중앙근접, u폭 사전, h 스팬>=50mm) 최대 성분 = 고리 벽 시드
  3. 시드 주변 평면좌표 윈도우 안에서 log I Otsu로 벽 재분할(세션별 노출 적응),
     벽+크라운 클라우드 = 밴드 & |v - v_wall| < 12mm & 시드와 연결
  4. 중심 추정기:
       wall_top      : (u=(p3+p97)/2 of 벽 u, v=median 벽 v, h=p98 벽+크라운 h)  <- 기본
       wall_centroid : 벽 픽셀 3D 평균 (v1식, 비교용)
       crown         : h >= h_top-10mm 점들의 3D 중앙값
       template_icp  : 기준 세션 벽+크라운 클라우드에 ICP 정합 -> 기준 wall_top을 역변환
  5. 반복성 (대차 실이동 제거 후 4/4 세션 RMS): 정렬 모드 4종을 v1 중심(hook_results.json)과
     v2 추정기 모두에 적용해 비교.
       raw        : 정렬 없음
       icp_global : hook_icp.py 그대로 (전체 장면 ICP). 하부 몸체/박스가 세션마다 달라 비강체 +
                    어안 워프 -> 고리 위치에서 8-13mm 정렬 오차 (정렬 후 고리 클라우드 NN 잔차로 확인)
       cart_frame : 구조 기반 대차 프레임 (플레이트 평면 h + 림 라인 v/yaw + 왼쪽 레일 안쪽 벽 u), ICP 없음
       icp_deck   : cart_frame 으로 초기화한 뒤 고리 주변 300mm 데크(플레이트+레일, 고리 박스 제외,
                    워프 보정 클라우드) 만으로 게이트 12->5mm ICP 정련. 고리를 정렬에 쓰지 않으므로
                    정렬 후 고리 클라우드 NN 잔차(1.5-2.8mm)가 정렬 품질의 독립 검증.
     + 기준 세션 4가지 모두에 대해 RMS 재계산 (기준 선택 민감도), 기준 평면 프레임 축별/면내 RMS.

주요 결과 (2026-09-03, 4세션):
  * v1 중심: raw 22.7 / icp_global 17.6 / icp_deck 2.9mm.  v2 wall_top: 23.8 / 18.7 / 3.2mm
    (기준 세션 바꿔도 3.0-3.2), 면내(u,v) RMS 1.8mm (v1 2.5mm).
  * 이전 로그의 "141923 세션 ~25mm 바이어스" 는 대부분 전체 장면 ICP 정렬 오차였음. v1 의 141923 중심은
    벽 대비 v 방향으로 다른 세션보다 ~5mm 만 치우쳐 있었고(v1-v2 오프셋 -11 vs -5.6~-6.7mm), v2 는 이를 제거.
  * 남은 3D 오차의 최대 성분은 h(크라운 높이 128->134mm 가 카메라 높이 400->463mm 에 따라 단조 증가,
    시점 의존 바이어스). 도킹용 면내 위치는 ~2mm 수준.

실행:  /d/anaconda/python E:/Robot_Sim/tools/hook_analysis_v2.py
출력:  E:/Robot_Sim/explore/hook/v2_*.png, hook_results_v2.json (콘솔 ASCII만)
"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from mim_loader import load_session, valid_mask  # noqa: E402
from hook_analysis import (ransac_plate_plane, plane_frame,  # noqa: E402
                           quad_refine_height, MAX_RANGE_MM, PLATE_INLIER_MM)
from hook_icp import cloud as scene_cloud, icp as rigid_icp  # noqa: E402

try:
    from local_paths import DAECHA_DIR
except ImportError:
    import os
    DAECHA_DIR = os.environ["DAECHA_DIR"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = Path(r"E:\Robot_Sim\explore\hook")

# ---- 파라미터 ----------------------------------------------------------------
H_MIN, H_MAX = 12.0, 150.0        # 고리 높이 밴드 (크라운 ~125-135mm 포함)
WALL_I_RATIO = 0.35               # 벽 후보: I > ratio * median(플레이트 중앙부 I)
CENTRAL_R_PX = 160.0              # 플레이트 기준 강도를 재는 화면 중앙 반경
MIN_WALL_PX = 80
WALL_U_PRIOR_MM, WALL_U_SIGMA_MM = 45.0, 25.0   # 벽 폭 사전 (스펙 30-60mm)
WALL_U_MAX_MM = 120.0
WALL_H_SPAN_MIN_MM = 50.0         # 벽은 플레이트에서 크라운까지 높이 스팬이 커야 함
CENTER_SIGMA_PX = 250.0
WIN_U_PAD_MM, WIN_V_BACK_MM, WIN_V_FRONT_MM = 15.0, 60.0, 30.0   # 국소 정련 윈도우
CROWN_V_TOL_MM = 12.0             # 벽+크라운: |v - v_wall| 허용치 (윗면 제외)
CROWN_BAND_MM = 10.0              # crown 추정: h >= h_top - 10
LOCAL_ICP_R_MM = 300.0            # 데크 ICP: 고리 중심 반경 (평면좌표)
LOCAL_ICP_VOXEL = 4.0
DECK_H_MM = (-8.0, 150.0)         # 데크 구조(플레이트+레일+림) 높이 범위; 아래 몸체/박스는 비강체라 제외
DECK_HOOK_EXCL_U_MM = 28.0        # 정렬에서 고리 자체 제외 박스 (|u-u_c|)
DECK_HOOK_EXCL_V_MM = (140.0, 20.0)   # (v_c - 140, v_c + 20)
DECK_ICP_GATES = (12.0, 8.0, 5.0)  # 카트 프레임 초기화 후 정련 게이트 (coarse-to-fine)
CART_RIM_HALF_U_MM = 300.0        # 림 라인 피팅에 쓰는 u 범위 (고리 중심 기준 +-)
CART_RAIL_V_MM = (175.0, 105.0)   # 레일 안쪽 벽 측정 v 구간: rim - 175 .. rim - 105 (고리 비가림)


# ---- 유틸 ---------------------------------------------------------------------
def otsu_threshold(vals, bins=256):
    vals = np.asarray(vals, np.float64)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-9:
        return lo
    hist, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    mid = 0.5 * (edges[:-1] + edges[1:])
    p = hist / max(hist.sum(), 1)
    w0 = np.cumsum(p)
    mu = np.cumsum(p * mid)
    sb = (mu[-1] * w0 - mu) ** 2 / (w0 * (1.0 - w0) + 1e-12)
    return float(mid[int(np.argmax(sb))])


def plane_to_cam(frame, u, v, h):
    origin, u_ax, v_ax, n = frame
    return origin + u * u_ax + v * v_ax + h * n


def cam_to_plane(frame, p):
    origin, u_ax, v_ax, n = frame
    rel = np.asarray(p, np.float64) - origin
    return np.array([rel @ u_ax, rel @ v_ax, rel @ n])


def r3(x):
    return [round(float(a), 2) for a in np.asarray(x).ravel()]


def warp_correct_points(P_valid, frame, quad_coef, clip_mm=40.0):
    """플레이트 2차면 잔차(e_h = quad(u,v))를 '광선 방향 깊이 오차'로 보고 3D 점을 보정.

    h_meas - h_true = delta * (ray . n)  ->  delta = e_h / (ray . n),  p_corr = p - delta * ray.
    보정 후 높이는 v1의 quad-refined 높이와 정확히 일치하고, 면내(u,v)도 같은 비율로 당겨진다.
    """
    origin, u_ax, v_ax, n = frame
    rel = P_valid - origin
    u, v = rel @ u_ax, rel @ v_ax
    Af = np.stack([np.ones(len(u)), u, v, u * u, u * v, v * v], axis=1)
    e_h = np.clip(Af @ quad_coef, -clip_mm, clip_mm)
    norm = np.linalg.norm(P_valid, axis=1)
    ray = P_valid / np.maximum(norm, 1e-6)[:, None]
    cosr = ray @ n
    cosr = np.where(np.abs(cosr) < 0.2, np.sign(cosr + 1e-12) * 0.2, cosr)
    delta = e_h / cosr
    return P_valid - delta[:, None] * ray


def structure_features(U, V, height, plate_px, band, u_c, v_c):
    """고리 주변 대차 구조 특징(플레이트 림 v, 좌/우 레일 안쪽 에지 u)을 고리 기준 상대값으로.

    세션 간 이 값들이 일정하면 고리가 플레이트에 대해 강체라는 증거 (정렬 오차와 검출 오차 분리용).
    """
    out = {}
    # 좌표 배치: 카메라 발점(v=0)은 플레이트 앞쪽 하부 몸체 위에 있고, 플레이트는 v < 림 (~v_c) 에
    # 펼쳐짐. 고리 벽은 림(플레이트의 +v 에지)에 붙어 카메라(+v)를 향하고, 윗면은 -v 로 뻗음.
    # 림: 고리 양옆(|u-u_c| 20~120mm) 플레이트 픽셀의 최대 v (p99.5)
    sel = plate_px & (np.abs(U - u_c) > 20) & (np.abs(U - u_c) < 120) \
        & (V > v_c - 150) & (V < v_c + 50)
    if sel.sum() > 50:
        out["rim_dv_mm"] = round(float(np.percentile(V[sel], 99.5) - v_c), 2)
    # 레일: 플레이트 위(v_c-140 ~ v_c-30) h 30-120mm 픽셀; 왼쪽 레일의 안쪽(고리쪽) 에지 = p97 u
    rail = band & (height > 30) & (height < 120) & (V > v_c - 140) & (V < v_c - 30)
    left = rail & (U > u_c - 150) & (U < u_c - 20)
    right = rail & (U > u_c + 20) & (U < u_c + 150)
    if left.sum() > 50:
        out["left_rail_inner_du_mm"] = round(float(np.percentile(U[left], 97) - u_c), 2)
    if right.sum() > 50:
        out["right_rail_inner_du_mm"] = round(float(np.percentile(U[right], 3) - u_c), 2)
    return out


def robust_line(x, y, rounds=4):
    """y = a x + b 강건 피팅 (MAD 재가중, 3-MAD 밖 제외). 반환 (a, b, inlier mask, mad)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    sel = np.ones(len(x), bool)
    a, b, mad = 0.0, float(np.median(y)), 0.0
    for _ in range(rounds):
        if sel.sum() < 4:
            break
        a, b = np.polyfit(x[sel], y[sel], 1)
        r = y - (a * x + b)
        mad = 1.4826 * np.median(np.abs(r[sel] - np.median(r[sel])))
        sel = np.abs(r) < max(3.0 * mad, 2.0)
    return float(a), float(b), sel, float(mad)


def cart_frame(U, V, height, plate_px, band, topface, frame, u_c, v_c):
    """구조 기반 대차 고정 프레임 (ICP 없이): 플레이트(h) + 림 라인(v', yaw) + 왼쪽 레일 안쪽 벽(u').

    림 = 플레이트 inlier 의 +v 경계 (u-빈 4mm 별 p99 v) 강건 직선 -> yaw 와 v 원점.
    레일 = 림 뒤(-v) 플레이트 위 h 15-80mm 점 중 고리 왼쪽 레일의 안쪽(+u) 수직벽 (v-빈별 p98 u) -> u 원점.
    (오른쪽 레일 안쪽 벽도 있으면 함께 측정: 레일 간격, 레일쌍 중심 기준 고리 u.)
    반환: dict(stats..., R_cart, t_cart, rim_px, rail_px)  with  p_cart = R_cart @ p_cam + t_cart.
    """
    origin, u_ax, v_ax, n = frame
    out = {}
    # 1) 림 에지 점
    sel = plate_px & (np.abs(U - u_c) < CART_RIM_HALF_U_MM) & (np.abs(U - u_c) > 40) \
        & (V > v_c - 200) & (V < v_c + 30)
    uu, vv = U[sel], V[sel]
    ub = np.floor(uu / 4.0).astype(np.int64)
    eu, ev = [], []
    for bidx in np.unique(ub):
        m = ub == bidx
        if m.sum() >= 5:
            eu.append((bidx + 0.5) * 4.0)
            ev.append(float(np.percentile(vv[m], 99)))
    eu, ev = np.asarray(eu), np.asarray(ev)
    a, b, inl, mad = robust_line(eu, ev)
    yaw = float(np.arctan(a))
    c, s = np.cos(-yaw), np.sin(-yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rim_v = float(np.median((s * eu[inl] + c * ev[inl])))       # 회전 후 v' (u' 무관)
    out.update(rim_yaw_deg=round(float(np.rad2deg(yaw)), 3), rim_v_plane=round(rim_v, 2),
               rim_bins=int(len(eu)), rim_inlier_bins=int(inl.sum()), rim_fit_mad_mm=round(mad, 2))
    # 2) 레일 안쪽 수직벽 (회전된 좌표에서). 고리 어두운 윗면(h>=85)과 u 로 7mm 밖에 안 떨어져 있고,
    #    고리(h~130)가 카메라에서 본 레일 안쪽 벽을 v in (v_c-100, v_c) 구간에서 가리므로
    #    h 15-80mm, topface 제외, 고리 비가림 구간 v in (rim-175, rim-105) 만 사용
    #    -> v-빈별 극값 u = 레일 안쪽 벽. (플레이트 경계 기반 대안은 윈도우 아티팩트라 폐기)
    Up = c * U - s * V
    Vp = s * U + c * V
    u_cp, v_cp = c * u_c - s * v_c, s * u_c + c * v_c
    base = band & (height > 15) & (height < 80) & ~topface \
        & (Vp > rim_v - CART_RAIL_V_MM[0]) & (Vp < rim_v - CART_RAIL_V_MM[1])
    rails = {}
    for side, lo, hi, pct in [("left", -170.0, -18.0, 98), ("right", 40.0, 320.0, 2)]:
        rail = base & (Up > u_cp + lo) & (Up < u_cp + hi)
        ru, rv = Up[rail], Vp[rail]
        vb = np.floor(rv / 4.0).astype(np.int64)
        edges_u, edges_v = [], []
        for bidx in np.unique(vb):
            m = vb == bidx
            if m.sum() >= 3:
                edges_u.append(float(np.percentile(ru[m], pct)))
                edges_v.append((bidx + 0.5) * 4.0)
        if len(edges_u) < 6:
            continue
        edges_u, edges_v = np.asarray(edges_u), np.asarray(edges_v)
        ra, rb, rinl, rmad = robust_line(edges_v, edges_u)    # u_edge = ra * v + rb
        rails[side] = dict(u=float(np.median(edges_u[rinl])), bins=int(len(edges_u)),
                           inl=int(rinl.sum()), mad=rmad,
                           yaw=float(np.rad2deg(np.arctan(ra))), mask=rail)
    if "left" not in rails:
        raise RuntimeError("cart_frame: left rail inner wall not found")
    rail_u = rails["left"]["u"]
    out.update(rail_u_plane=round(rail_u, 2), rail_bins=rails["left"]["bins"],
               rail_inlier_bins=rails["left"]["inl"], rail_edge_mad_mm=round(rails["left"]["mad"], 2),
               rail_yaw_vs_rim_deg=round(rails["left"]["yaw"], 3),
               hook_in_cart_mm=[round(u_cp - rail_u, 2), round(v_cp - rim_v, 2)])
    if "right" in rails:
        out.update(right_rail_u_plane=round(rails["right"]["u"], 2),
                   right_rail_bins=rails["right"]["bins"],
                   right_rail_edge_mad_mm=round(rails["right"]["mad"], 2),
                   rail_gap_mm=round(rails["right"]["u"] - rail_u, 2),
                   hook_u_from_rail_pair_center_mm=round(
                       u_cp - 0.5 * (rails["right"]["u"] + rail_u), 2))
    # 3) 3D 변환: p_cart = Rz(-yaw) @ M @ (p - origin) - [rail_u, rim_v, 0]
    M = np.stack([u_ax, v_ax, n])
    R_cart = Rz @ M
    t_cart = -R_cart @ origin - np.array([rail_u, rim_v, 0.0])
    out["R_cart"], out["t_cart"] = R_cart, t_cart
    # 시각화용: 림 에지 픽셀(림 라인 3mm 이내 플레이트 픽셀), 레일 안쪽 벽 픽셀
    out["rim_px"] = sel & (np.abs(Vp - rim_v) < 3.0)
    rail_px = rails["left"]["mask"] & (np.abs(Up - rail_u) < 4.0)
    if "right" in rails:
        rail_px |= rails["right"]["mask"] & (np.abs(Up - rails["right"]["u"]) < 4.0)
    out["rail_px"] = rail_px
    return out


# ---- 세션 분석 ----------------------------------------------------------------
def analyze_session_v2(sess_dir):
    sess = load_session(sess_dir)
    D, I = sess["D"], sess["I"].astype(np.float64)
    H, W = D.shape
    vm = valid_mask(sess) & (D < MAX_RANGE_MM)
    P = np.stack([sess["X"], sess["Y"], D], axis=-1).astype(np.float64)
    pts = P[vm]

    # 1. 평면 (v1 재사용)
    n, d = ransac_plate_plane(pts)
    origin, u_ax, v_ax = plane_frame(n, d)
    frame = (origin, u_ax, v_ax, n)
    rel = pts - origin
    uu, vv, hh = rel @ u_ax, rel @ v_ax, pts @ n - d
    hq, quad_coef, plate_mad = quad_refine_height(uu, vv, hh)
    # 워프 보정 클라우드: 높이는 hq 와 일치, 면내(u,v)도 광선 방향으로 함께 보정 (고리 부근 <1mm)
    P_corr = np.full_like(P, np.nan)
    P_corr[vm] = warp_correct_points(pts, frame, quad_coef)
    rel_c = P_corr[vm] - origin
    height = np.full((H, W), np.nan)
    height[vm] = hq
    U = np.full((H, W), np.nan)
    U[vm] = rel_c @ u_ax
    V = np.full((H, W), np.nan)
    V[vm] = rel_c @ v_ax

    plate_px = vm & (np.abs(height) < PLATE_INLIER_MM)
    band = vm & (height >= H_MIN) & (height <= H_MAX)

    # 2. 강도 기반 벽 후보
    yy, xx = np.mgrid[:H, :W]
    central = np.hypot(xx - W / 2.0, yy - H / 2.0) < CENTRAL_R_PX
    i_plate = float(np.median(I[plate_px & central]))
    i_thr = WALL_I_RATIO * i_plate
    bright = band & (I > i_thr)
    k3 = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(bright.astype(np.uint8), cv2.MORPH_OPEN, k3, iterations=1)
    n_lab, labels, stats, cents = cv2.connectedComponentsWithStats(opened, connectivity=8)

    cands = []
    for lab in range(1, n_lab):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < MIN_WALL_PX:
            continue
        m = labels == lab
        cu, ch = U[m], height[m]
        ext_u = float(np.percentile(cu, 97) - np.percentile(cu, 3))
        span_h = float(np.percentile(ch, 98) - np.percentile(ch, 2))
        if ext_u > WALL_U_MAX_MM or span_h < WALL_H_SPAN_MIN_MM:
            continue
        dc = float(np.hypot(cents[lab][0] - W / 2.0, cents[lab][1] - H / 2.0))
        size_w = float(np.exp(-0.5 * ((ext_u - WALL_U_PRIOR_MM) / WALL_U_SIGMA_MM) ** 2))
        score = area * float(np.exp(-dc / CENTER_SIGMA_PX)) * size_w
        cands.append(dict(label=lab, area_px=area, ext_u_mm=round(ext_u, 1),
                          span_h_mm=round(span_h, 1), dist_center_px=round(dc, 1),
                          median_I=round(float(np.median(I[m]))), score=round(score, 1)))
    cands.sort(key=lambda c: -c["score"])
    if not cands:
        raise RuntimeError(f"{sess_dir.name}: no wall candidate")
    seed = labels == cands[0]["label"]

    # 3. 국소 정련: 시드 주변 평면좌표 윈도우 안에서 log I Otsu (벽 vs 레일/윗면)
    su, sv = U[seed], V[seed]
    u_lo, u_hi = np.percentile(su, 3), np.percentile(su, 97)
    v_c0 = float(np.median(sv))
    win = band & (U > u_lo - WIN_U_PAD_MM) & (U < u_hi + WIN_U_PAD_MM) \
        & (V > v_c0 - WIN_V_BACK_MM) & (V < v_c0 + WIN_V_FRONT_MM)
    log_i = np.log1p(I)
    thr_local = otsu_threshold(log_i[win])
    wall_raw = win & (log_i > thr_local)
    wall_open = cv2.morphologyEx(wall_raw.astype(np.uint8), cv2.MORPH_OPEN, k3, iterations=1)
    nl, wl = cv2.connectedComponents(wall_open, connectivity=8)
    best, best_ov = 0, -1
    for lab in range(1, nl):
        ov = int(((wl == lab) & seed).sum())
        if ov > best_ov:
            best, best_ov = lab, ov
    wall = wl == best
    # 플라잉 픽셀 제거: 벽은 v=const 인 수직면 -> |v - median v| > tol 인 픽셀 제외
    v_wall = float(np.median(V[wall]))
    wall &= np.abs(V - v_wall) < CROWN_V_TOL_MM

    # 벽+크라운 클라우드: 밴드 & |v - v_wall| < tol & 벽과 8-연결
    v_wall = float(np.median(V[wall]))
    near_v = band & (np.abs(V - v_wall) < CROWN_V_TOL_MM) & (U > u_lo - WIN_U_PAD_MM) \
        & (U < u_hi + WIN_U_PAD_MM)
    grow_src = (near_v | wall).astype(np.uint8)
    ng, gl = cv2.connectedComponents(grow_src, connectivity=8)
    keep = np.zeros(ng, bool)
    keep[np.unique(gl[wall])] = True
    keep[0] = False
    wallcrown = keep[gl]
    topface = band & ~wallcrown & (U > u_lo - WIN_U_PAD_MM) & (U < u_hi + WIN_U_PAD_MM) \
        & (V < v_wall - CROWN_V_TOL_MM) & (V > v_wall - 120.0)

    # 4. 중심 추정기
    wu, wv, wh = U[wall], V[wall], height[wall]
    ch = height[wallcrown]
    u_c = 0.5 * float(np.percentile(wu, 3) + np.percentile(wu, 97))
    v_c = float(np.median(wv))
    h_top = float(np.percentile(ch, 98))
    est = {}
    est["wall_top"] = dict(plane=[u_c, v_c, h_top])
    est["wall_centroid"] = dict(cam=P_corr[wall].mean(axis=0))
    crown_sel = wallcrown & (height >= h_top - CROWN_BAND_MM)
    est["crown"] = dict(cam=np.median(P_corr[crown_sel], axis=0))
    for k, e in est.items():
        if "plane" in e:
            e["cam"] = plane_to_cam(frame, *e["plane"])
        else:
            e["plane"] = cam_to_plane(frame, e["cam"])
        e["cam"], e["plane"] = r3(e["cam"]), r3(e["plane"])

    result = dict(
        session=sess_dir.name,
        n_valid=int(vm.sum()),
        plane=dict(normal_toward_camera=r3(n), d_mm=round(float(d), 1),
                   camera_height_above_plate_mm=round(float(-d), 1),
                   tilt_from_camera_y_axis_deg=round(float(np.rad2deg(np.arccos(abs(n[1])))), 1),
                   plate_residual_mad_mm=round(plate_mad, 2),
                   n_plate_inliers=int(plate_px.sum())),
        intensity=dict(plate_central_median=round(i_plate), wall_thr_global=round(i_thr),
                       wall_thr_local_otsu=round(float(np.expm1(thr_local))),
                       wall_median=round(float(np.median(I[wall]))),
                       topface_median=(round(float(np.median(I[topface]))) if topface.any() else None)),
        wall=dict(area_px=int(wall.sum()), seed_area_px=int(seed.sum()),
                  wallcrown_px=int(wallcrown.sum()), crown_px=int(crown_sel.sum()),
                  u_extent_mm=round(float(np.percentile(wu, 97) - np.percentile(wu, 3)), 1),
                  v_spread_mad_mm=round(float(1.4826 * np.median(np.abs(wv - v_c))), 2),
                  h_range_mm=[round(float(np.percentile(wh, 2)), 1), round(h_top, 1)],
                  centroid_px=[round(float(x), 1) for x in np.argwhere(wall).mean(axis=0)[::-1]]),
        estimates=est,
        n_candidates=len(cands),
        candidates=cands[:5],
    )
    result["structure_rel_mm"] = structure_features(U, V, height, plate_px, band, u_c, v_c)
    cf = cart_frame(U, V, height, plate_px, band, topface, frame, u_c, v_c)
    R_cart, t_cart = cf.pop("R_cart"), cf.pop("t_cart")
    rim_px, rail_px = cf.pop("rim_px"), cf.pop("rail_px")
    result["cart_frame"] = cf
    for e in est.values():
        e["cart"] = r3(R_cart @ np.asarray(e["cam"]) + t_cart)
    # 데크(플레이트+레일) 정렬용 클라우드: 고리 박스 제외, 고리 중심 반경 내, 워프 보정, 복셀 4mm
    hookbox = (np.abs(U - u_c) < DECK_HOOK_EXCL_U_MM) & (V > v_c - DECK_HOOK_EXCL_V_MM[0]) \
        & (V < v_c + DECK_HOOK_EXCL_V_MM[1]) & (height > 3)
    deck = vm & (height > DECK_H_MM[0]) & (height < DECK_H_MM[1]) & ~hookbox \
        & (np.hypot(U - u_c, V - v_c) < LOCAL_ICP_R_MM)
    result["deck_px"] = int(deck.sum())
    viz = dict(I=I, vm=vm, plate_px=plate_px, band=band, wall=wall, crown=crown_sel,
               wallcrown=wallcrown, topface=topface, seed=seed, height=height,
               rim_px=rim_px, rail_px=rail_px)
    aux = dict(frame=frame, P=P, P_corr=P_corr, wallcrown_pts=P[wallcrown],
               wallcrown_pts_corr=P_corr[wallcrown], wall_pts=P[wall],
               deck_pts_corr=voxel_down(P_corr[deck], LOCAL_ICP_VOXEL),
               R_cart=R_cart, t_cart=t_cart,
               sess=sess, vm=vm, quad_coef=quad_coef)
    return result, viz, aux


# ---- 템플릿 ICP ------------------------------------------------------------------
def template_icp_centers(results, auxs, ref_idx):
    """기준 세션의 벽+크라운 클라우드에 각 세션 클라우드를 ICP -> 기준 wall_top 역변환."""
    ref_pts = auxs[ref_idx]["wallcrown_pts_corr"]
    ref_c = np.asarray(results[ref_idx]["estimates"]["wall_top"]["cam"], np.float64)
    tree = cKDTree(ref_pts)
    for i, (res, aux) in enumerate(zip(results, auxs)):
        src = aux["wallcrown_pts_corr"]
        c_i = np.asarray(res["estimates"]["wall_top"]["cam"], np.float64)
        R, t = np.eye(3), ref_c - c_i          # 초기: wall_top 끼리 정렬
        cur = src + t
        m = np.ones(len(src), bool)
        for _ in range(40):
            dist, j = tree.query(cur, k=1)
            m = dist < 15.0
            if m.sum() < 30:
                break
            A, B = cur[m], ref_pts[j[m]]
            ca, cb = A.mean(0), B.mean(0)
            Hm = (A - ca).T @ (B - cb)
            Uu, _, Vt = np.linalg.svd(Hm)
            Ri = Vt.T @ Uu.T
            if np.linalg.det(Ri) < 0:
                Vt[2] *= -1
                Ri = Vt.T @ Uu.T
            ti = cb - Ri @ ca
            cur = cur @ Ri.T + ti
            R, t = Ri @ R, Ri @ t + ti
        dist, _ = tree.query(cur, k=1)
        rmse = float(np.sqrt(np.mean(np.minimum(dist, 15.0) ** 2)))
        c_sess = R.T @ (ref_c - t)             # p_ref = R p + t  ->  p = R^T (p_ref - t)
        ang = float(np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        res["estimates"]["template_icp"] = dict(
            cam=r3(c_sess), plane=r3(cam_to_plane(aux["frame"], c_sess)),
            cart=r3(aux["R_cart"] @ c_sess + aux["t_cart"]),
            icp_rmse_mm=round(rmse, 2), icp_rot_deg=round(ang, 2),
            n_src=int(len(src)), inlier_frac=round(float(m.mean()), 3))


# ---- 장면 ICP 정렬 (hook_icp.py 로직 재사용 + 국소 정련) --------------------------
def voxel_down(pts, voxel):
    key = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[idx]


def icp_scheduled(src, dst, gates=DECK_ICP_GATES, iters=30):
    """hook_icp.icp 를 게이트를 줄여가며 반복 (coarse-to-fine). 반환 (R, t, 최종 rmse)."""
    R, t = np.eye(3), np.zeros(3)
    cur = src.copy()
    rmse = float("nan")
    for g in gates:
        Ri, ti, rmse = rigid_icp(cur, dst, iters=iters, dist_thresh=g)
        cur = cur @ Ri.T + ti
        R, t = Ri @ R, Ri @ t + ti
    return R, t, rmse


ALIGN_MODES = ["raw", "icp_global", "cart_frame", "icp_deck"]


def scene_alignment(auxs, results, ref_idx, global_cache=None):
    """각 세션 -> 기준 세션 강체 변환.

    icp_global : hook_icp.py 그대로 (전체 장면 원본 클라우드, 복셀 8mm, 게이트 40mm)
    cart_frame : 구조 기반 (플레이트 평면 + 림 라인 + 왼쪽 레일 에지) 대차 프레임끼리 맞춤 (ICP 없음)
    icp_deck   : cart_frame 으로 초기화 후, 고리 주변 반경 내 데크 구조(플레이트+레일, h in DECK_H_MM,
                 고리 박스 제외, 워프 보정, 복셀 4mm)만으로 게이트 12->5mm ICP 정련.
                 고리를 정렬에 쓰지 않으므로 정렬 후 고리 클라우드 잔차가 정렬 품질의 독립 검증이 됨.
    """
    ref_full = scene_cloud(auxs[ref_idx]["sess"])
    dst_deck = auxs[ref_idx]["deck_pts_corr"]
    R_ref, t_ref = auxs[ref_idx]["R_cart"], auxs[ref_idx]["t_cart"]
    ident = (np.eye(3), np.zeros(3))
    out = []
    for i, aux in enumerate(auxs):
        if i == ref_idx:
            out.append(dict(T={m: ident for m in ALIGN_MODES},
                            rmse=dict(icp_global=0.0, icp_deck=0.0), motion_mm=0.0))
            continue
        key = (ref_idx, i)
        if global_cache is not None and key in global_cache:
            Rg, tg, rmse_g = global_cache[key]
        else:
            Rg, tg, rmse_g = rigid_icp(scene_cloud(aux["sess"]), ref_full)
            if global_cache is not None:
                global_cache[key] = (Rg, tg, rmse_g)
        c_i = np.asarray(results[i]["estimates"]["wall_top"]["cam"], np.float64)
        # cart_frame: p_ref = R_ref^T (R_s p + t_s - t_ref)
        Rc = R_ref.T @ aux["R_cart"]
        tc = R_ref.T @ (aux["t_cart"] - t_ref)
        Rd0, td0, rmse_d = icp_scheduled(aux["deck_pts_corr"] @ Rc.T + tc, dst_deck)
        T = {"raw": ident, "icp_global": (Rg, tg), "cart_frame": (Rc, tc),
             "icp_deck": (Rd0 @ Rc, Rd0 @ tc + td0)}
        out.append(dict(T=T, rmse=dict(icp_global=rmse_g, icp_deck=rmse_d),
                        motion_mm=float(np.linalg.norm(Rg @ c_i + tg - c_i)),
                        deck_refine_shift_mm=float(np.linalg.norm(
                            (Rd0 @ (Rc @ c_i + tc) + td0) - (Rc @ c_i + tc)))))
    return out


def apply_T(T, c):
    R, t = T
    return R @ np.asarray(c, np.float64) + t


def spread_stats(arr):
    arr = np.asarray(arr, np.float64)
    dev = arr - arr.mean(axis=0)
    return dict(
        mean=r3(arr.mean(axis=0)), std=r3(arr.std(axis=0)),
        per_session_dev_mm=[round(float(np.linalg.norm(x)), 2) for x in dev],
        max_abs_dev=r3(np.abs(dev).max(axis=0)),
        rms_3d=round(float(np.sqrt((dev ** 2).sum(axis=1).mean())), 2),
        max_pairwise_3d=round(float(max(np.linalg.norm(a - b) for i, a in enumerate(arr)
                                        for b in arr[i + 1:])), 2))


def evaluate_repeatability(centers_by_est, align):
    """centers_by_est: {name: (N,3) cam-frame}. 정렬 모드별(ALIGN_MODES) spread."""
    rep = {}
    for name, arr in centers_by_est.items():
        arr = np.asarray(arr, np.float64)
        rep[name] = {}
        for mode in ALIGN_MODES:
            ali = np.stack([apply_T(a["T"][mode], c) for c, a in zip(arr, align)])
            rep[name][mode] = spread_stats(ali)
    return rep


def hook_cloud_alignment_residual(auxs, align, ref_idx):
    """정렬 변환을 벽+크라운 클라우드에 적용 -> 기준 클라우드와의 NN 거리 (정렬 품질 검증).

    icp_deck 는 워프 보정 클라우드끼리, icp_global 은 원본 클라우드끼리 비교.
    """
    tree_raw = cKDTree(auxs[ref_idx]["wallcrown_pts"])
    tree_cor = cKDTree(auxs[ref_idx]["wallcrown_pts_corr"])
    out = {}
    for i, (aux, a) in enumerate(zip(auxs, align)):
        if i == ref_idx:
            continue
        for mode in ALIGN_MODES[1:]:
            corr = mode in ("icp_deck", "cart_frame")
            src = aux["wallcrown_pts_corr"] if corr else aux["wallcrown_pts"]
            R, t = a["T"][mode]
            dist, _ = (tree_cor if corr else tree_raw).query(src @ R.T + t, k=1)
            out.setdefault(mode, {})[aux["sess_name"]] = dict(
                median_nn_mm=round(float(np.median(dist)), 2),
                p90_nn_mm=round(float(np.percentile(dist, 90)), 2))
    return out


# ---- 시각화 -------------------------------------------------------------------
def project_center_px(aux, center_cam):
    """카메라 좌표 3D 점 -> 가장 가까운 유효 픽셀 (표시용)."""
    P, vm = aux["P"], aux["vm"]
    ys, xs = np.nonzero(vm)
    sub = P[vm]
    m = np.linalg.norm(sub - center_cam, axis=1) < 60
    if not m.any():
        return None
    j = np.argmin(np.linalg.norm(sub[m] - center_cam, axis=1))
    return [float(xs[m][j]), float(ys[m][j])]


def save_overlay(viz, res, out_path):
    I, vm = viz["I"], viz["vm"]
    b = I.copy()
    b[~np.isfinite(b)] = 0
    lo, hi = np.percentile(b[vm], [1, 99])
    g = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.cvtColor((g * 200).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def blend(mask, color, a):
        img[mask] = ((1 - a) * img[mask] + a * np.array(color)).astype(np.uint8)

    blend(viz["plate_px"], (150, 150, 150), 0.4)
    blend(viz["band"], (0, 140, 255), 0.5)          # 밴드: 주황
    blend(viz["topface"], (255, 120, 0), 0.5)       # 제외된 윗면: 파랑
    blend(viz["wallcrown"], (0, 200, 255), 0.6)     # 벽+크라운 클라우드: 노랑-주황
    blend(viz["wall"], (0, 0, 255), 0.65)           # 벽: 빨강
    img[viz["crown"]] = (0, 255, 255)               # 크라운 밴드: 노랑
    img[viz["rim_px"]] = (0, 255, 0)                # 카트 프레임 림 라인: 초록
    img[viz["rail_px"]] = (255, 0, 255)             # 카트 프레임 레일 에지: 마젠타
    H, W = I.shape
    for key, col in [("wall_top", (255, 255, 255)), ("wall_centroid", (255, 0, 255)),
                     ("crown", (0, 255, 0)), ("template_icp", (255, 255, 0))]:
        px = res["estimates"].get(key, {}).get("px")
        if px is not None:
            cv2.drawMarker(img, (int(round(px[0])), int(round(px[1]))), col,
                           cv2.MARKER_CROSS, 14, 1)
    cv2.putText(img, f"{res['session']} v2 red=wall yellow=crown orange=band blue=topface(excl) "
                "green=rim magenta=rail-edge", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    cv2.imwrite(str(out_path), img)
    cx, cy = [int(round(c)) for c in res["wall"]["centroid_px"]]
    r = 80
    y0, y1, x0, x1 = max(cy - r, 0), min(cy + r, H), max(cx - r, 0), min(cx + r, W)
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.putText(crop, "markers: white=wall_top magenta=wall_centroid green=crown cyan=template_icp",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    cv2.imwrite(str(out_path).replace(".png", "_crop.png"), crop)


def save_scatter(results, rep, align, out_path, est_keys):
    sessions = [r["session"][-6:] for r in results]
    fig, axes = plt.subplots(1, len(ALIGN_MODES), figsize=(5.5 * len(ALIGN_MODES), 5.5))
    for ax, mode in zip(axes, ALIGN_MODES):
        for key, mk in zip(est_keys, ["o", "s", "^", "D", "x"]):
            arr = np.array([r["estimates"][key]["cam"] for r in results], np.float64)
            arr = np.stack([apply_T(a["T"][mode], c) for c, a in zip(arr, align)])
            ax.scatter(arr[:, 0], arr[:, 1], marker=mk, s=50,
                       label=f"{key} rms3d={rep[key][mode]['rms_3d']:.1f}mm")
            if key == est_keys[0]:
                for s, (x, y) in zip(sessions, arr[:, :2]):
                    ax.annotate(s, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_title(f"v2 hook centers ({mode}) - camera X/Y (mm)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---- main ---------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions = sorted(p for p in Path(DAECHA_DIR).iterdir() if (p / "4_tof_D.mim").exists())
    print("ToF sessions:", len(sessions))

    results, vizs, auxs = [], [], []
    for sp in sessions:
        res, viz, aux = analyze_session_v2(sp)
        aux["sess_name"] = sp.name
        results.append(res)
        vizs.append(viz)
        auxs.append(aux)
        w, it = res["wall"], res["intensity"]
        e = res["estimates"]["wall_top"]
        print(f"[OK ] {sp.name} plate_I={it['plate_central_median']} thr_g={it['wall_thr_global']} "
              f"thr_loc={it['wall_thr_local_otsu']} wall_I={it['wall_median']} top_I={it['topface_median']} "
              f"| wall px={w['area_px']} (seed {w['seed_area_px']}) u_ext={w['u_extent_mm']} "
              f"v_mad={w['v_spread_mad_mm']} h={w['h_range_mm']} cand={res['n_candidates']}")
        print(f"       wall_top cam=({e['cam'][0]:.1f},{e['cam'][1]:.1f},{e['cam'][2]:.1f}) "
              f"plane=(u{e['plane'][0]:.1f}, v{e['plane'][1]:.1f}, h{e['plane'][2]:.1f})")

    ref_idx = 0    # hook_icp.py 와 동일: 첫 세션(126020310141923) 기준
    template_icp_centers(results, auxs, ref_idx)
    for res, aux in zip(results, auxs):
        ti = res["estimates"]["template_icp"]
        print(f"       template_icp {res['session'][-6:]}: rmse={ti['icp_rmse_mm']} rot={ti['icp_rot_deg']}deg "
              f"inl={ti['inlier_frac']} cam=({ti['cam'][0]:.1f},{ti['cam'][1]:.1f},{ti['cam'][2]:.1f})")
        for key in res["estimates"]:
            res["estimates"][key]["px"] = project_center_px(aux, np.asarray(res["estimates"][key]["cam"]))

    global_cache = {}
    align = scene_alignment(auxs, results, ref_idx, global_cache)
    for res, a in zip(results, align):
        cf = res["cart_frame"]
        print(f"       cart_frame {res['session'][-6:]}: rim yaw={cf['rim_yaw_deg']:.2f}deg "
              f"(bins {cf['rim_inlier_bins']}/{cf['rim_bins']}, mad {cf['rim_fit_mad_mm']}) "
              f"rail edge mad={cf['rail_edge_mad_mm']} (bins {cf['rail_inlier_bins']}/{cf['rail_bins']}, "
              f"yaw vs rim {cf['rail_yaw_vs_rim_deg']:.2f}deg) hook_in_cart(u,v)={cf['hook_in_cart_mm']} "
              f"rail_gap={cf.get('rail_gap_mm')} hook_u_pair={cf.get('hook_u_from_rail_pair_center_mm')}")
        print(f"       scene ICP {res['session'][-6:]}: rmse global={a['rmse'].get('icp_global', 0):.2f} "
              f"deck={a['rmse'].get('icp_deck', 0):.2f} motion={a['motion_mm']:.1f}mm "
              f"deck_refine_shift={a.get('deck_refine_shift_mm', 0):.2f}mm deck_px={res['deck_px']} "
              f"struct={res['structure_rel_mm']}")

    v1_path = OUT_DIR / "hook_results.json"
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    v1_centers = {s["session"]: s["hook"]["center_cam_mm"] for s in v1["sessions"]}
    centers = {"v1_cluster_mean": np.array([v1_centers[r["session"]] for r in results], np.float64)}
    est_keys = ["wall_top", "template_icp", "crown", "wall_centroid"]
    for key in est_keys:
        centers["v2_" + key] = np.array([r["estimates"][key]["cam"] for r in results], np.float64)
    rep = evaluate_repeatability(centers, align)
    resid = hook_cloud_alignment_residual(auxs, align, ref_idx)

    print("\nrepeatability RMS-3D (mm), 4/4 sessions, camera frame:")
    print(f"  {'estimator':18s}" + "".join(f"{m:>12s}" for m in ALIGN_MODES)
          + "   per-session dev / max pair (icp_deck)")
    for key in centers:
        r = rep[key]
        print(f"  {key:18s}" + "".join(f"{r[m]['rms_3d']:12.2f}" for m in ALIGN_MODES)
              + f"   {r['icp_deck']['per_session_dev_mm']}  {r['icp_deck']['max_pairwise_3d']}")
    print("hook-cloud alignment residual (NN mm vs ref):", json.dumps(resid))

    # 기준 세션 선택 민감도: 4세션 각각을 기준으로 정렬 -> 추정기/모드별 RMS
    by_ref = {}
    for ri in range(len(results)):
        al = align if ri == ref_idx else scene_alignment(auxs, results, ri, global_cache)
        rp = evaluate_repeatability(centers, al)
        by_ref[results[ri]["session"]] = {k: {m: rp[k][m]["rms_3d"] for m in ALIGN_MODES} for k in centers}
    print("\nRMS-3D by reference session (mode icp_deck | cart_frame):")
    for key in centers:
        vals_d = [by_ref[s][key]["icp_deck"] for s in by_ref]
        vals_c = [by_ref[s][key]["cart_frame"] for s in by_ref]
        print(f"  {key:18s} deck: " + " ".join(f"{v:5.2f}" for v in vals_d)
              + f"  mean={np.mean(vals_d):.2f} max={np.max(vals_d):.2f} | cart: "
              + " ".join(f"{v:5.2f}" for v in vals_c)
              + f"  mean={np.mean(vals_c):.2f} max={np.max(vals_c):.2f}")

    plane_rep = {}
    for key in est_keys:
        arr = np.array([r["estimates"][key]["plane"] for r in results], np.float64)
        plane_rep["v2_" + key] = spread_stats(arr)
    cart_rep = {}
    for key in est_keys:
        arr = np.array([r["estimates"][key]["cart"] for r in results], np.float64)
        cart_rep["v2_" + key] = spread_stats(arr)
    v1_cart = np.array([auxs[i]["R_cart"] @ centers["v1_cluster_mean"][i] + auxs[i]["t_cart"]
                        for i in range(len(results))])
    cart_rep["v1_cluster_mean"] = spread_stats(v1_cart)
    print("\ncart-frame (structural, no ICP) spread of hook center, 4/4 sessions:")
    for key, v in cart_rep.items():
        print(f"  {key:18s} std(u,v,h)={v['std']} rms3d={v['rms_3d']} maxpair={v['max_pairwise_3d']}")

    # 정렬 후 중심을 기준 세션 평면 프레임(u,v,h)으로 -> 축별 spread + 면내 2D RMS
    ref_plane = {}
    for mode in ("icp_deck", "cart_frame"):
        ref_plane[mode] = {}
        for key, arr in centers.items():
            ali = np.stack([cam_to_plane(auxs[ref_idx]["frame"], apply_T(a["T"][mode], c))
                            for c, a in zip(arr, align)])
            st = spread_stats(ali)
            dev = ali - ali.mean(axis=0)
            st["rms_uv_inplane"] = round(float(np.sqrt((dev[:, :2] ** 2).sum(axis=1).mean())), 2)
            ref_plane[mode][key] = st
    print("\nicp_deck-aligned centers in the reference plate frame: std(u,v,h) / rms_uv / rms3d")
    for key, st in ref_plane["icp_deck"].items():
        print(f"  {key:18s} std={st['std']}  rms_uv={st['rms_uv_inplane']}  rms3d={st['rms_3d']}")

    for res, viz in zip(results, vizs):
        save_overlay(viz, res, OUT_DIR / f"v2_{res['session']}_overlay.png")
    rep_for_plot = {k[3:]: v for k, v in rep.items() if k.startswith("v2_")}
    save_scatter(results, rep_for_plot, align, OUT_DIR / "v2_centers_scatter.png", est_keys)

    payload = dict(
        meta=dict(
            units="mm", date="2026-09-03", version="v2",
            camera_frame="X right+, Y down+ (image), D forward+",
            reference_session=results[ref_idx]["session"],
            method_notes=(
                "Plane RANSAC + quadratic refinement as v1. Hook wall candidates = "
                f"height band [{H_MIN},{H_MAX}]mm AND intensity > {WALL_I_RATIO} x median plate "
                "intensity (central 160px); best component by area x center-proximity x width "
                "prior (45+/-25mm) with height span >= 50mm. Local refinement: Otsu on log(I) "
                "inside a plane-coordinate window around the seed; wall+crown cloud = band "
                f"pixels within |v - v_wall| < {CROWN_V_TOL_MM}mm connected to the wall (dark top "
                "face behind the crown excluded: low intensity -> unreliable depth). Estimators: "
                "wall_top = (u mid of p3/p97, median v, p98 h), wall_centroid, crown (median of "
                "top 10mm), template_icp (ICP of wall+crown cloud onto the reference session's; "
                "reference wall_top mapped back). All 3D points are warp-corrected along the ray "
                "using the plate quadratic residual (delta = e_h / (ray.n)). Alignment modes: raw; "
                "icp_global = hook_icp.py whole-scene ICP (voxel 8mm, 40mm gate; the lower cart "
                "body/boxes differ between sessions so this has 8-13mm error at the hook); "
                "cart_frame = structural cart frame (plate plane for h, robust rim line for v/yaw, "
                "left rail inner wall for u, measured in the v-range the hook cannot occlude); "
                f"icp_deck = cart_frame init then ICP on deck points within {LOCAL_ICP_R_MM}mm of "
                f"the hook (h in {list(DECK_H_MM)}mm, hook box excluded, voxel {LOCAL_ICP_VOXEL}mm, "
                f"gates {list(DECK_ICP_GATES)}mm). Because the hook is excluded from icp_deck, the "
                "hook-cloud NN residual after alignment is an independent check of alignment "
                "quality. RMS-3D = sqrt(mean squared deviation from the 4-session mean); also "
                "recomputed with each session as reference (repeatability_rms_by_reference)."),
            params=dict(h_band_mm=[H_MIN, H_MAX], wall_i_ratio=WALL_I_RATIO,
                        central_r_px=CENTRAL_R_PX, min_wall_px=MIN_WALL_PX,
                        wall_u_prior_mm=[WALL_U_PRIOR_MM, WALL_U_SIGMA_MM],
                        wall_h_span_min_mm=WALL_H_SPAN_MIN_MM, crown_v_tol_mm=CROWN_V_TOL_MM,
                        crown_band_mm=CROWN_BAND_MM, local_icp_r_mm=LOCAL_ICP_R_MM,
                        local_icp_voxel_mm=LOCAL_ICP_VOXEL, deck_h_mm=list(DECK_H_MM),
                        deck_hook_excl_mm=[DECK_HOOK_EXCL_U_MM, list(DECK_HOOK_EXCL_V_MM)],
                        deck_icp_gates_mm=list(DECK_ICP_GATES),
                        cart_rim_half_u_mm=CART_RIM_HALF_U_MM,
                        cart_rail_v_mm=list(CART_RAIL_V_MM)),
            findings=(
                "v1 centers re-evaluated with icp_deck give 2.9mm RMS: the earlier '141923 ~25mm "
                "bias' was mostly whole-scene ICP alignment error; v1's 141923 center sits only "
                "~5mm further in v from the wall than in the other sessions. v2 wall_top: 3.2mm "
                "RMS 3D (3.0-3.2 over reference choices), 1.8mm in-plane; remaining 3D error is "
                "dominated by the crown height h_top (128->134mm) which grows monotonically with "
                "camera height (400->463mm) - a viewpoint-dependent bias of the top edge."),
        ),
        sessions=results,
        scene_alignment={r["session"]: dict(
            icp_rmse_mm={k: round(v, 2) for k, v in a["rmse"].items()},
            applied_motion_mm=round(a["motion_mm"], 1),
            R_icp_deck=[r3(row) for row in a["T"]["icp_deck"][0]],
            t_icp_deck=r3(a["T"]["icp_deck"][1]))
            for r, a in zip(results, align)},
        hook_cloud_alignment_residual=resid,
        repeatability_cam_frame=rep,
        repeatability_ref_plane_frame=ref_plane,
        repeatability_rms_by_reference=by_ref,
        repeatability_cart_frame=cart_rep,
        repeatability_plane_frame_unaligned=plane_rep,
        v1_reference=dict(raw_rms_mm=v1["repeatability"]["center_cam_mm"]["rms_3d"],
                          source="hook_results.json"),
    )
    out_json = OUT_DIR / "hook_results_v2.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved:", out_json)


if __name__ == "__main__":
    main()
