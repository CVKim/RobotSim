# -*- coding: utf-8 -*-
"""대차(카트) 견인 고리 검출 + 대차 고정 좌표계 — tools/hook_analysis_v2.py 의 프레임 단위 이식.

빈피킹(탑다운 카메라, 격자 박스)과 달리 카메라가 대차 데크를 **비스듬히(법선 대비 약 40°)** 내려다본다.
그래서 좌표계를 카메라가 아니라 **대차 구조물**에서 뽑아야 한다:

  1. 데크 플레이트 평면 RANSAC + 2차면 잔차 정련(ToF 어안 워프 보정)  ->  plane frame (u, v, h)
  2. 높이 밴드 [12, 150] mm 안에서 '밝은' 성분 = 고리 벽 (레일·고리 윗면은 어둡다) -> 시드
  3. 시드 주변 log(I) Otsu 재분할 -> 벽 + 크라운 클라우드 -> wall_top = (u 중앙, v 중앙값, h p98)
  4. 대차 프레임: 플레이트 림 라인(v 원점·yaw) + 왼쪽 레일 안쪽 벽(u 원점)   p_cart = R @ p_cam + t

출력 hook_cart_mm 은 세션 4개(카메라 높이 400~463 mm, 대차 실이동 21~52 mm)에서 **3.85 mm RMS(면내 3.1 mm)** 로
반복됐다(results/hook_repeatability.json 의 cart_frame 모드). 공개 수치 3.2 mm 는 이 구조 프레임을 초기값으로
기준 세션 데크에 ICP 정련(icp_deck)한 값이며, 세션 간 정합이라 프레임 단위 런타임인 이 모듈에는 넣지 않았다.
알고리즘·상수는 tools/hook_analysis_v2.py 와 동일하고 tests/test_real_cart.py 가 그 JSON 과의 수치 파리티를 검사한다.
(검토에서 고친 것: 림 창에 잘린 빈·이미지 경계 빈 제거 + |yaw|>2° 면 회전 좌표에서 재추출 — 원본은 yaw 6° 이상에서
잘린 빈이 만든 가짜 수평선에 붙었다. 실측 4세션은 |yaw|<0.4° 라 결과가 같다.)

상태: OK | NO_PLANE | NO_HOOK | NO_CART_FRAME  (뒤 둘은 평면 정보는 채운 채 반환)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2
import numpy as np

from .frame import Frame

# ---- 상수 (tools/hook_analysis.py · hook_analysis_v2.py 와 동일) ------------------
MAX_RANGE_MM = 1200.0
RANSAC_ITERS = 500
RANSAC_THRESH = 8.0
NORMAL_MAX_TILT_DEG = 65.0
QUAD_REFINE_BAND = 30.0
PLATE_INLIER_MM = 8.0
RNG_SEED = 42

H_MIN, H_MAX = 12.0, 150.0
WALL_I_RATIO = 0.35
CENTRAL_R_PX = 160.0
MIN_WALL_PX = 80
WALL_U_PRIOR_MM, WALL_U_SIGMA_MM = 45.0, 25.0
WALL_U_MAX_MM = 120.0
WALL_H_SPAN_MIN_MM = 50.0
CENTER_SIGMA_PX = 250.0
WIN_U_PAD_MM, WIN_V_BACK_MM, WIN_V_FRONT_MM = 15.0, 60.0, 30.0
CROWN_V_TOL_MM = 12.0
CROWN_BAND_MM = 10.0
CART_RIM_HALF_U_MM = 300.0
CART_RAIL_V_MM = (175.0, 105.0)
MIN_POINTS = 2000


# ---- 평면 -----------------------------------------------------------------------

def _fit_plane_svd(pts):
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    return n, float(n @ c)


def ransac_plate_plane(pts, iters=RANSAC_ITERS, thresh=RANSAC_THRESH,
                       max_tilt_deg=NORMAL_MAX_TILT_DEG, seed=RNG_SEED):
    """데크 플레이트 평면 RANSAC. 반환 (n, d): n·p = d, n 은 카메라 쪽, 카메라 높이 = -d > 0."""
    rng = np.random.default_rng(seed)
    sc = pts[rng.choice(len(pts), size=min(len(pts), 60000), replace=False)]
    cos_max = np.cos(np.deg2rad(max_tilt_deg))
    best = (None, None, -1)
    for _ in range(iters):
        p0, p1, p2 = pts[rng.choice(len(pts), size=3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn
        if abs(n[1]) < cos_max:
            continue
        d = float(n @ p0)
        cnt = int(np.count_nonzero(np.abs(sc @ n - d) < thresh))
        if cnt > best[2]:
            best = (n, d, cnt)
    if best[0] is None:
        raise RuntimeError("RANSAC failed: no plate plane candidate")
    n, d = best[0], best[1]
    for _ in range(2):
        inl = np.abs(pts @ n - d) < thresh
        n, d = _fit_plane_svd(pts[inl])
    if d > 0:
        n, d = -n, -d
    return n, d


def plane_frame(n, d):
    """평면 좌표계: origin = 카메라 발점, u = 카메라 X 투영, v = n × u, h = 법선(카메라 쪽 +)."""
    origin = d * n
    u = np.array([1.0, 0.0, 0.0]) - n[0] * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return origin, u, v


def quad_refine_height(uu, vv, hh, band=QUAD_REFINE_BAND):
    sel = np.abs(hh) < band
    coef = np.zeros(6)
    Af = np.stack([np.ones(len(uu)), uu, vv, uu ** 2, uu * vv, vv ** 2], axis=1)
    hq, mad = hh, 0.0
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(Af[sel], hh[sel], rcond=None)
        hq = hh - Af @ coef
        mad = 1.4826 * np.median(np.abs(hq[sel] - np.median(hq[sel])))
        sel = np.abs(hq) < max(3 * mad, 6.0)
    return hq, coef, float(mad)


def warp_correct_points(P_valid, frame, quad_coef, clip_mm=40.0):
    """플레이트 2차면 잔차를 광선 방향 깊이 오차로 보고 3D 점 보정."""
    origin, u_ax, v_ax, n = frame
    rel = P_valid - origin
    u, v = rel @ u_ax, rel @ v_ax
    Af = np.stack([np.ones(len(u)), u, v, u * u, u * v, v * v], axis=1)
    e_h = np.clip(Af @ quad_coef, -clip_mm, clip_mm)
    norm = np.linalg.norm(P_valid, axis=1)
    ray = P_valid / np.maximum(norm, 1e-6)[:, None]
    cosr = ray @ n
    cosr = np.where(np.abs(cosr) < 0.2, np.sign(cosr + 1e-12) * 0.2, cosr)
    return P_valid - (e_h / cosr)[:, None] * ray


def plane_to_cam(frame, u, v, h):
    origin, u_ax, v_ax, n = frame
    return origin + u * u_ax + v * v_ax + h * n


def cam_to_plane(frame, p):
    origin, u_ax, v_ax, n = frame
    rel = np.asarray(p, np.float64) - origin
    return np.array([rel @ u_ax, rel @ v_ax, rel @ n])


def _otsu(vals, bins=256):
    vals = np.asarray(vals, np.float64)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-9:
        return lo
    hist, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    mid = 0.5 * (edges[:-1] + edges[1:])
    p = hist / max(hist.sum(), 1)
    w0, mu = np.cumsum(p), np.cumsum(p * mid)
    sb = (mu[-1] * w0 - mu) ** 2 / (w0 * (1.0 - w0) + 1e-12)
    return float(mid[int(np.argmax(sb))])


def _robust_line(x, y, rounds=4):
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


def _r3(x):
    return tuple(round(float(a), 2) for a in np.asarray(x, np.float64).ravel())


# ---- 결과 ------------------------------------------------------------------------

@dataclass
class CartResult:
    status: str                                   # OK | NO_PLANE | NO_HOOK | NO_CART_FRAME
    reason: str = ""
    n_valid: int = 0
    valid_frac: float = 0.0
    latency_ms: float = 0.0
    plane: Optional[dict] = None                  # normal_toward_camera, d_mm, camera_height_mm, tilt_deg, plate_mad_mm, n_inliers
    hook_cam_mm: Optional[tuple] = None           # wall_top, 카메라 좌표
    hook_plane_mm: Optional[tuple] = None         # (u, v, h)
    hook_cart_mm: Optional[tuple] = None          # 대차 프레임 (u' from left rail, v' from rim, h)
    R_cart: Optional[list] = None                 # 3x3, p_cart = R @ p_cam + t  (mm)
    t_cart: Optional[list] = None
    cart: dict = field(default_factory=dict)      # rim_yaw_deg, rim_fit_mad_mm, rail_edge_mad_mm, rail_gap_mm ...
    wall: dict = field(default_factory=dict)      # area_px, u_extent_mm, h_range_mm, v_spread_mad_mm
    deck_extent_cart_mm: Optional[tuple] = None   # (u_min, u_max, v_min, v_max) 플레이트 inlier, 대차 프레임
    n_candidates: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def log_line(self) -> str:
        d = self.to_dict()
        d.pop("R_cart", None)
        d.pop("t_cart", None)
        return json.dumps(d, ensure_ascii=False)


# ---- 메인 ------------------------------------------------------------------------

def analyze_cart(frame: Frame, return_masks: bool = False):
    """Frame -> CartResult (+ masks dict, return_masks=True).

    masks: plate, band, wall, wallcrown, rim, rail (bool HxW) — 오버레이·디버그용.
    """
    t0 = time.perf_counter()
    D = frame.D.astype(np.float64)
    I = frame.I.astype(np.float64)
    H, W = D.shape
    vm = frame.valid & (D < MAX_RANGE_MM)
    n_valid = int(vm.sum())
    res = CartResult(status="NO_PLANE", n_valid=n_valid, valid_frac=n_valid / float(H * W))
    masks = {}

    def _done(r):
        r.latency_ms = round((time.perf_counter() - t0) * 1e3, 1)
        return (r, masks) if return_masks else r

    if n_valid < MIN_POINTS:
        res.reason = f"only {n_valid} valid points within {MAX_RANGE_MM:.0f} mm"
        return _done(res)

    P = np.stack([frame.X, frame.Y, frame.D], axis=-1).astype(np.float64)
    pts = P[vm]
    try:
        n, d = ransac_plate_plane(pts)
    except RuntimeError as e:
        res.reason = str(e)
        return _done(res)
    origin, u_ax, v_ax = plane_frame(n, d)
    pf = (origin, u_ax, v_ax, n)
    rel = pts - origin
    uu, vv, hh = rel @ u_ax, rel @ v_ax, pts @ n - d
    hq, quad_coef, plate_mad = quad_refine_height(uu, vv, hh)
    P_corr = np.full_like(P, np.nan)
    P_corr[vm] = warp_correct_points(pts, pf, quad_coef)
    rel_c = P_corr[vm] - origin
    height = np.full((H, W), np.nan)
    height[vm] = hq
    U = np.full((H, W), np.nan)
    U[vm] = rel_c @ u_ax
    V = np.full((H, W), np.nan)
    V[vm] = rel_c @ v_ax
    plate_px = vm & (np.abs(height) < PLATE_INLIER_MM)
    band = vm & (height >= H_MIN) & (height <= H_MAX)
    masks.update(plate=plate_px, band=band)
    res.plane = dict(normal_toward_camera=_r3(n), d_mm=round(float(d), 1),
                     camera_height_mm=round(float(-d), 1),
                     tilt_deg=round(float(np.rad2deg(np.arccos(abs(n[1])))), 1),
                     plate_mad_mm=round(plate_mad, 2), n_inliers=int(plate_px.sum()))
    res.status = "NO_HOOK"

    # 2. 강도 기반 벽 후보
    yy, xx = np.mgrid[:H, :W]
    central = np.hypot(xx - W / 2.0, yy - H / 2.0) < CENTRAL_R_PX
    if not (plate_px & central).any():
        res.reason = "no plate pixels in image centre"
        return _done(res)
    i_plate = float(np.median(I[plate_px & central]))
    bright = band & (I > WALL_I_RATIO * i_plate)
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
        cands.append((area * float(np.exp(-dc / CENTER_SIGMA_PX)) * size_w, lab))
    cands.sort(reverse=True)
    res.n_candidates = len(cands)
    if not cands:
        res.reason = "no hook wall candidate in height band"
        return _done(res)
    seed = labels == cands[0][1]

    # 3. 국소 정련 (log I Otsu) -> 벽, 벽+크라운
    su, sv = U[seed], V[seed]
    u_lo, u_hi = np.percentile(su, 3), np.percentile(su, 97)
    v_c0 = float(np.median(sv))
    win = band & (U > u_lo - WIN_U_PAD_MM) & (U < u_hi + WIN_U_PAD_MM) \
        & (V > v_c0 - WIN_V_BACK_MM) & (V < v_c0 + WIN_V_FRONT_MM)
    log_i = np.log1p(I)
    thr_local = _otsu(log_i[win])
    wall_open = cv2.morphologyEx((win & (log_i > thr_local)).astype(np.uint8), cv2.MORPH_OPEN, k3, iterations=1)
    nl, wl = cv2.connectedComponents(wall_open, connectivity=8)
    best, best_ov = 0, 0
    for lab in range(1, nl):
        ov = int(((wl == lab) & seed).sum())
        if ov > best_ov:
            best, best_ov = lab, ov
    if best == 0:
        # 전경 성분이 없거나(창 안 강도가 균일해 Otsu 분리 불가) 시드와 겹치는 성분이 없음.
        # 라벨 0(배경) 을 벽으로 잡으면 전체 화면이 벽이 되어 뒤에서 빈 배열 percentile 로 죽는다 (검토에서 발견).
        res.reason = "hook wall lost in local refinement (no bright component overlapping the seed)"
        return _done(res)
    wall = wl == best
    if wall.sum() < MIN_WALL_PX // 2:
        res.reason = "hook wall lost in local refinement"
        return _done(res)
    v_wall = float(np.median(V[wall]))
    wall &= np.abs(V - v_wall) < CROWN_V_TOL_MM
    if wall.sum() < MIN_WALL_PX // 2:
        res.reason = "hook wall too thin after flying-pixel filter"
        return _done(res)
    v_wall = float(np.median(V[wall]))
    near_v = band & (np.abs(V - v_wall) < CROWN_V_TOL_MM) & (U > u_lo - WIN_U_PAD_MM) & (U < u_hi + WIN_U_PAD_MM)
    ng, gl = cv2.connectedComponents((near_v | wall).astype(np.uint8), connectivity=8)
    keep = np.zeros(ng, bool)
    keep[np.unique(gl[wall])] = True
    keep[0] = False
    wallcrown = keep[gl]
    topface = band & ~wallcrown & (U > u_lo - WIN_U_PAD_MM) & (U < u_hi + WIN_U_PAD_MM) \
        & (V < v_wall - CROWN_V_TOL_MM) & (V > v_wall - 120.0)
    masks.update(wall=wall, wallcrown=wallcrown)

    wu, wv, wh = U[wall], V[wall], height[wall]
    u_c = 0.5 * float(np.percentile(wu, 3) + np.percentile(wu, 97))
    v_c = float(np.median(wv))
    h_top = float(np.percentile(height[wallcrown], 98))
    hook_cam = plane_to_cam(pf, u_c, v_c, h_top)
    res.hook_plane_mm = _r3((u_c, v_c, h_top))
    res.hook_cam_mm = _r3(hook_cam)
    res.wall = dict(area_px=int(wall.sum()), wallcrown_px=int(wallcrown.sum()),
                    u_extent_mm=round(float(np.percentile(wu, 97) - np.percentile(wu, 3)), 1),
                    v_spread_mad_mm=round(float(1.4826 * np.median(np.abs(wv - v_c))), 2),
                    h_range_mm=(round(float(np.percentile(wh, 2)), 1), round(h_top, 1)),
                    centre_px=tuple(int(round(x)) for x in np.argwhere(wall).mean(axis=0)[::-1]))
    res.status = "NO_CART_FRAME"

    # 4. 대차 프레임
    try:
        cf = _cart_frame(U, V, height, plate_px, band, topface, pf, u_c, v_c)
    except RuntimeError as e:
        res.reason = str(e)
        return _done(res)
    R_cart, t_cart = cf.pop("R_cart"), cf.pop("t_cart")
    masks.update(rim=cf.pop("rim_px"), rail=cf.pop("rail_px"))
    res.cart = cf
    res.R_cart = [[round(float(x), 6) for x in row] for row in R_cart]
    res.t_cart = [round(float(x), 3) for x in t_cart]
    res.hook_cart_mm = _r3(R_cart @ hook_cam + t_cart)
    pc = (R_cart @ P_corr[plate_px].T).T + t_cart
    res.deck_extent_cart_mm = _r3((np.percentile(pc[:, 0], 1), np.percentile(pc[:, 0], 99),
                                   np.percentile(pc[:, 1], 1), np.percentile(pc[:, 1], 99)))
    res.status, res.reason = "OK", "hook + cart frame"
    return _done(res)


def _rim_edge_bins(U, V, sel, v_hi, border):
    """플레이트 픽셀 sel 을 u 4 mm 빈으로 나눠 빈별 p99 v = 림 에지 후보.

    창 경계(v_hi)에 잘린 빈과 에지 픽셀이 이미지 경계인 빈은 버린다. 잘린 빈들은 정확히 v_hi 에 놓인 '가짜 수직선'을
    만들어 강건 직선 피팅을 yaw≈0 으로 끌어당겼고(합성 yaw 8° 에서 25~31 mm 오차, 상태는 OK — 검토에서 발견),
    이미지 경계 빈은 경계의 기울기를 림으로 오인해 음의 yaw 를 0.5~1.4° 과소 추정했다."""
    uu, vv, bb = U[sel], V[sel], border[sel]
    ub = np.floor(uu / 4.0).astype(np.int64)
    eu, ev = [], []
    for bidx in np.unique(ub):
        m = ub == bidx
        if m.sum() < 5:
            continue
        p99 = float(np.percentile(vv[m], 99))
        if p99 >= v_hi - 1.0:
            continue
        if bb[m][vv[m] >= p99 - 1.0].any():
            continue
        eu.append((bidx + 0.5) * 4.0)
        ev.append(p99)
    return np.asarray(eu), np.asarray(ev)


def _fit_rim(U, V, plate_px, u_c, v_c, border):
    """림 라인 피팅 (주어진 좌표계에서). 반환 (yaw, rim_v, n_bins, n_inliers, mad, sel)."""
    sel = plate_px & (np.abs(U - u_c) < CART_RIM_HALF_U_MM) & (np.abs(U - u_c) > 40) \
        & (V > v_c - 200) & (V < v_c + 30)
    eu, ev = _rim_edge_bins(U, V, sel, v_c + 30, border)
    if len(eu) < 8:
        raise RuntimeError("cart_frame: plate rim edge not found")
    a, b, inl, mad = _robust_line(eu, ev)
    if inl.sum() < max(8, 0.5 * len(eu)):
        raise RuntimeError(f"cart_frame: rim line unreliable ({int(inl.sum())}/{len(eu)} bins)")
    yaw = float(np.arctan(a))
    c, s = np.cos(-yaw), np.sin(-yaw)
    rim_v = float(np.median(s * eu[inl] + c * ev[inl]))
    return yaw, rim_v, int(len(eu)), int(inl.sum()), float(mad), sel


def _cart_frame(U, V, height, plate_px, band, topface, frame, u_c, v_c):
    """구조 기반 대차 프레임: 림 라인(v', yaw) + 왼쪽 레일 안쪽 벽(u'). p_cart = R_cart @ p_cam + t_cart."""
    origin, u_ax, v_ax, n = frame
    out = {}
    border = np.zeros(U.shape, bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    yaw, rim_v, nb, ni, mad, sel = _fit_rim(U, V, plate_px, u_c, v_c, border)
    if abs(np.rad2deg(yaw)) > 2.0:
        # 림 창(v_c−200 ~ v_c+30)은 회전 전 좌표라 |yaw| 가 크면 먼 빈이 창에 잘린다. 1차 yaw 로 회전한 좌표에서
        # 다시 추출하면 창이 림과 나란해진다. 2D 회전은 더해지므로 최종 yaw = yaw1 + yaw2, rim_v 는 2차 좌표계 값.
        c1, s1 = np.cos(-yaw), np.sin(-yaw)
        U1, V1 = c1 * U - s1 * V, s1 * U + c1 * V
        yaw2, rim_v, nb, ni, mad, sel = _fit_rim(U1, V1, plate_px, c1 * u_c - s1 * v_c, s1 * u_c + c1 * v_c, border)
        yaw = yaw + yaw2
    c, s = np.cos(-yaw), np.sin(-yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    out.update(rim_yaw_deg=round(float(np.rad2deg(yaw)), 3), rim_v_plane=round(rim_v, 2),
               rim_bins=nb, rim_inlier_bins=ni, rim_fit_mad_mm=round(mad, 2))
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
        ra, rb, rinl, rmad = _robust_line(edges_v, edges_u)
        if rinl.sum() < 3:                        # 빈 inlier 로 median 이 NaN 이 되는 것을 막는다
            continue
        rails[side] = dict(u=float(np.median(edges_u[rinl])), bins=int(len(edges_u)), inl=int(rinl.sum()),
                           mad=rmad, yaw=float(np.rad2deg(np.arctan(ra))), mask=rail)
    if "left" not in rails:
        raise RuntimeError("cart_frame: left rail inner wall not found")
    rail_u = rails["left"]["u"]
    out.update(rail_u_plane=round(rail_u, 2), rail_bins=rails["left"]["bins"],
               rail_inlier_bins=rails["left"]["inl"], rail_edge_mad_mm=round(rails["left"]["mad"], 2),
               rail_yaw_vs_rim_deg=round(rails["left"]["yaw"], 3),
               hook_in_cart_mm=(round(u_cp - rail_u, 2), round(v_cp - rim_v, 2)))
    if "right" in rails:
        out.update(right_rail_u_plane=round(rails["right"]["u"], 2),
                   right_rail_edge_mad_mm=round(rails["right"]["mad"], 2),
                   rail_gap_mm=round(rails["right"]["u"] - rail_u, 2))
    M = np.stack([u_ax, v_ax, n])
    R_cart = Rz @ M
    t_cart = -R_cart @ origin - np.array([rail_u, rim_v, 0.0])
    if not (np.all(np.isfinite(R_cart)) and np.all(np.isfinite(t_cart))):
        raise RuntimeError("cart_frame: non-finite transform")
    out["R_cart"], out["t_cart"] = R_cart, t_cart
    out["rim_px"] = sel & (np.abs(Vp - rim_v) < 3.0)
    rail_px = rails["left"]["mask"] & (np.abs(Up - rail_u) < 4.0)
    if "right" in rails:
        rail_px |= rails["right"]["mask"] & (np.abs(Up - rails["right"]["u"]) < 4.0)
    out["rail_px"] = rail_px
    return out
