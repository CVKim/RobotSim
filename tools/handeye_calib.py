# -*- coding: utf-8 -*-
"""핸드아이 캘리브레이션을 셀 트윈에서 처음부터 끝까지 돌린다 — 판을 손에 들고 여러 자세로 서서 찍고, 외참을 푼다.

    .venv\\Scripts\\python.exe tools/handeye_calib.py                 # 기본 스윕 -> results/handeye_calib.json + assets/handeye_calib.png
    .venv\\Scripts\\python.exe tools/handeye_calib.py --poses 12 --noise tof --show

왜 트윈에서 하나: 실제 셀에서는 카메라-로봇 변환의 **정답이 없다**. 트윈은 그 값을 알고 있으므로
 (1) 절차가 맞는지, (2) 자세를 몇 개나 모아야 하는지, (3) 센서 노이즈가 결과를 얼마나 흔드는지를 mm 단위로 잴 수 있다.
실제 셀로 옮길 때 바뀌는 것은 '판 포즈를 어떻게 관측하느냐' 하나뿐이고, 푸는 부분(robotsim_perception/handeye.py)은 그대로다.

절차
  1. 로봇 손에 캘리브 판(200×140 mm, 모서리에 15 mm 탭)을 단다 — sim/arm.py 의 calib_plate.
  2. 팔을 무작위 자세 N 개로 세운다. 자세마다 기울기(최대 25도)와 요를 흔든다 — 회전축이 한 방향에 몰리면 풀리지 않는다.
  3. 자세마다 ToF 프레임을 렌더해 **판의 6-DoF 포즈를 인식으로** 구한다(깊이 군집 → 평면 피팅 → 장축 → 탭으로 180도 해소).
  4. 로봇 쪽은 순기구학으로 T_base_tool 을 안다. 상대 운동으로 AX = XB 를 만들어 X = T_cam_base 를 푼다.
  5. 트윈이 아는 참값과 비교한다: 회전(도), 이동(mm), 그리고 픽 좌표가 몇 mm 밀리는지.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT))

from arm import PLATE_L, PLATE_T, PLATE_W, TAB, Arm, rot_from_approach_yaw  # noqa: E402
from cell_scene import CAM_H, DECK_H, build_xml  # noqa: E402
from cell_twin import TwinRenderer  # noqa: E402
from robotsim_perception import handeye as he  # noqa: E402
from robotsim_perception.geometry import valid_mask  # noqa: E402
from robotsim_perception.pose import invert, topdown_camera_transform  # noqa: E402

ARM_CFG = dict(base_xy=(-0.6, 0.75), pedestal_h=1.08, track_range=0.6, meshes=False, calib_plate=True)
PLATE_DIMS_MM = (PLATE_L * 1000.0, PLATE_W * 1000.0)
TAB_PHI_DEG = math.degrees(math.atan2(PLATE_W / 2 - TAB, PLATE_L / 2 - TAB))   # 판 좌표에서 탭이 있는 방위각
OUT_JSON = ROOT / "results" / "handeye_calib.json"
OUT_PNG = ROOT / "assets" / "handeye_calib.png"


# ----------------------------------------------------------------------------- 트윈
class CalibCell:
    """캘리브용 셀: 박스 없이 팔 + 판만. 물리는 쓰지 않고 관절각을 직접 세워 렌더만 한다."""

    def __init__(self, seed: int = 0):
        import mujoco
        self.mj = mujoco
        xml, _ = build_xml([], seed=seed, arm=ARM_CFG)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.arm = Arm(self.m, self.d)
        self.arm.set_q(self.arm.home)
        mujoco.mj_forward(self.m, self.d)
        self.rend = TwinRenderer(self.m)
        self.plate_bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "calib_plate")
        self.tool_bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "tool")
        self.T_base_cam = topdown_camera_transform(CAM_H * 1000.0)      # 트윈이 아는 참값 (mm)

    def set_q(self, q):
        self.arm.set_q(q)
        self.mj.mj_forward(self.m, self.d)

    def body_pose_mm(self, bid) -> np.ndarray:
        """월드(=로봇 베이스) 기준 4×4, 이동은 mm."""
        T = np.eye(4)
        T[:3, :3] = self.d.xmat[bid].reshape(3, 3).copy()
        T[:3, 3] = self.d.xpos[bid] * 1000.0
        return T

    def frame(self, rng, noise=None):
        return self.rend.frame(self.d, noise=noise, rng=rng)


# ----------------------------------------------------------------------------- 판 포즈 인식
def _plane_fit(P: np.ndarray):
    """점 (X, Y, D) 에 평면을 맞춘다 -> (법선, 중심, RMS). 법선은 카메라 쪽(-D)을 향하게 한다."""
    c = P.mean(axis=0)
    u, s, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2]
    if n[2] > 0:
        n = -n
    rms = float(np.sqrt(np.mean(((P - c) @ n) ** 2)))
    return n / np.linalg.norm(n), c, rms


def _dims_score(L, W, tol=0.12):
    """판 크기 사전(200×140 mm)에 얼마나 맞나 — 손목·레일 같은 다른 평면을 거른다."""
    l0, w0 = PLATE_DIMS_MM
    e = max(abs(L - l0) / l0, abs(W - w0) / w0)
    return float(max(0.0, 1.0 - e / tol))


def _ransac_plane(P: np.ndarray, rng, iters: int = 800, thresh: float = 6.0, min_inl: int = 150):
    """근거리 점구름에서 '판다운 평면' 을 찾는다.

    깊이 슬랩(같은 깊이끼리 묶기)은 판이 기울면 못 잡는다 — 25도 기운 200 mm 판은 깊이가 84 mm 에 걸쳐 퍼진다.
    그래서 평면 자체를 뽑고, 크기 사전(200×140)과 평면 RMS 로 손목·레일 같은 다른 표면을 거른다."""
    n_pts = len(P)
    if n_pts < min_inl:
        return None
    best = None
    for _ in range(iters):
        # 씨앗 하나를 고르고 나머지 두 점은 그 **근처**(판 크기 안)에서 고른다.
        # 무작위 3점이면 15000 점 중 판이 7% 밖에 안 되어 300번을 돌려도 판 위 3점이 뽑힐 확률이 10% 남짓이다.
        i0 = int(rng.integers(n_pts))
        p0 = P[i0]
        near_i = np.nonzero(np.linalg.norm(P - p0, axis=1) < 120.0)[0]
        if len(near_i) < 3:
            continue
        p1, p2 = P[rng.choice(near_i, 2, replace=False)]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = float(np.linalg.norm(nrm))
        if ln < 1e-6:
            continue
        nrm = nrm / ln
        if nrm[2] > 0:
            nrm = -nrm                                   # 법선은 카메라 쪽(-D)
        # 같은 평면 위에 있어도 멀리 떨어진 조각(레일·팔뚝)까지 인라이어로 세면 크기 점수가 무너진다 —
        # 씨앗 주변 250 mm(판 반대각 122 mm 의 두 배) 안으로 제한해서 본다.
        inl = (np.abs((P - p0) @ nrm) < thresh) & (np.linalg.norm(P - p0, axis=1) < 250.0)
        k = int(inl.sum())
        if k < min_inl:
            continue
        n2, c2, rms = _plane_fit(P[inl])                 # 인라이어로 다시 피팅
        u, v = _plane_axes(n2)
        Q = P[inl] - c2
        uv = np.stack([Q @ u, Q @ v], 1).astype(np.float32)
        (_, _), (w, h), _ = cv2.minAreaRect(uv)
        L, W = (w, h) if w >= h else (h, w)
        score = _dims_score(L, W, tol=0.30) * max(0.0, 1.0 - rms / 8.0)
        if score <= 0.0:
            continue
        if best is None or score > best["score"]:
            best = dict(score=score, n=n2, c=c2, rms=rms, L=L, W=W, inl=inl, uv=uv, u=u, v=v)
    return best


def _plane_axes(n: np.ndarray):
    """법선에 수직인 임의의 정규직교 두 축 (방향은 뒤에서 판의 장축·탭으로 정한다)."""
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v / np.linalg.norm(v)


def plate_pose_from_frame(f: dict, min_px: int = 120, seed: int = 0, debug: bool = False):
    """ToF 프레임 한 장 -> 카메라 좌표계의 판 포즈 T_cam_plate (mm). 못 찾으면 None.

    카메라 좌표는 이 레포 관례대로 (X, Y, D) 이고 D 가 깊이다. 판의 보이는 면이 카메라를 향하므로
    판 프레임의 +Z 는 법선의 반대(= 툴 +Z, 접근 방향)로 잡아 트윈의 판 바디 축과 같은 규약을 쓴다.
    절차: 근거리 띠 -> RANSAC 평면(크기 사전으로 선별) -> 최대 연결 요소 -> 장축 -> 탭으로 네 방향 중 선택."""
    X, Y, D = f["X"], f["Y"], f["D"]
    valid = valid_mask(f)          # 센티넬(무효 화소)을 빼는 것은 이 패키지 공통 규약이다 — 안 빼면 배경이 근거리 띠에 섞인다
    if valid.sum() < min_px:
        return None
    near = float(np.percentile(D[valid], 0.1))
    band = valid & (D < near + 450.0)                     # 판·탭·손목이 들어오는 근거리 띠 (데크는 훨씬 멀다)
    if band.sum() < min_px:
        return None
    ys, xs = np.nonzero(band)
    P = np.stack([X[band], Y[band], D[band]], 1).astype(np.float64)
    rng = np.random.default_rng(seed)
    best = _ransac_plane(P, rng, min_inl=min_px)
    if best is None or best["score"] <= 0.15:      # RANSAC 후보 단계는 느슨하게 — 최종 판정은 아래 연결 요소에서
        return ("no_plane", None) if debug else None

    # 인라이어를 이미지로 되돌려 가장 큰 연결 요소만 남긴다 (같은 평면 위의 다른 조각 제거)
    mask = np.zeros(D.shape, np.uint8)
    mask[ys[best["inl"]], xs[best["inl"]]] = 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))   # ToF 결손으로 뚫린 구멍을 메운다
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n_cc < 2:
        return ("no_cc", None) if debug else None
    c = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[c, cv2.CC_STAT_AREA] < min_px:
        return ("cc_small", None) if debug else None
    # 닫힘 연산은 인라이어가 아니던 화소도 채운다 — 무효(센티넬) 화소가 섞이면 크기가 수십 m 로 튄다(실제로 겪음).
    # 그래서 유효 마스크와 평면 근방 조건을 다시 건다.
    cc = (lab == c) & valid
    dist = np.full(D.shape, 1e9, np.float64)
    dist[cc] = np.abs((np.stack([X[cc], Y[cc], D[cc]], 1) - best["c"]) @ best["n"])
    cc &= dist < 14.0
    if cc.sum() < min_px:
        return ("cc_small", None) if debug else None
    Pc = np.stack([X[cc], Y[cc], D[cc]], 1).astype(np.float64)
    n, c3, rms = _plane_fit(Pc)
    u, v = _plane_axes(n)
    Q = Pc - c3
    uv = np.stack([Q @ u, Q @ v], 1).astype(np.float32)
    (cu, cv_), (w, h), ang = cv2.minAreaRect(uv)
    L, W = (w, h) if w >= h else (h, w)
    if _dims_score(L, W, tol=0.12) < 0.3 or rms > 12.0:  # 최종 판정: 크기 8% 안, 평면 RMS 12 mm 안(노이즈 프레임 허용)
        return (f"dims {L:.0f}x{W:.0f} rms {rms:.1f}", None) if debug else None
    th = math.radians(ang if w >= h else ang + 90.0)      # 장축 방향 (평면 좌표)
    a1 = math.cos(th) * u + math.sin(th) * v
    a1 /= np.linalg.norm(a1)
    z = -n                                                # 판 프레임 +Z = 툴 접근 방향
    a2 = np.cross(z, a1)
    a2 /= np.linalg.norm(a2)
    center = c3 + u * cu + v * cv_                        # 사각형 중심 (점 분포의 무게중심이 아니라)

    tab = _tab_direction(Pc, X, Y, D, valid, n, center, a1, a2, L, W)
    if tab is None:
        return ("no_tab", None) if debug else None
    # 방향은 장축이 정확하다. 탭은 네 방향(±장축, ±단축) 중 하나를 고르는 데만 쓴다 —
    # 탭 무게중심으로 각도를 직접 정하면 30 mm 짜리 작은 면의 잡음이 그대로 각도 오차가 된다.
    phi = math.radians(TAB_PHI_DEG)
    obs = tab / max(float(np.linalg.norm(tab)), 1e-9)
    # 장축(200:300 = 1.5)은 믿을 만하므로 남는 모호성은 180도뿐이다. 탭은 그 둘 중 하나를 고르는 데만 쓴다 —
    # 90도 이웃까지 후보에 넣으면 탭 방향이 40도만 틀려도 90도 틀린 해가 뽑힌다(초기 구현에서 실제로 발생).
    best_e, best_dot = None, -2.0
    for e1c in (a1, -a1):
        e2c = np.cross(z, e1c)
        e2c /= np.linalg.norm(e2c)
        pred = math.cos(phi) * e1c + math.sin(phi) * e2c
        dot = float(pred @ (obs[0] * a1 + obs[1] * a2))
        if dot > best_dot:
            best_dot, best_e = dot, e1c
    e1 = best_e
    e2 = np.cross(z, e1)
    e2 /= np.linalg.norm(e2)
    T = np.eye(4)
    T[:3, :3] = np.stack([e1, e2, z], axis=1)
    T[:3, 3] = center - n * (PLATE_T * 1000.0 / 2.0)      # 관측면은 판의 윗면 -> 법선 반대로 두께 절반만큼 내려야 판 중심이다
    if debug:
        return T, dict(best, L=L, W=W, rms=rms, n_px=int(cc.sum()), tab_dot=best_dot)
    return T


def _tab_direction(Pc, X, Y, D, valid, n, center, a1, a2, L, W):
    """평면에서 6~30 mm 튀어나온 점 = 탭. 판 중심에서 본 평면 내 방향 (a1, a2 좌표)."""
    P = np.stack([X[valid], Y[valid], D[valid]], 1).astype(np.float64)
    Q = P - center
    s = Q @ n                                             # 법선이 카메라 쪽이므로 +는 판보다 앞(카메라 쪽)
    inplane = Q - np.outer(s, n)
    du, dv = inplane @ a1, inplane @ a2
    keep = (s > 6.0) & (s < 30.0) & (np.abs(du) < L / 2 + 12.0) & (np.abs(dv) < W / 2 + 12.0)
    if keep.sum() < 20:
        return None
    v2 = np.array([float(du[keep].mean()), float(dv[keep].mean())])
    # 탭은 판 중심에서 (L/2-TAB, W/2-TAB) 만큼 떨어져 있다. 무게중심이 그 절반도 안 되게 나오면
    # 판 전체나 엉뚱한 돌기를 잡은 것이다 — 그런 관측은 쓰지 않는다.
    expect = math.hypot(PLATE_DIMS_MM[0] / 2 - TAB * 1000.0, PLATE_DIMS_MM[1] / 2 - TAB * 1000.0)
    if np.linalg.norm(v2) < expect * 0.4:
        return None
    return v2


# ----------------------------------------------------------------------------- 자세 생성
def sample_joint_poses(cell: CalibCell, n: int, seed: int, max_tilt_deg: float = 25.0, tries: int = 400):
    """판이 카메라를 향하는 무작위 자세 n 개(기구학만 본다). 기울기·요를 흔들어 회전축을 퍼뜨린다 —
    모든 자세가 같은 축으로 돌면 외참의 이동 성분이 결정되지 않는다."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(tries):
        if len(out) >= n:
            break
        x = rng.uniform(-0.30, 0.30)
        y = rng.uniform(-0.30, 0.30)
        z = DECK_H + rng.uniform(0.55, 1.00)
        tilt = math.radians(rng.uniform(3.0, max_tilt_deg))
        az = rng.uniform(0, 2 * math.pi)
        approach = np.array([math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), -math.cos(tilt)])
        R = rot_from_approach_yaw(approach, float(rng.uniform(-90, 90)))
        res = cell.arm.solve_ik_multi(np.array([x, y, z]), R, seeds=3, rng=rng)
        if not res.ok:
            continue
        cell.set_q(res.q)
        if cell.arm.contacts():                                   # 자기 충돌·설비 충돌 자세는 버린다
            continue
        out.append(res.q.copy())
    return out


def visible_pose_pool(cell: CalibCell, n: int = 24, seed: int = 0, tries: int = 120):
    """판이 실제로 **검출되는** 자세만 모은 풀. 현장에서도 사람이 화면을 보고 타깃이 보이는 자세에서 찍는다.

    풀을 한 번 만들어 두고 실험(자세 수·노이즈·시드)마다 부분집합을 뽑아 쓰면, 자세 수의 효과만 따로 볼 수 있다."""
    rng = np.random.default_rng(seed)
    pool, checked = [], 0
    for q in sample_joint_poses(cell, tries, seed, tries=tries * 8):
        checked += 1
        cell.set_q(q)
        if plate_pose_from_frame(cell.frame(rng, noise=None)) is not None:
            pool.append(q.copy())
            if len(pool) >= n:
                break
    return pool, checked


# ----------------------------------------------------------------------------- 한 번 실행
def run_once(cell: CalibCell, pool, n_poses: int, seed: int, noise: str | None, method: str = "park_martin",
             captures: int = 3) -> dict:
    """자세 풀에서 n_poses 개를 뽑아 캘리브 한 번을 돌린다."""
    rng = np.random.default_rng(1000 + seed)
    pick = rng.choice(len(pool), size=min(n_poses, len(pool)), replace=False)
    qs = [pool[i] for i in pick]
    cam_target, base_tool, plate_err = [], [], []
    for q in qs:
        cell.set_q(q)
        T_cam_plate = None
        for _ in range(captures):          # 결손 블롭이 판을 통째로 덮는 프레임이 있다 — 실제 캘리브도 자세마다 여러 장 찍는다
            T_cam_plate = plate_pose_from_frame(cell.frame(rng, noise=noise))
            if T_cam_plate is not None:
                break
        if T_cam_plate is None:
            continue
        T_base_plate_true = cell.body_pose_mm(cell.plate_bid)
        T_cam_plate_true = invert(cell.T_base_cam) @ T_base_plate_true
        plate_err.append(he.transform_error(T_cam_plate, T_cam_plate_true))
        cam_target.append(T_cam_plate)
        base_tool.append(cell.body_pose_mm(cell.tool_bid))
    if len(cam_target) < 3:
        return {"n_poses": n_poses, "seed": seed, "noise": noise or "none", "detected": len(cam_target),
                "ok": False, "reason": "판을 찾은 자세가 3개 미만"}
    r = he.solve_robust(cam_target, base_tool, method=method)   # 잘못 관측된 자세는 잔차로 걸러낸다
    # 픽 좌표가 몇 mm 밀리는지: 팔레트 위 격자 점들을 두 변환으로 옮겨 비교
    gx = np.linspace(-600, 600, 5)
    pts = [[x, y, (CAM_H - DECK_H - 0.283) * 1000.0] for x in gx for y in gx]
    err = he.transform_error(r["T_base_cam"], cell.T_base_cam, points_mm=pts)
    return {"n_poses": n_poses, "seed": seed, "noise": noise or "none", "method": method,
            "detected": len(cam_target), "used": len(r.get("used", cam_target)), "dropped": len(r.get("dropped", [])),
            "n_pairs": r["n_pairs"], "ok": True,
            "rot_deg": round(err["rot_deg"], 4), "trans_mm": round(err["trans_mm"], 3),
            "pick_shift_mm_mean": round(err["point_shift_mm_mean"], 3),
            "pick_shift_mm_max": round(err["point_shift_mm_max"], 3),
            "residual_rot_deg": round(r["residual"]["rot_deg_mean"], 4),
            "residual_trans_mm": round(r["residual"]["trans_mm_mean"], 3),
            "axis_rank_ratio": r["diagnostics"]["axis_rank_ratio"],
            "plate_rot_deg_mean": round(float(np.mean([e["rot_deg"] for e in plate_err])), 3),
            "plate_trans_mm_mean": round(float(np.mean([e["trans_mm"] for e in plate_err])), 3)}


# ----------------------------------------------------------------------------- 스윕 + 그림
def chart(rows, path):
    """세 장: (1) 자세 수 대 픽 좌표 오차, (2) 자세 수 대 회전 오차, (3) 정답 없이 볼 수 있는 잔차가 실제 오차를 예고하는가."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    for cand in (r"C:\Windows\Fonts\malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        if Path(cand).exists():
            fm.fontManager.addfont(cand)
            plt.rcParams["font.family"] = fm.FontProperties(fname=cand).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"   # 한글 폰트에 U+2212 가 없어 로그 눈금의 지수가 깨진다
    ok = [r for r in rows if r.get("ok")]
    ns = sorted({r["n_poses"] for r in ok})
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    styles = (("none", "#1d4ed8", "노이즈 없음"), ("tof", "#d97706", "ToF 노이즈 모델"))
    for noise, color, label in styles:
        sub = [r for r in ok if r["noise"] == noise]
        if not sub:
            continue
        med = [float(np.median([r["pick_shift_mm_mean"] for r in sub if r["n_poses"] == n])) for n in ns]
        lo = [float(np.min([r["pick_shift_mm_mean"] for r in sub if r["n_poses"] == n])) for n in ns]
        hi = [float(np.max([r["pick_shift_mm_mean"] for r in sub if r["n_poses"] == n])) for n in ns]
        ax[0].plot(ns, med, marker="o", color=color, label=label)
        ax[0].fill_between(ns, lo, hi, color=color, alpha=0.15)
        ax[1].plot(ns, [float(np.median([r["rot_deg"] for r in sub if r["n_poses"] == n])) for n in ns],
                   marker="o", color=color, label=label)
        ax[2].scatter([r["residual_trans_mm"] for r in sub], [r["pick_shift_mm_mean"] for r in sub],
                      s=22, color=color, alpha=0.8, label=label)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("자세 수")
    ax[0].set_ylabel("픽 좌표가 밀리는 거리 (mm)")
    ax[0].set_title("자세가 적으면 가끔 크게 틀린다 (중앙값과 최소~최대)")
    ax[1].set_xlabel("자세 수")
    ax[1].set_ylabel("외참 회전 오차 (deg)")
    ax[1].set_yscale("log")
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda y, _: ("%g" % y))          # 한글 폰트에 없는 유니코드 마이너스를 피한다
    ax[1].yaxis.set_major_formatter(fmt)
    ax[1].yaxis.set_minor_formatter(fmt)
    ax[1].set_title("회전 오차")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("AX-XB 잔차 (mm) — 정답 없이 볼 수 있는 값")
    ax[2].set_ylabel("픽 좌표가 밀리는 거리 (mm)")
    ax[2].set_title("잔차가 크면 결과도 틀렸다")
    for a in ax:
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poses", type=int, nargs="*", default=[4, 6, 8, 12, 16, 20])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--noise", nargs="*", default=["none", "tof"])
    ap.add_argument("--method", default="park_martin", choices=["park_martin", "tsai_lenz"])
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cell = CalibCell()
    pool, checked = visible_pose_pool(cell, n=max(args.poses) + 6, seed=0)
    print(f"자세 풀: 후보 {checked}개 중 판이 보이는 자세 {len(pool)}개", flush=True)
    rows = []
    for noise in args.noise:
        for n in args.poses:
            for s in range(args.seeds):
                r = run_once(cell, pool, n, s, None if noise == "none" else noise, args.method)
                rows.append(r)
                print(f"  자세 {n:2d} 시드 {s} 노이즈 {noise:4s}: "
                      + (f"판 검출 {r['detected']}/{n} · 픽 좌표 이동 {r['pick_shift_mm_mean']:.2f} mm · "
                         f"회전 {r['rot_deg']:.3f}° · 잔차 {r['residual_trans_mm']:.2f} mm"
                         if r["ok"] else r["reason"]), flush=True)

    ok = [r for r in rows if r.get("ok")]
    best = min(ok, key=lambda r: r["pick_shift_mm_mean"]) if ok else None
    summary = {
        "what": "핸드아이 캘리브레이션(eye-to-hand, AX=XB)을 셀 트윈에서 검증한 기록. 관측은 렌더한 ToF 프레임에서 인식으로 구한 판 포즈",
        "target": {"plate_mm": list(PLATE_DIMS_MM), "tab_mm": TAB * 1000.0, "tab_phi_deg": round(TAB_PHI_DEG, 2)},
        "truth_T_base_cam_mm": np.asarray(cell.T_base_cam).round(4).tolist(),
        "pose_pool": {"size": len(pool), "candidates_checked": checked},
        "method": args.method,
        "runs": rows,
        "best": best,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    if ok:
        chart(rows, OUT_PNG)
    print("saved", args.out, "및", OUT_PNG)


if __name__ == "__main__":
    main()
