# -*- coding: utf-8 -*-
"""핸드아이 캘리브레이션 — 카메라와 로봇 사이의 고정 변환을 자세 여러 개에서 푼다.

왜 필요한가: 이 패키지의 픽 좌표는 전부 `T_base_cam`(카메라 좌표 → 로봇 좌표) 하나에 얹혀 있다. 그 값이 틀리면
검출이 아무리 정확해도 로봇은 그만큼 빗나간 자리를 집는다. 지금까지 이 변환은 '탑다운 4.183 m' 라는 **가정값**이었다.
여기서는 그것을 **측정으로 구하는 절차**를 구현한다.

설치 형태: 카메라는 셀 위에 고정되어 있고 타깃(캘리브 판)이 로봇 손에 붙어 있다 (eye-to-hand).
  관측 k:  T_cam_target(k) = T_cam_base · T_base_tool(k) · T_tool_target      (T_tool_target 은 미지의 상수)
  두 자세 i, j 의 상대 운동을 만들면 상수항이 지워진다:
      A = T_cam_target(i) · T_cam_target(j)^-1
      B = T_base_tool(i)  · T_base_tool(j)^-1
      A = X · B · X^-1        (X = T_cam_base)
  즉 고전적인 AX = XB 다. 여기 두 가지 해법을 둔다.

  tsai_lenz   회전을 로드리게스 벡터의 선형 문제로 풀고(스큐 대칭), 그 다음 이동을 최소제곱으로 푼다. (Tsai & Lenz 1989)
  park_martin 회전을 so(3) 로그의 최소제곱(Procrustes)으로 풀고 이동은 같은 방식. (Park & Martin 1994)

  둘 다 **회전축이 서로 다른 자세 쌍**이 있어야 풀린다. 모든 운동이 같은 축(예: 요만 돌림)이면 이동 성분이
  그 축 방향으로 결정되지 않는다 — `motion_diagnostics` 가 그것을 미리 알려 준다.

단위: 이동은 mm. 회전 행렬은 3×3, 변환은 4×4 동차행렬.
이 모듈은 numpy 만 쓴다(ROS·mujoco 없이 pytest).
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

__all__ = ["log_so3", "exp_so3", "rot_angle_deg", "relative_motions", "tsai_lenz", "park_martin",
           "solve", "solve_robust", "residuals", "motion_diagnostics", "transform_error"]


# ----------------------------------------------------------------------------- so(3) 유틸
def _skew(v: Sequence[float]) -> np.ndarray:
    x, y, z = (float(t) for t in v)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def log_so3(R: np.ndarray) -> np.ndarray:
    """회전 행렬 -> 회전 벡터(축 × 각, rad). 180도 근방은 대각 성분에서 축을 복원한다."""
    R = np.asarray(R, float)
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    if abs(math.pi - ang) < 1e-6:                     # sin(ang) ~ 0: 축을 (R+I) 에서 뽑는다
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        i = int(np.argmax(axis))
        if axis[i] < 1e-9:
            return np.zeros(3)
        axis = axis * np.sign(A[i] / axis[i])
        axis = axis / np.linalg.norm(axis)
        return axis * ang
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (ang / (2.0 * math.sin(ang)))


def exp_so3(w: Sequence[float]) -> np.ndarray:
    """회전 벡터 -> 회전 행렬 (로드리게스)."""
    w = np.asarray(w, float)
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    K = _skew(w / th)
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def rot_angle_deg(R: np.ndarray) -> float:
    """회전 행렬의 회전각(도)."""
    return float(math.degrees(np.linalg.norm(log_so3(R))))


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(R, float))
    Rn = u @ vt
    if np.linalg.det(Rn) < 0:                          # 반사 제거
        u[:, -1] *= -1
        Rn = u @ vt
    return Rn


def _split(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, float)
    return T[:3, :3], T[:3, 3]


def _make(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, float)
    return T


# ----------------------------------------------------------------------------- 상대 운동
def relative_motions(cam_target: Sequence[np.ndarray], base_tool: Sequence[np.ndarray],
                     min_angle_deg: float = 5.0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """관측 목록에서 (A, B) 쌍을 만든다. 회전이 거의 없는 쌍은 버린다 — 노이즈만 키우고 정보가 없다.

    cam_target[k] = T_cam_target(k)  (카메라가 본 타깃)
    base_tool[k]  = T_base_tool(k)   (로봇 순기구학)
    """
    assert len(cam_target) == len(base_tool) and len(cam_target) >= 3, "자세 3개 이상이 필요하다"
    A, B = [], []
    n = len(cam_target)
    for i in range(n):
        for j in range(i + 1, n):
            a = np.asarray(cam_target[i], float) @ np.linalg.inv(np.asarray(cam_target[j], float))
            b = np.asarray(base_tool[i], float) @ np.linalg.inv(np.asarray(base_tool[j], float))
            if min(rot_angle_deg(a[:3, :3]), rot_angle_deg(b[:3, :3])) < min_angle_deg:
                continue
            A.append(a)
            B.append(b)
    return A, B


def motion_diagnostics(A: Sequence[np.ndarray], B: Sequence[np.ndarray]) -> dict:
    """이 자세 묶음으로 풀 수 있는지 미리 본다.

    회전축이 한 방향에 몰려 있으면(axis_rank_ratio 가 작으면) 이동 성분이 그 축 방향으로 결정되지 않는다.
    실제 캘리브에서 '요만 여러 번 돌린' 데이터가 이 경우다."""
    if not A:
        return {"n_pairs": 0, "axis_rank_ratio": 0.0, "max_angle_deg": 0.0, "ok": False,
                "reason": "회전이 있는 자세 쌍이 없다"}
    axes = []
    angs = []
    for a in A:
        w = log_so3(np.asarray(a, float)[:3, :3])
        th = float(np.linalg.norm(w))
        angs.append(math.degrees(th))
        if th > 1e-6:
            axes.append(w / th)
    if len(axes) < 2:
        return {"n_pairs": len(A), "axis_rank_ratio": 0.0, "max_angle_deg": max(angs), "ok": False,
                "reason": "회전축이 하나뿐이다"}
    s = np.linalg.svd(np.asarray(axes), compute_uv=False)
    ratio = float(s[1] / s[0]) if s[0] > 0 else 0.0        # 두 번째 축 방향이 얼마나 살아 있나
    ok = ratio > 0.1 and max(angs) > 5.0
    return {"n_pairs": len(A), "axis_rank_ratio": round(ratio, 4), "max_angle_deg": round(max(angs), 2),
            "ok": bool(ok), "reason": "" if ok else "회전축이 한 방향에 몰려 있다 (이동 성분이 결정되지 않는다)"}


# ----------------------------------------------------------------------------- 해법
def _translation_ls(A: Sequence[np.ndarray], B: Sequence[np.ndarray], Rx: np.ndarray) -> np.ndarray:
    """회전을 알고 있을 때 이동을 푼다: (Ra - I) tx = Rx · tb - ta  를 쌓아 최소제곱."""
    M, v = [], []
    for a, b in zip(A, B):
        Ra, ta = _split(a)
        _, tb = _split(b)
        M.append(Ra - np.eye(3))
        v.append(Rx @ tb - ta)
    M = np.vstack(M)
    v = np.concatenate(v)
    tx, *_ = np.linalg.lstsq(M, v, rcond=None)
    return tx


def tsai_lenz(A: Sequence[np.ndarray], B: Sequence[np.ndarray]) -> np.ndarray:
    """Tsai-Lenz: 회전을 수정 로드리게스 벡터의 선형 문제로 푼다."""
    M, v = [], []
    for a, b in zip(A, B):
        wa, wb = log_so3(_split(a)[0]), log_so3(_split(b)[0])
        tha, thb = np.linalg.norm(wa), np.linalg.norm(wb)
        if tha < 1e-9 or thb < 1e-9:
            continue
        # 수정 로드리게스: P = 2 sin(th/2) * axis
        Pa = 2.0 * math.sin(tha / 2.0) * (wa / tha)
        Pb = 2.0 * math.sin(thb / 2.0) * (wb / thb)
        M.append(_skew(Pa + Pb))
        v.append(Pb - Pa)      # 부호는 X 의 방향 규약(A = X B X^-1)에 맞춘 것 — 뒤집으면 100도 넘게 틀린 해가 나온다
    if not M:
        raise ValueError("회전이 있는 쌍이 없다")
    P, *_ = np.linalg.lstsq(np.vstack(M), np.concatenate(v), rcond=None)
    n2 = float(P @ P)
    Px = 2.0 * P / math.sqrt(1.0 + n2)                    # 스케일 복원
    Rx = ((1.0 - (Px @ Px) / 2.0) * np.eye(3)
          + 0.5 * (np.outer(Px, Px) + math.sqrt(max(4.0 - (Px @ Px), 0.0)) * _skew(Px)))
    Rx = _orthonormalize(Rx)
    return _make(Rx, _translation_ls(A, B, Rx))


def park_martin(A: Sequence[np.ndarray], B: Sequence[np.ndarray]) -> np.ndarray:
    """Park-Martin: 회전축 로그를 모아 Procrustes 로 회전을 푼다 (닫힌 형태, 노이즈에 더 얌전하다)."""
    M = np.zeros((3, 3))
    for a, b in zip(A, B):
        wa, wb = log_so3(_split(a)[0]), log_so3(_split(b)[0])
        if np.linalg.norm(wa) < 1e-9 or np.linalg.norm(wb) < 1e-9:
            continue
        M += np.outer(wb, wa)
    if not np.any(M):
        raise ValueError("회전이 있는 쌍이 없다")
    w, v = np.linalg.eigh(M.T @ M)
    Rx = v @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ v.T @ M.T
    Rx = _orthonormalize(Rx)
    return _make(Rx, _translation_ls(A, B, Rx))


def solve(cam_target: Sequence[np.ndarray], base_tool: Sequence[np.ndarray], method: str = "park_martin",
          min_angle_deg: float = 5.0) -> dict:
    """관측 목록 -> X = T_cam_base 와 진단·잔차.

    돌려주는 dict: X, T_base_cam(=X^-1), method, n_pairs, diagnostics, residual(회전 deg·이동 mm)
    """
    A, B = relative_motions(cam_target, base_tool, min_angle_deg)
    diag = motion_diagnostics(A, B)
    if not A:
        raise ValueError(f"쓸 수 있는 자세 쌍이 없다: {diag['reason']}")
    X = {"tsai_lenz": tsai_lenz, "park_martin": park_martin}[method](A, B)
    res = residuals(A, B, X)
    return {"X": X, "T_base_cam": np.linalg.inv(X), "method": method, "n_poses": len(cam_target),
            "n_pairs": len(A), "diagnostics": diag, "residual": res}


def solve_robust(cam_target: Sequence[np.ndarray], base_tool: Sequence[np.ndarray], method: str = "park_martin",
                 min_angle_deg: float = 5.0, max_trans_mm: float = 15.0, min_poses: int = 5) -> dict:
    """자세 하나가 잘못 관측되면(타깃을 다른 평면으로 착각) 해 전체가 틀어진다. 잔차가 가장 큰 자세부터 하나씩 빼면서
    남은 자세로 다시 푼다 — 현장 데이터에는 이런 관측이 섞이므로 기본 절차로 둔다.

    돌려주는 dict 는 solve 와 같고 `dropped`(뺀 자세 인덱스)가 붙는다."""
    idx = list(range(len(cam_target)))
    dropped = []
    while True:
        r = solve([cam_target[i] for i in idx], [base_tool[i] for i in idx], method, min_angle_deg)
        if r["residual"]["trans_mm_max"] <= max_trans_mm or len(idx) <= min_poses:
            r["dropped"] = dropped
            r["used"] = list(idx)
            return r
        # 자세별 잔차: 그 자세가 들어간 쌍들의 이동 잔차 평균
        per = []
        for k, i in enumerate(idx):
            keep = [j for j in idx if j != i]
            rk = solve([cam_target[j] for j in keep], [base_tool[j] for j in keep], method, min_angle_deg)
            per.append(rk["residual"]["trans_mm_mean"])
        worst = int(np.argmin(per))          # 그 자세를 빼면 잔차가 가장 많이 줄어드는 자세
        dropped.append(idx.pop(worst))


def residuals(A: Sequence[np.ndarray], B: Sequence[np.ndarray], X: np.ndarray) -> dict:
    """AX 와 XB 가 얼마나 다른가 — 정답을 모르는 현장에서 쓸 수 있는 유일한 품질 지표."""
    rot, trans = [], []
    for a, b in zip(A, B):
        L = np.asarray(a, float) @ X
        R = X @ np.asarray(b, float)
        rot.append(rot_angle_deg(L[:3, :3].T @ R[:3, :3]))
        trans.append(float(np.linalg.norm(L[:3, 3] - R[:3, 3])))
    return {"rot_deg_mean": float(np.mean(rot)), "rot_deg_max": float(np.max(rot)),
            "trans_mm_mean": float(np.mean(trans)), "trans_mm_max": float(np.max(trans))}


def transform_error(T_est: np.ndarray, T_true: np.ndarray, points_mm: Sequence[Sequence[float]] | None = None) -> dict:
    """추정 변환과 참 변환의 차이. 회전(도)·이동(mm) 과, 실제로 중요한 '점이 얼마나 밀리는가'.

    points_mm 를 주면 그 점들을 두 변환으로 옮겨 평균·최대 변위를 낸다 — 픽 좌표가 몇 mm 어긋나는지가 결국 이 값이다."""
    T_est = np.asarray(T_est, float)
    T_true = np.asarray(T_true, float)
    dR = T_est[:3, :3].T @ T_true[:3, :3]
    out = {"rot_deg": rot_angle_deg(dR), "trans_mm": float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]))}
    if points_mm is not None and len(points_mm):
        P = np.asarray(points_mm, float)
        Pe = (T_est[:3, :3] @ P.T).T + T_est[:3, 3]
        Pt = (T_true[:3, :3] @ P.T).T + T_true[:3, 3]
        d = np.linalg.norm(Pe - Pt, axis=1)
        out["point_shift_mm_mean"] = float(d.mean())
        out["point_shift_mm_max"] = float(d.max())
    return out
