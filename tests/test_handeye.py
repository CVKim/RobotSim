# -*- coding: utf-8 -*-
"""핸드아이 솔버 (robotsim_perception/handeye.py) — 합성 자세로 정답 복원·노이즈·퇴화 검사. mujoco·회사 데이터 불필요."""
from __future__ import annotations

import numpy as np
import pytest

from robotsim_perception import handeye as he


def make_poses(n, X, T_tool_target, seed=0, rot_sigma=0.5, span_mm=400.0):
    """로봇이 n 자세로 움직였을 때의 (카메라가 본 타깃, 로봇 순기구학) 목록."""
    rng = np.random.default_rng(seed)
    cam, base = [], []
    for _ in range(n):
        T = np.eye(4)
        T[:3, :3] = he.exp_so3(rng.normal(0, rot_sigma, 3))
        T[:3, 3] = rng.uniform(-span_mm, span_mm, 3)
        base.append(T)
        cam.append(X @ T @ T_tool_target)
    return cam, base


@pytest.fixture(scope="module")
def truth():
    X = np.eye(4)                                   # X = T_cam_base
    X[:3, :3] = he.exp_so3([0.05, -0.03, 0.9])
    X[:3, 3] = [120.0, -80.0, 4183.0]
    Z = np.eye(4)                                   # T_tool_target (미지 상수)
    Z[:3, :3] = he.exp_so3([0.1, 0.2, -0.3])
    Z[:3, 3] = [12.0, -7.0, 45.0]
    return X, Z


@pytest.mark.parametrize("method", ["tsai_lenz", "park_martin"])
def test_exact_recovery(truth, method):
    """잡음이 없으면 두 해법 모두 정답을 그대로 복원해야 한다 (부호 규약 회귀 검사)."""
    X, Z = truth
    cam, base = make_poses(8, X, Z, seed=3)
    r = he.solve(cam, base, method=method)
    e = he.transform_error(r["X"], X)
    assert e["rot_deg"] < 1e-6, e
    assert e["trans_mm"] < 1e-6, e
    assert r["residual"]["rot_deg_max"] < 1e-6
    assert r["diagnostics"]["ok"]


def test_inverse_is_T_base_cam(truth):
    """solve 가 주는 T_base_cam 은 X 의 역변환이다 (픽 좌표에 쓰는 방향)."""
    X, Z = truth
    cam, base = make_poses(6, X, Z, seed=5)
    r = he.solve(cam, base)
    assert np.allclose(r["T_base_cam"] @ r["X"], np.eye(4), atol=1e-9)


@pytest.mark.parametrize("method", ["tsai_lenz", "park_martin"])
def test_noise_tolerance(truth, method):
    """관측에 회전 0.3도·이동 1.5 mm 잡음을 넣어도 자세 12개면 회전 1도·이동 20 mm 안에 든다."""
    X, Z = truth
    cam, base = make_poses(12, X, Z, seed=7)
    rng = np.random.default_rng(11)
    noisy = []
    for T in cam:
        N = np.eye(4)
        N[:3, :3] = he.exp_so3(rng.normal(0, np.radians(0.3), 3))
        N[:3, 3] = rng.normal(0, 1.5, 3)
        noisy.append(T @ N)
    r = he.solve(noisy, base, method=method)
    e = he.transform_error(r["X"], X)
    assert e["rot_deg"] < 1.0, (method, e)
    assert e["trans_mm"] < 20.0, (method, e)


def test_more_poses_help(truth):
    """자세를 늘리면 오차가 줄어야 한다 (같은 잡음 수준에서 4개 vs 20개)."""
    X, Z = truth

    def err(n, seed):
        cam, base = make_poses(n, X, Z, seed=seed)
        rng = np.random.default_rng(100 + seed)
        noisy = []
        for T in cam:
            N = np.eye(4)
            N[:3, :3] = he.exp_so3(rng.normal(0, np.radians(0.4), 3))
            N[:3, 3] = rng.normal(0, 2.0, 3)
            noisy.append(T @ N)
        return he.transform_error(he.solve(noisy, base)["X"], X)["trans_mm"]

    few = float(np.median([err(4, s) for s in range(6)]))
    many = float(np.median([err(20, s) for s in range(6)]))
    assert many < few, (few, many)


def test_degenerate_single_axis_is_flagged(truth):
    """요(z축)만 돌린 자세 묶음은 이동 성분이 결정되지 않는다 — 진단이 미리 걸러야 한다."""
    X, Z = truth
    rng = np.random.default_rng(2)
    cam, base = [], []
    for _ in range(8):
        T = np.eye(4)
        T[:3, :3] = he.exp_so3([0.0, 0.0, rng.uniform(-1.0, 1.0)])     # 같은 축
        T[:3, 3] = rng.uniform(-400, 400, 3)
        base.append(T)
        cam.append(X @ T @ Z)
    A, B = he.relative_motions(cam, base)
    d = he.motion_diagnostics(A, B)
    assert not d["ok"] and d["axis_rank_ratio"] < 0.1, d


def test_point_shift_reports_what_matters():
    """변환 오차를 '점이 몇 mm 밀리는가' 로 바꿔 준다 — 픽 좌표에서 실제로 중요한 값."""
    T_true = np.eye(4)
    T_est = np.eye(4)
    T_est[:3, 3] = [3.0, 0.0, 0.0]                  # 3 mm 평행 이동만 틀린 경우
    e = he.transform_error(T_est, T_true, points_mm=[[0, 0, 0], [500, 500, 0]])
    assert abs(e["point_shift_mm_mean"] - 3.0) < 1e-9
    assert e["rot_deg"] < 1e-9


def test_rejects_too_few_poses():
    with pytest.raises(AssertionError):
        he.relative_motions([np.eye(4), np.eye(4)], [np.eye(4), np.eye(4)])


def test_robust_solver_drops_a_bad_observation(truth):
    """관측 하나를 크게 틀리게 만들면(타깃을 다른 평면으로 착각한 경우) 그 자세를 빼고 풀어야 한다."""
    X, Z = truth
    cam, base = make_poses(9, X, Z, seed=13)
    bad = np.eye(4)
    bad[:3, :3] = he.exp_so3([0.0, 0.0, np.pi / 2])        # 90도 틀린 관측
    bad[:3, 3] = [40.0, -30.0, 25.0]
    cam[4] = cam[4] @ bad
    plain = he.transform_error(he.solve(cam, base)["X"], X)
    r = he.solve_robust(cam, base, max_trans_mm=5.0)
    robust = he.transform_error(r["X"], X)
    assert 4 in r["dropped"], r["dropped"]
    assert robust["trans_mm"] < plain["trans_mm"] / 5.0, (plain, robust)
