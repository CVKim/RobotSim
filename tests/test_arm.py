# -*- coding: utf-8 -*-
"""UR10e 팔 모듈 (sim/arm.py) — FK/IK 일관성, 충돌 검사, 구간 시간. 메시 없이도 돈다."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
mujoco = pytest.importorskip("mujoco")

from arm import Arm, rot_from_approach_yaw, rotvec_between, box_body_geom_ids  # noqa: E402
from cell_scene import BOX, DECK_H, build_xml, full_layout, grid_xy  # noqa: E402
from cell_twin import settle  # noqa: E402


@pytest.fixture(scope="module")
def cell():
    xml, gt = build_xml(full_layout(1), seed=3, arm=dict(base_xy=(-0.6, 0.85), pedestal_h=0.8, meshes=False))
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    settle(m, d, 400)
    arm = Arm(m, d)
    arm.set_q(arm.home)
    return m, d, arm


def test_legacy_scene_unchanged_by_radian_switch():
    """팔 없는 씬은 그대로 빌드되고 mocap 석션이 남아 있어야 한다 (기존 트윈 도구 호환)."""
    xml, _ = build_xml(full_layout(1), seed=0)
    m = mujoco.MjModel.from_xml_string(xml)
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "suction_target") >= 0
    assert 'angle="radian"' in xml


def test_tool_points_down_at_home(cell):
    m, d, arm = cell
    p, R = arm.tcp()
    assert np.allclose(R[:, 2], (0, 0, -1), atol=1e-4)         # 홈 자세에서 툴 z = 아래 (홈 각도가 1.5708 이라 1e-5 오차)
    assert p[2] > DECK_H + BOX[2] + 0.1                          # 상면보다 위에서 시작
    assert arm.contacts() == []                                  # 홈에서 환경과 접촉 없음


def test_fk_ik_roundtrip(cell):
    """관절 한계 안의 무작위 자세를 FK 로 목표를 만들고, 다른 시드에서 IK 로 되찾는다."""
    m, d, arm = cell
    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(6):
        q = np.clip(arm.home + rng.normal(0, 0.5, arm.n), arm.lo, arm.hi)
        p, R = arm.fk(q)
        r = arm.solve_ik_multi(p, R, arm.home + rng.normal(0, 0.2, arm.n), rng=rng)
        assert r.pos_err_m < 2e-3 and r.rot_err_deg < 1.0, (r.pos_err_m, r.rot_err_deg)
        ok += int(r.ok)
    assert ok == 6


def test_ik_reaches_near_box_and_rejects_far_target(cell):
    m, d, arm = cell
    Rt = rot_from_approach_yaw((0, 0, -1), 12.0)
    x, y = grid_xy(0, 0)                                          # 팔 쪽 모서리 박스
    z_top = DECK_H + 0.004 + BOX[2]
    r = arm.solve_ik_multi(np.array([x, y, z_top + 0.002]), Rt)
    assert r.ok and r.pos_err_m < 1e-3
    p, R = arm.fk(r.q)
    assert np.allclose(R[:, 2], (0, 0, -1), atol=2e-2)
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    assert abs(((yaw - 12.0) + 90) % 180 - 90) < 1.0             # 요 방향(180도 대칭) 일치
    far = arm.solve_ik_multi(np.array([3.0, -2.0, 1.0]), Rt)      # 도달 1.3 m 밖
    assert not far.ok and far.pos_err_m > 0.5


def test_collision_detected_when_descending_into_box(cell):
    """상면 아래로 TCP 를 내리면 컵이 아니라 툴 몸통/손목이 박스와 접촉한다 -> 충돌로 잡혀야 한다."""
    m, d, arm = cell
    Rt = rot_from_approach_yaw((0, 0, -1), 0.0)
    x, y = grid_xy(1, 0)
    z_top = DECK_H + 0.004 + BOX[2]
    boxes = box_body_geom_ids(m)
    r_ok = arm.solve_ik_multi(np.array([x, y, z_top + 0.002]), Rt)
    assert r_ok.ok and arm.contacts(r_ok.q, allowed_gids=list(boxes.values())) == []
    r_deep = arm.solve_ik_multi(np.array([x, y, z_top - 0.12]), Rt, r_ok.q)
    assert r_deep.ok
    hits = arm.contacts(r_deep.q, allowed_gids=list(boxes.values()))
    assert hits and any("box" in h[1] for h in hits)
    assert arm.path_collides(r_ok.q, r_deep.q, n=6, allowed_gids=list(boxes.values())) is not None


def test_segment_time_trapezoid(cell):
    m, d, arm = cell
    q0 = arm.home.copy()
    q1 = q0.copy()
    q1[1] += np.radians(120.0)                                    # 숄더 120도 -> 1 s @120deg/s + 가속 0.25 s
    assert abs(arm.segment_time(q0, q1) - 1.25) < 1e-6
    q2 = q0.copy()
    q2[5] += np.radians(1.0)                                      # 짧은 구간: 삼각 프로파일
    assert 0.0 < arm.segment_time(q0, q2) < 0.3
    assert arm.segment_time(q0, q0) == 0.0


def test_rotvec_between():
    R = rot_from_approach_yaw((0, 0, -1), 30.0)
    R2 = rot_from_approach_yaw((0, 0, -1), 0.0)
    v = rotvec_between(R, R2)
    assert abs(abs(np.degrees(np.linalg.norm(v))) - 30.0) < 1e-6
    assert np.allclose(rotvec_between(R, R), 0.0)
