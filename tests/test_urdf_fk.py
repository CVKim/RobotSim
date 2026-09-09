# -*- coding: utf-8 -*-
"""URDF(rviz 가 그리는 팔)와 MJCF(트윈이 돌리는 팔)의 기구학이 같은지 검사한다.

두 파일이 어긋나면 rviz 의 팔은 트윈과 다른 자리에 서고, 화면으로 확인한 것을 믿을 수 없게 된다.
tools/make_ur10e_urdf.py 가 MJCF 에서 URDF 를 생성하므로, 여기서는 생성물을 읽어 무작위 관절각에서 TCP 를 비교한다.
mujoco 가 없으면 건너뛴다(회사 데이터는 필요 없다)."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

ROOT = Path(__file__).resolve().parent.parent
URDF = ROOT / "ros2_ws" / "src" / "twin_bridge" / "urdf" / "ur10e_track.urdf"
BASE_XY, PEDESTAL_H, TRACK_RANGE = (-0.6, 0.75), 1.08, 0.6


def rpy_to_R(r, p, y) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle_R(axis, ang) -> np.ndarray:
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def urdf_chain(path: Path):
    """URDF 조인트를 부모->자식 순서로 읽어 [(type, name, xyz, rpy, axis)] 로 돌려준다 (직렬 사슬 가정)."""
    root = ET.parse(path).getroot()
    joints = {}
    for j in root.findall("joint"):
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()]
        ax = j.find("axis")
        axis = [float(v) for v in (ax.get("xyz") if ax is not None else "0 0 1").split()]
        joints[parent] = (j.get("type"), j.get("name"), xyz, rpy, axis, child)
    chain, link = [], "base_link"
    while link in joints:
        t, name, xyz, rpy, axis, child = joints[link]
        chain.append((t, name, xyz, rpy, axis))
        link = child
    return chain


def urdf_fk(chain, q: dict) -> np.ndarray:
    """관절값 dict(이름 -> rad 또는 m)로 tcp 위치를 계산한다."""
    T = np.eye(4)
    for t, name, xyz, rpy, axis in chain:
        A = np.eye(4)
        A[:3, :3] = rpy_to_R(*rpy)
        A[:3, 3] = xyz
        T = T @ A
        v = float(q.get(name, 0.0))
        M = np.eye(4)
        if t == "revolute":
            M[:3, :3] = axis_angle_R(axis, v)
        elif t == "prismatic":
            M[:3, 3] = np.asarray(axis, float) * v
        T = T @ M
    return T[:3, 3]


@pytest.fixture(scope="module")
def cell():
    import sys
    sys.path.insert(0, str(ROOT / "sim"))
    from arm import Arm                       # noqa: E402
    from cell_scene import build_xml, full_layout   # noqa: E402
    xml, _ = build_xml(full_layout(1), seed=3,
                       arm=dict(base_xy=BASE_XY, pedestal_h=PEDESTAL_H, track_range=TRACK_RANGE, meshes=False))
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    return Arm(m, d), m, d


def test_urdf_exists_and_parses():
    assert URDF.exists(), f"{URDF} 가 없다 — tools/make_ur10e_urdf.py 를 돌린다"
    chain = urdf_chain(URDF)
    names = [n for t, n, *_ in chain if t in ("revolute", "prismatic")]
    assert names == ["track_joint", "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                     "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], names


def test_urdf_fk_matches_mujoco(cell):
    arm, m, d = cell
    chain = urdf_chain(URDF)
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(12):
        q = rng.uniform(-1.4, 1.4, size=6)
        track = float(rng.uniform(-TRACK_RANGE, TRACK_RANGE))
        p_mj, _ = arm.fk(np.concatenate([[track], q]) if arm.n == 7 else q)
        if arm.n == 7:
            names = dict(zip(["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], q))
            names["track_joint"] = track
        else:
            names = dict(zip(["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], q))
        p_urdf = urdf_fk(chain, names)
        worst = max(worst, float(np.linalg.norm(p_mj - p_urdf)))
    assert worst < 1e-4, f"URDF 와 MJCF 의 TCP 가 {worst * 1000:.2f} mm 어긋난다"
