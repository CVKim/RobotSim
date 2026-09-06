# -*- coding: utf-8 -*-
"""메시지 변환 테스트 — ROS 그래프 없이 실행 (colcon test 또는 pytest, ROS 환경 source 필요).

합성 프레임만 사용하므로 회사 데이터 없이 돈다.
"""
import json

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticStatus
from visualization_msgs.msg import Marker

from robotsim_perception.pose import topdown_camera_transform
from robotsim_perception.runtime import HealthMonitor, Thresholds, decide

from robotsim_perception_ros import msgs
from robotsim_perception_ros.sources import SyntheticSource

STAMP = Time(sec=1, nanosec=0)
T = topdown_camera_transform(4183.0)


@pytest.fixture(scope="module")
def scene():
    frame, name = SyntheticSource(seed=3).next()
    dec = decide(frame, Thresholds(lattice=False, min_confidence=0.3, source_roi_mm=620.0))
    return frame, dec


def test_pointcloud_fields_and_frame(scene):
    frame, _ = scene
    pc = msgs.frame_to_pointcloud2(frame, STAMP, "tof_optical", stride=4)
    assert pc.header.frame_id == "tof_optical"
    assert [f.name for f in pc.fields] == ["x", "y", "z", "intensity"]
    assert pc.width * pc.height == int(frame.valid[::4, ::4].sum())
    assert pc.point_step == 16


def test_static_tf_topdown():
    ts = msgs.static_transform(T, STAMP, "base_link", "tof_optical")
    assert ts.header.frame_id == "base_link" and ts.child_frame_id == "tof_optical"
    assert abs(ts.transform.translation.z - 4.183) < 1e-6
    q = ts.transform.rotation
    # diag(1,-1,-1) = X 축 180도 회전 -> (x=1, y=0, z=0, w=0)
    assert abs(abs(q.x) - 1.0) < 1e-6 and abs(q.w) < 1e-6


def test_pick_poses_in_base_frame(scene):
    frame, dec = scene
    assert dec.status == "OK", dec.reason
    pps = msgs.pick_poses(dec.boxes, dec.plan, T)
    assert len(pps) == len(dec.plan) == dec.n_pickable
    for p in pps:
        z_base = p.position_mm[2]
        assert 4183.0 - 2970.0 - 60 < z_base < 4183.0 - 2970.0 + 60     # 상면 높이 = 카메라 높이 - 깊이
        assert p.approach[2] < -0.99                                    # 탑다운: 접근 = -Z


def test_posearray_and_next(scene):
    frame, dec = scene
    pps = msgs.pick_poses(dec.boxes, dec.plan, T)
    pa = msgs.poses_to_posearray(pps, STAMP, "base_link")
    assert pa.header.frame_id == "base_link" and len(pa.poses) == len(pps)
    ps = msgs.pose_stamped(pps[0], STAMP, "base_link")
    q = ps.pose.orientation
    assert abs(q.x ** 2 + q.y ** 2 + q.z ** 2 + q.w ** 2 - 1.0) < 1e-6


def test_markers_cover_all_boxes(scene):
    frame, dec = scene
    pps = msgs.pick_poses(dec.boxes, dec.plan, T)
    ma = msgs.boxes_to_markers(dec.boxes, T, pps, STAMP, "base_link")
    assert ma.markers[0].action == Marker.DELETEALL                    # DELETEALL 먼저
    added = [m for m in ma.markers if m.action == Marker.ADD]
    kinds = [m.ns for m in added]
    assert kinds.count("boxes") == len(dec.boxes)
    assert kinds.count("boxes_label") == len(dec.boxes)
    assert kinds.count("boxes_next") == 1
    cube = next(m for m in added if m.ns == "boxes")
    assert 0.2 < cube.scale.x < 0.35 and 0.15 < cube.scale.y < 0.26     # SKU 293x219 mm 근처


def test_status_and_diagnostics(scene):
    frame, dec = scene
    s = msgs.decision_to_string(dec, {"source": "synthetic#1"})
    d = json.loads(s.data)
    assert d["status"] == "OK" and d["source"] == "synthetic#1"
    hm = HealthMonitor()
    hm.set_reference(frame)
    da = msgs.diagnostics(dec, hm.check(frame), STAMP)
    names = [st.name for st in da.status]
    assert "robotsim_perception/decision" in names and "robotsim_perception/camera_drift" in names
    assert all(st.level == DiagnosticStatus.OK for st in da.status)    # level 은 byte 타입 — 정수 0 과 비교하면 안 된다
