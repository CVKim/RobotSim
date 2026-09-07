# -*- coding: utf-8 -*-
"""대차 모드 메시지 변환 테스트 — ROS 그래프 없이 실행. 합성 대차 장면만 사용."""
import json

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticStatus
from visualization_msgs.msg import Marker

from robotsim_perception.cart import analyze_cart
from robotsim_perception.synthetic_cart import make_cart_frame

from robotsim_perception_ros import msgs
from robotsim_perception_ros.sources import SyntheticCartSource, make_cart_source

STAMP = Time(sec=2, nanosec=0)


def _quat_to_R(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@pytest.fixture(scope="module")
def scene():
    frame, gt = make_cart_frame(seed=5)
    res = analyze_cart(frame)
    assert res.status == "OK", res.reason
    return frame, gt, res


def test_cart_tf_roundtrip(scene):
    """TF(tof_optical -> cart) 로 고리 카메라 좌표를 옮기면 hook_cart 와 같아야 한다."""
    frame, gt, res = scene
    ts = msgs.cart_transform(res, STAMP, "tof_optical", "cart")
    assert ts.header.frame_id == "tof_optical" and ts.child_frame_id == "cart"
    R_pc = _quat_to_R(ts.transform.rotation)                       # cart 축을 카메라 좌표로
    o = np.array([ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z])
    p_cam = np.asarray(res.hook_cam_mm) * 1e-3
    p_cart = R_pc.T @ (p_cam - o)                                   # 부모 -> 자식 좌표
    assert np.allclose(p_cart, np.asarray(res.hook_cart_mm) * 1e-3, atol=2e-4)


def test_hook_pose_frame_and_orientation(scene):
    frame, gt, res = scene
    ps = msgs.hook_pose_stamped(res, STAMP, "tof_optical")
    assert ps.header.frame_id == "tof_optical"
    assert abs(ps.pose.position.z - res.hook_cam_mm[2] * 1e-3) < 1e-6
    R = _quat_to_R(ps.pose.orientation)
    # 포즈의 z 축 = 대차 h 축 = 데크 법선(카메라 쪽) 과 일치
    n = np.asarray(res.plane["normal_toward_camera"])
    assert float(R[:, 2] @ n) > 0.999


def test_markers_unique_and_in_cart_frame(scene):
    frame, gt, res = scene
    ma = msgs.cart_markers(res, STAMP, "cart")
    assert ma.markers[0].action == Marker.DELETEALL
    added = [m for m in ma.markers if m.action == Marker.ADD]
    assert all(m.header.frame_id == "cart" for m in added)
    keys = [(m.ns, m.id) for m in ma.markers]
    assert len(keys) == len(set(keys))                              # rviz Status: Error 방지
    hook = next(m for m in added if m.ns == "cart" and m.type == Marker.CUBE and m.scale.z > 0.05)
    assert abs(hook.pose.position.x - res.hook_cart_mm[0] * 1e-3) < 1e-6
    assert 0.1 < hook.scale.z < 0.16                                 # 고리 높이 ~130 mm


def test_markers_only_clear_when_not_ok():
    frame, gt = make_cart_frame(seed=6)
    frame.valid[:] = False
    res = analyze_cart(frame)
    ma = msgs.cart_markers(res, STAMP, "cart")
    assert ma.markers[0].action == Marker.DELETEALL
    assert all(m.action in (Marker.DELETEALL, Marker.DELETE) for m in ma.markers)      # ADD 없음, 지난 id 명시적 삭제
    da = msgs.cart_diagnostics(res, STAMP)
    assert da.status[0].level == DiagnosticStatus.ERROR


def test_status_and_diagnostics(scene):
    frame, gt, res = scene
    s = msgs.cart_status_string(res, {"source": "synthetic_cart#1", "seq": 0})
    d = json.loads(s.data)
    assert d["status"] == "OK" and d["source"] == "synthetic_cart#1" and "R_cart" not in d
    da = msgs.cart_diagnostics(res, STAMP)
    assert da.status[0].level == DiagnosticStatus.OK
    keys = {kv.key for kv in da.status[0].values}
    assert {"rim_fit_mad_mm", "rail_edge_mad_mm", "hook_cart_mm", "camera_height_mm"} <= keys


def test_synthetic_cart_source_invariance():
    """카메라 높이·yaw 가 프레임마다 달라도 대차 프레임의 고리 좌표는 같아야 한다."""
    src = SyntheticCartSource(seed=0, loop=False)
    got = []
    while True:
        item = src.next()
        if item is None:
            break
        frame, name = item
        res = analyze_cart(frame)
        assert res.status == "OK", (name, res.reason)
        got.append(res.hook_cart_mm[:2])
    got = np.asarray(got)
    assert len(got) == 4
    assert np.ptp(got, axis=0).max() < 6.0, got
    assert isinstance(make_cart_source("synthetic"), SyntheticCartSource)
