# -*- coding: utf-8 -*-
"""twin_bridge 순수 로직 테스트 (ROS 없이)."""
import pytest

from twin_bridge.logic import execute_request, path_points, result_message

TOPDOWN = (1.0, 0.0, 0.0, 0.0)


def test_execute_request_from_three_poses():
    poses = [((0.1, 0.2, 1.35), TOPDOWN), ((0.1, 0.2, 1.2), TOPDOWN), ((0.1, 0.2, 1.6), TOPDOWN)]
    req = execute_request(poses, "base_link")
    assert req["pre_pick"] == [0.1, 0.2, 1.35] and req["pick"] == [0.1, 0.2, 1.2] and req["lift"] == [0.1, 0.2, 1.6]
    assert req["quat"] == [1.0, 0.0, 0.0, 0.0] and req["frame_id"] == "base_link"


def test_execute_request_rejects_short_or_zero_quat():
    with pytest.raises(ValueError):
        execute_request([((0, 0, 0), TOPDOWN)])
    with pytest.raises(ValueError):
        execute_request([((0, 0, 1), (0, 0, 0, 0))] * 3)


def test_result_message_strips_path_and_adds_seq():
    res = {"ok": True, "result": "placed", "cycle_s": 7.9, "tcp_path": [[0.05, 0, 0, 1]] * 100, "remaining": 11, "placed": 1}
    out = result_message(res, 3, (12, 500))
    assert "tcp_path" not in out and out["cmd_seq"] == 3 and out["cmd_stamp"] == [12, 500]
    assert out["ok"] is True and out["result"] == "placed" and out["remaining"] == 11
    empty = result_message({}, 1)
    assert empty["ok"] is False and empty["result"] == "unknown"


def test_path_points_downsamples():
    res = {"tcp_path": [[i * 0.05, i, 0.0, 1.0] for i in range(1000)]}
    pts = path_points(res, max_points=100)
    assert 100 <= len(pts) <= 101 and pts[0] == (0.0, 0.0, 1.0) and pts[-1] == (999.0, 0.0, 1.0)
    assert path_points({}) == []
