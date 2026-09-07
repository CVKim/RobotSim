# -*- coding: utf-8 -*-
"""pick_executor 판단 로직 테스트 — ROS 없이 돈다 (Windows 의 일반 pytest 로도 실행 가능)."""
import numpy as np
import pytest

from pick_executor.logic import Executor, quat_to_matrix, three_point_trajectory

TOPDOWN = (1.0, 0.0, 0.0, 0.0)          # R = diag(1,-1,-1): 툴 +Z = base -Z (위에서 아래로 접근)
PICK = (-0.512, -0.297, 0.923)          # 실측 세션 재생에서 나온 1번 픽 (m, base_link)


def test_quat_to_matrix_topdown():
    R = quat_to_matrix(*TOPDOWN)
    assert np.allclose(R, np.diag([1.0, -1.0, -1.0]))
    assert np.allclose(R @ R.T, np.eye(3))


def test_three_points_topdown():
    t = three_point_trajectory(PICK, TOPDOWN, clearance_m=0.15, lift_m=0.25)
    assert np.allclose(t.approach, (0, 0, -1))
    assert np.allclose(t.pick, PICK)
    assert np.allclose(t.pre_pick, (PICK[0], PICK[1], PICK[2] + 0.15))     # 접근 시작점은 pick 위 15 cm
    assert np.allclose(t.lift, (PICK[0], PICK[1], PICK[2] + 0.40))         # 들어올린 뒤 40 cm 위
    assert t.orientation == TOPDOWN and len(t.points()) == 3


def test_tilted_approach_follows_quaternion():
    # 툴 +Z 를 (1,0,0) 으로 돌린 자세: y 축 +90도 회전 -> quat (0, sin45, 0, cos45)
    s = np.sqrt(0.5)
    t = three_point_trajectory((0, 0, 0), (0.0, s, 0.0, s), 0.1, 0.2)
    assert np.allclose(t.approach, (1, 0, 0), atol=1e-9)
    assert np.allclose(t.pre_pick, (-0.1, 0, 0)) and np.allclose(t.lift, (-0.3, 0, 0))


def _poses(*positions):
    return [(p, TOPDOWN) for p in positions]


def test_cycle_send_then_duplicate_then_new_scene():
    ex = Executor(min_dist_m=0.05)
    assert ex.on_poses(_poses(PICK)) is None                       # status 가 아직 없으면 판단 보류
    a = ex.on_status({"status": "OK", "n_pickable": 1})
    assert a.kind == "send" and ex.state == "EXECUTING" and a.trajectory.pick == PICK
    # 실행 중에 같은 장면이 다시 오면(인식 노드가 1 Hz 로 계속 내는 경우) busy — 명령을 바꾸지 않는다. status 가 먼저 와도 된다.
    ex.on_status({"status": "OK"})
    b = ex.on_poses(_poses((PICK[0] + 0.01, PICK[1], PICK[2])))
    assert b.kind == "busy" and ex.counters["busy"] == 1
    # 실행 완료 뒤 같은 장면이 또 오면 중복으로 걸러진다 (박스가 아직 그 자리에 있다 = 집기 실패 또는 장면 미갱신)
    r = ex.on_execute_done()
    assert r.kind == "recapture" and ex.state == "IDLE"
    ex.on_status({"status": "OK"})
    d = ex.on_poses(_poses((PICK[0] + 0.01, PICK[1], PICK[2])))
    assert d.kind == "skip_duplicate" and ex.counters["skipped_duplicate"] == 1 and ex.state == "IDLE"
    # 다른 박스 -> 다시 send
    ex.on_status({"status": "OK"})
    c = ex.on_poses(_poses((PICK[0] + 0.3, PICK[1], PICK[2])))
    assert c.kind == "send" and ex.counters["sent"] == 2


def test_status_gate_and_empty():
    ex = Executor()
    ex.on_poses(_poses(PICK))
    a = ex.on_status({"status": "RETAKE", "reason": "valid 0.20 < 0.25"})
    assert a.kind == "skip_status" and ex.counters["skipped_status"] == 1 and ex.state == "IDLE"
    ex.on_poses([])
    b = ex.on_status({"status": "OK"})
    assert b.kind == "empty"
    # LAYER_EMPTY 는 empty_retries(2) 번까지는 재촬영, 3번째 연속이면 DONE
    for i in range(2):
        ex.on_poses([])
        c = ex.on_status({"status": "LAYER_EMPTY"})
        assert c.kind == "empty" and ex.state == "IDLE", (i, c)
    ex.on_poses([])
    c = ex.on_status({"status": "LAYER_EMPTY"})
    assert c.kind == "done" and ex.state == "DONE" and ex.counters["empty"] == 4      # 위의 'OK 인데 포즈 0개' 1건 포함
    # DONE 뒤에는 무엇이 와도 명령을 내지 않는다
    ex.on_poses(_poses(PICK))
    assert ex.on_status({"status": "OK"}).kind == "done"


def test_layer_empty_counter_resets_on_ok():
    """빈 프레임 2번 뒤 박스가 다시 보이면(층 선택이 흔들리는 실제 실패 모드) 카운터가 리셋돼야 한다."""
    ex = Executor()
    for _ in range(2):
        ex.on_poses([])
        assert ex.on_status({"status": "LAYER_EMPTY"}).kind == "empty"
    ex.on_poses(_poses(PICK))
    assert ex.on_status({"status": "OK"}).kind == "send" and ex.consecutive_empty == 0
    ex.on_execute_done()
    for _ in range(2):
        ex.on_poses([])
        assert ex.on_status({"status": "LAYER_EMPTY"}).kind == "empty"      # 다시 2번까지는 재촬영


def test_layer_empty_without_stop_keeps_waiting():
    ex = Executor(stop_on_empty=False)
    for _ in range(5):
        ex.on_poses([])
        a = ex.on_status({"status": "LAYER_EMPTY"})
        assert a.kind == "empty" and ex.state == "IDLE"


def test_capture_failed_exhausted_finishes():
    ex = Executor()
    assert ex.on_capture_failed("source exhausted").kind == "done" and ex.state == "DONE"
    ex2 = Executor()
    assert ex2.on_capture_failed("timeout").kind == "recapture" and ex2.state == "IDLE"


def test_capture_response_with_status_json_is_not_a_failure():
    """인식 노드는 status != OK 인 프레임에도 success=False + 상태 JSON 을 돌려준다. 그 프레임은 토픽으로 이미 나갔으니
    여기서 재촬영을 또 하면 안 된다 (검토에서 발견: 즉시 재촬영 + 타이머 재촬영이 겹쳐 사이클이 두 배로 돌았다)."""
    ex = Executor()
    a = ex.on_capture_failed('{"status": "RETAKE", "reason": "valid 0.20 < 0.25", "n_boxes": 0}')
    assert a.kind == "noop" and ex.state == "IDLE" and ex.counters["recapture"] == 0


def test_busy_while_executing():
    """실행 중에 다른 박스가 보여도 새 명령을 내지 않는다 (로봇이 움직이는 동안 목표가 바뀌면 안 된다)."""
    ex = Executor()
    ex.on_poses(_poses(PICK))
    assert ex.on_status({"status": "OK"}).kind == "send"
    ex.on_poses(_poses((PICK[0] + 0.3, PICK[1], PICK[2])))
    b = ex.on_status({"status": "OK"})
    assert b.kind == "busy" and ex.counters["sent"] == 1 and ex.counters["busy"] == 1
    ex.on_execute_done()
    ex.on_poses(_poses((PICK[0] + 0.3, PICK[1], PICK[2])))
    assert ex.on_status({"status": "OK"}).kind == "send" and ex.counters["sent"] == 2


def test_zero_quaternion_is_rejected():
    ex = Executor()
    ex.on_poses([(PICK, (0.0, 0.0, 0.0, 0.0))])
    a = ex.on_status({"status": "OK"})
    assert a.kind == "skip_status" and "invalid" in a.reason and ex.counters["sent"] == 0
    with pytest.raises(ValueError):
        quat_to_matrix(0.0, 0.0, 0.0, 0.0)


def test_pairing_by_stamp_drops_the_older_half():
    """status 가 두 번 오고(프레임 1, 2) 그 뒤 프레임 2 의 poses 가 오면, 프레임 1 status 는 버리고 2끼리 짝을 짓는다."""
    ex = Executor()
    assert ex.on_status({"status": "OK"}, key=(10, 0)) is None
    assert ex.on_status({"status": "OK"}, key=(11, 0)) is None            # 최신 status 로 교체
    assert ex.on_poses(_poses(PICK), key=(10, 0)) is None                  # 오래된 poses(10) 는 버려진다
    assert ex.counters["unpaired_dropped"] == 1
    a = ex.on_poses(_poses(PICK), key=(11, 0))
    assert a is not None and a.kind == "send"
    # 키가 없으면(구형 인식 노드) 그냥 도착 순서로 짝을 짓는다
    ex2 = Executor()
    ex2.on_status({"status": "OK"})
    assert ex2.on_poses(_poses(PICK)).kind == "send"


def test_snapshot_is_json_friendly():
    import json
    ex = Executor()
    ex.on_poses(_poses(PICK))
    ex.on_status({"status": "OK"})
    d = json.loads(json.dumps(ex.snapshot()))
    assert d["state"] == "EXECUTING" and d["sent"] == 1 and len(d["last_pick_m"]) == 3 and d["busy"] == 0
