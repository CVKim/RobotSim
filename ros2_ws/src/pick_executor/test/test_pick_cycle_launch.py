# -*- coding: utf-8 -*-
"""launch_testing 통합 테스트: 인식 노드(합성 12박스, 트리거 모드) + pick_executor 를 실제로 띄워 사이클이 끝까지 도는지 본다.

  colcon test --packages-select pick_executor        (pytest 가 launch_testing 플러그인으로 이 파일을 실행한다)
  또는  launch_test test/test_pick_cycle_launch.py

검사하는 것: /robot/target_poses 가 12번(박스 12개) 나오고 각 메시지가 base_link 좌표의 3점이며, /pick_executor/state 가 DONE 으로
끝나고 카운터가 sent=12, busy=0 인지. 단위 테스트(test_logic.py)가 못 보는 것 — 두 노드의 토픽·서비스 배선, 스탬프 짝 맞춤,
Trigger 응답 처리, 타이머 — 를 프로세스째 검사한다. 합성 소스만 쓰므로 회사 데이터 없이 돈다.
WSL2 에서는 Fast DDS 공유메모리가 안 되므로 UDP 전용 프로필을 강제하고, 다른 실행과 겹치지 않게 도메인 71 을 쓴다."""
import json
import os
import time
import unittest
from pathlib import Path

import launch
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import pytest
import rclpy
from geometry_msgs.msg import PoseArray
from launch_ros.actions import Node
from std_msgs.msg import String

WS = Path(__file__).resolve().parents[3]
os.environ.setdefault("FASTRTPS_DEFAULT_PROFILES_FILE", str(WS / "fastdds_no_shm.xml"))
os.environ.setdefault("ROS_DOMAIN_ID", "71")
N_BOXES = 12


@pytest.mark.launch_test
def generate_test_description():
    perception = Node(package="robotsim_perception_ros", executable="perception_node", name="robotsim_perception",
                      output="screen",
                      parameters=[{"source": "synthetic", "rate_hz": 0.0, "loop": False, "synthetic_remove_picked": True,
                                   "publish_cloud": False}])
    executor = Node(package="pick_executor", executable="pick_executor", name="pick_executor", output="screen",
                    parameters=[{"trigger_capture": True, "execute_time_s": 0.3, "startup_delay_s": 2.0, "watchdog_s": 10.0}])
    return launch.LaunchDescription([perception, executor, launch_testing.actions.ReadyToTest()]), \
        {"perception": perception, "executor": executor}


class TestPickCycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node("test_pick_cycle_observer")

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_cycle_runs_to_done(self, proc_output):
        targets, states = [], []
        self.node.create_subscription(PoseArray, "/robot/target_poses", targets.append, 50)
        self.node.create_subscription(String, "/pick_executor/state", lambda m: states.append(json.loads(m.data)), 50)
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.5)
            if states and states[-1]["state"] == "DONE":
                break
        self.assertTrue(states, "no /pick_executor/state received")
        final = states[-1]
        self.assertEqual(final["state"], "DONE", f"cycle did not finish: {final}")
        self.assertEqual(final["sent"], N_BOXES, f"expected {N_BOXES} commands, counters: {final}")
        self.assertEqual(final["busy"], 0)
        self.assertEqual(len(targets), N_BOXES)
        for pa in targets:
            self.assertEqual(pa.header.frame_id, "base_link")
            self.assertEqual(len(pa.poses), 3)                  # pre-pick, pick, lift
            pre, pick, lift = pa.poses
            self.assertAlmostEqual(pre.position.z - pick.position.z, 0.15, delta=1e-3)    # clearance_m
            self.assertAlmostEqual(lift.position.z - pick.position.z, 0.40, delta=1e-3)   # clearance + lift
            self.assertGreater(pick.position.z, 0.5)                                       # 박스 상면은 바닥 위
        # 로그로도 종료를 확인 (launch_testing 의 프로세스 출력 검사)
        proc_output.assertWaitFor("DONE", timeout=5)


@launch_testing.post_shutdown_test()
class TestAfterShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        # launch_testing 이 SIGINT 로 내린다 — rclpy 노드는 0 또는 시그널 코드로 끝난다
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15, 130, 143])
