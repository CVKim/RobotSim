# -*- coding: utf-8 -*-
"""launch_testing 통합 테스트(트윈 연결): 인식 노드(twin:// 소스) + pick_executor(완료 보고 대기) + twin_bridge 를 띄워
Windows 의 트윈 서버(tools/twin_server.py)와 한 팔레트를 끝까지 비우는지 본다.

트윈 서버가 있어야 하므로 기본 colcon test 에서는 건너뛴다. 돌리려면:
  (Windows)  .venv\\Scripts\\python.exe tools\\twin_server.py --arm track --boxes 12 --seed 500 --exit-after-s 600
  (WSL2)     ROBOTSIM_TWIN_HOST=auto python3 -m pytest test/test_twin_cycle_launch.py -q      # auto = WSL2 기본 게이트웨이(Windows 호스트)
검사: DONE 으로 끝나고, 로봇 완료 보고(robot_ok)가 명령 수와 같고 전송 오류(robot_error)가 0, 배치 ≥ 8/12 (인식 노이즈로 층 선택이
놓치는 마지막 1~3개는 허용), 모든 /robot/execution_result 에 cmd_stamp 가 있고 pick_err_mm 가 100 mm 미만."""
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
from launch_ros.actions import Node
from std_msgs.msg import String

WS = Path(__file__).resolve().parents[3]
os.environ.setdefault("FASTRTPS_DEFAULT_PROFILES_FILE", str(WS / "fastdds_no_shm.xml"))
os.environ.setdefault("ROS_DOMAIN_ID", "73")
HOST = os.environ.get("ROBOTSIM_TWIN_HOST", "")
PORT = os.environ.get("ROBOTSIM_TWIN_PORT", "5555")


@pytest.mark.launch_test
def generate_test_description():
    if not HOST:
        pytest.skip("set ROBOTSIM_TWIN_HOST (auto | ip) with tools/twin_server.py running on Windows")
    host = "" if HOST == "auto" else HOST
    source = f"twin://{host}:{PORT}"
    perception = Node(package="robotsim_perception_ros", executable="perception_node", name="robotsim_perception", output="screen",
                      parameters=[{"source": source, "rate_hz": 0.0, "loop": False, "synthetic_remove_picked": False,
                                   "publish_cloud": False}])
    executor = Node(package="pick_executor", executable="pick_executor", name="pick_executor", output="screen",
                    parameters=[{"trigger_capture": True, "done_topic": "/robot/execution_result", "execute_timeout_s": 120.0,
                                 "execute_time_s": 1.0, "watchdog_s": 60.0, "startup_delay_s": 2.0}])
    bridge = Node(package="twin_bridge", executable="twin_bridge", name="twin_bridge", output="screen",
                  parameters=[{"host": host, "port": int(PORT), "realtime_factor": 0.0}])   # 최대 속도: 재생 없이 보고
    return launch.LaunchDescription([perception, executor, bridge, launch_testing.actions.ReadyToTest()]), \
        {"perception": perception, "executor": executor, "bridge": bridge}


class TestTwinCycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node("test_twin_cycle_observer")

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_twin_pallet_is_emptied(self, proc_output):
        results, states = [], []
        self.node.create_subscription(String, "/robot/execution_result", lambda m: results.append(json.loads(m.data)), 50)
        self.node.create_subscription(String, "/pick_executor/state", lambda m: states.append(json.loads(m.data)), 50)
        deadline = time.monotonic() + 240.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.5)
            if states and states[-1]["state"] == "DONE":
                break
        self.assertTrue(states and states[-1]["state"] == "DONE", f"no DONE: {states[-1] if states else None}")
        final = states[-1]
        self.assertGreaterEqual(final["sent"], 8, f"too few commands: {final}")
        self.assertEqual(final["robot_error"], 0, f"transport/bridge errors: {final}")
        self.assertEqual(final["robot_ok"] + final["robot_fail"], final["sent"], f"every command must be reported: {final}")
        self.assertGreaterEqual(final["outcomes"].get("placed", 0), 8, f"outcomes: {final['outcomes']}")
        self.assertEqual(len(results), final["sent"])
        for r in results:
            self.assertIn("cmd_stamp", r)
            if r["ok"]:
                self.assertLess(r["pick_err_mm"], 100.0, r)
        proc_output.assertWaitFor("DONE", timeout=5)


@launch_testing.post_shutdown_test()
class TestAfterShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15, 130, 143])
