# -*- coding: utf-8 -*-
"""launch_testing 통합 테스트: 픽 실행을 **액션**(robotsim_interfaces/ExecutePick)으로 주고받는 경로.

  colcon test --packages-select pick_executor
  또는  launch_test test/test_action_cycle_launch.py

토픽 경로(test_pick_cycle_launch.py)와 같은 사이클을 액션으로 돈다. 로봇 자리에는 이 테스트가 직접 띄우는 가짜 액션 서버가 앉는다 —
트윈(Windows·MuJoCo)이 없어도 계약을 검사할 수 있고, 실제 로봇 드라이버의 액션 서버도 이 자리에 그대로 들어온다.

검사하는 것: (1) pick_executor 가 목표를 12번 보내고 세 점이 base_link 좌표로 온전히 실렸는지, (2) 서버가 낸 피드백이 클라이언트에
도착하는지, (3) 서버가 실패로 끝낸 목표(abort) 뒤에도 사이클이 멈추지 않고 다음 후보로 넘어가는지, (4) 마지막에 DONE 으로 끝나는지.
합성 소스만 쓰므로 회사 데이터 없이 돈다. 도메인 79 (다른 통합 테스트와 겹치지 않게)."""
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
from geometry_msgs.msg import Point
from launch_ros.actions import Node
from rclpy.action import ActionServer
from robotsim_interfaces.action import ExecutePick
from std_msgs.msg import String

WS = Path(__file__).resolve().parents[3]
os.environ.setdefault("FASTRTPS_DEFAULT_PROFILES_FILE", str(WS / "fastdds_no_shm.xml"))
os.environ.setdefault("ROS_DOMAIN_ID", "79")
N_BOXES = 12
FAIL_ON = 3                     # 세 번째 목표는 일부러 실패로 끝낸다 (그 뒤에도 사이클이 이어져야 한다)


@pytest.mark.launch_test
def generate_test_description():
    perception = Node(package="robotsim_perception_ros", executable="perception_node", name="robotsim_perception",
                      output="screen",
                      parameters=[{"source": "synthetic", "rate_hz": 0.0, "loop": False, "synthetic_remove_picked": True,
                                   "publish_cloud": False}])
    executor = Node(package="pick_executor", executable="pick_executor", name="pick_executor", output="screen",
                    parameters=[{"trigger_capture": True, "use_action": True, "action_name": "/robot/execute_pick",
                                 "execute_time_s": 0.3, "execute_timeout_s": 20.0, "startup_delay_s": 3.0,
                                 "watchdog_s": 10.0}])
    return launch.LaunchDescription([perception, executor, launch_testing.actions.ReadyToTest()]), \
        {"perception": perception, "executor": executor}


class TestActionCycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node("test_action_robot")
        cls.goals = []
        cls.server = ActionServer(cls.node, ExecutePick, "/robot/execute_pick", cls.execute)

    @classmethod
    def tearDownClass(cls):
        cls.server.destroy()
        cls.node.destroy_node()
        rclpy.shutdown()

    @classmethod
    def execute(cls, goal_handle):
        """가짜 로봇: 목표를 기록하고 피드백 두 번을 낸 뒤 결과를 돌려준다 (FAIL_ON 번째만 실패)."""
        g = goal_handle.request
        cls.goals.append(g)
        n = len(cls.goals)
        for phase, progress in (("executing", 0.0), ("replaying", 1.0)):
            fb = ExecutePick.Feedback()
            fb.phase, fb.progress = phase, float(progress)
            fb.tcp = Point(x=g.pick.pose.position.x, y=g.pick.pose.position.y, z=g.pick.pose.position.z)
            goal_handle.publish_feedback(fb)
        result = ExecutePick.Result()
        ok = n != FAIL_ON
        result.ok = ok
        result.result = "placed" if ok else "grasp_miss"
        result.cycle_s, result.pick_err_mm, result.place_err_mm = 7.5, 6.1, 4.0
        result.placed = sum(1 for i in range(n) if i + 1 != FAIL_ON)
        result.remaining = max(N_BOXES - result.placed, 0)
        if ok:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    def test_cycle_runs_over_the_action(self, proc_output):
        states = []
        self.node.create_subscription(String, "/pick_executor/state", lambda m: states.append(json.loads(m.data)), 50)
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.2)          # 가짜 서버도 이 spin 으로 돈다
            if states and states[-1]["state"] == "DONE":
                break
        self.assertTrue(states, "no /pick_executor/state received")
        final = states[-1]
        self.assertEqual(final["state"], "DONE", f"cycle did not finish: {final}")
        self.assertGreaterEqual(len(self.goals), N_BOXES, f"expected at least {N_BOXES} goals, got {len(self.goals)}")
        self.assertEqual(final["robot_ok"], len(self.goals) - 1)     # 하나는 일부러 실패시켰다
        self.assertEqual(final["robot_fail"], 1)
        for g in self.goals:
            for ps in (g.pre_pick, g.pick, g.lift):
                self.assertEqual(ps.header.frame_id, "base_link")
            self.assertAlmostEqual(g.pre_pick.pose.position.z - g.pick.pose.position.z, 0.15, delta=1e-3)
            self.assertAlmostEqual(g.lift.pose.position.z - g.pick.pose.position.z, 0.40, delta=1e-3)
            self.assertGreater(g.pick.pose.position.z, 0.5)
            q = g.pick.pose.orientation
            self.assertAlmostEqual(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w, 1.0, delta=1e-6)
        proc_output.assertWaitFor("DONE", timeout=5)


@launch_testing.post_shutdown_test()
class TestAfterShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15, 130, 143])
