# -*- coding: utf-8 -*-
"""launch_testing 통합 테스트: 대차 도킹 폐루프 세 노드(cart_node 합성 대차, dock_node, agv_sim_node)를 띄워 결합 위치에 도달하는지 본다.

검사: /dock/state 가 docked=true 가 되고, 그 시점의 (측정) 오차가 허용치 안이며, /agv/rel_pose 의 정답 포즈도 허용치(+인식 오차 여유) 안.
시작 자세는 인식 범위(림 거리 ≤ 500 mm) 안에서 측방 120 mm, 요 −6° 로 준다. 합성 장면만 쓰므로 회사 데이터 없이 돈다."""
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
os.environ.setdefault("ROS_DOMAIN_ID", "74")


@pytest.mark.launch_test
def generate_test_description():
    cart = Node(package="robotsim_perception_ros", executable="cart_node", name="robotsim_cart", output="screen",
                parameters=[{"source": "synthetic_dock", "rate_hz": 0.0, "publish_cloud": False}])
    agv = Node(package="robotsim_perception_ros", executable="agv_sim_node", name="agv_sim", output="screen",
               parameters=[{"init_hook_u_mm": 120.0, "init_rim_v_mm": -480.0, "init_yaw_deg": -6.0}])
    dock = Node(package="robotsim_perception_ros", executable="dock_node", name="dock_node", output="screen")
    return launch.LaunchDescription([cart, agv, dock, launch_testing.actions.ReadyToTest()]), {"cart": cart, "agv": agv, "dock": dock}


class TestDocking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node("test_dock_observer")

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_agv_docks(self, proc_output):
        states, poses = [], []
        self.node.create_subscription(String, "/dock/state", lambda m: states.append(json.loads(m.data)), 50)
        self.node.create_subscription(String, "/agv/rel_pose", lambda m: poses.append(json.loads(m.data)), 50)
        deadline = time.monotonic() + 90.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.2)
            if states and states[-1].get("docked"):
                break
        self.assertTrue(states, "no /dock/state")
        final = states[-1]
        self.assertTrue(final.get("docked"), f"not docked: {final}")
        e = final.get("errors") or {}
        self.assertLess(abs(e.get("e_u", 99)), 10.0 + 1e-6)
        self.assertLess(abs(e.get("e_v", 99)), 10.0 + 1e-6)
        self.assertLess(abs(e.get("e_yaw", 99)), 2.0 + 1e-6)
        self.assertTrue(poses, "no /agv/rel_pose")
        # 정답 포즈: 요 0 근처, 고리가 정면·결합 거리 근처 (인식 오차 여유 15 mm / 3°)
        rp = poses[-1]
        import math
        th = math.radians(rp["yaw_deg"])
        u = math.cos(th) * rp["hook_u_mm"] - math.sin(th) * rp["rim_v_mm"]
        v = math.sin(th) * rp["hook_u_mm"] + math.cos(th) * rp["rim_v_mm"]
        self.assertLess(abs(u), 25.0, rp)
        self.assertLess(abs(v + 200.0), 25.0, rp)
        self.assertLess(abs(rp["yaw_deg"]), 4.0, rp)
        proc_output.assertWaitFor("DOCKED", timeout=5)


@launch_testing.post_shutdown_test()
class TestAfterShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15, 130, 143])
