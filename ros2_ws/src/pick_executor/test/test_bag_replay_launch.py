# -*- coding: utf-8 -*-
"""launch_testing 재생 테스트: 기록해 둔 bag(합성 12박스 사이클의 /perception/pick_poses + /perception/status)을 `ros2 bag play` 로
흘리고, pick_executor(타이머 모드)가 OK 프레임마다 명령을 하나씩 내는지 본다.

bag 은 scripts/wsl_bag_record.sh 가 만든 test/data/pick_cycle_synth (수십 KB, 합성 데이터만). 기대 명령 수는 bag 안의 status 메시지에서
직접 센다(status OK 이고 pickable ≥ 1 인 프레임 수) — 기록이 바뀌어도 테스트가 따라간다. 인식 노드 없이 제어 노드만 검사하므로
'인식 출력을 고정해 두고 제어를 회귀 검사' 하는 형태다. 실측 세션을 재생한 bag 은 로컬 전용(회사 데이터)이라 커밋하지 않는다."""
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
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from std_msgs.msg import String

WS = Path(__file__).resolve().parents[3]
BAG = Path(__file__).resolve().parent / "data" / "pick_cycle_synth"
os.environ.setdefault("FASTRTPS_DEFAULT_PROFILES_FILE", str(WS / "fastdds_no_shm.xml"))
os.environ.setdefault("ROS_DOMAIN_ID", "72")


def expected_commands(bag_dir: Path) -> int:
    """bag 의 /perception/status 를 읽어 명령이 나가야 하는 프레임 수를 센다."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=""),
                rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    n = 0
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == "/perception/status":
            st = json.loads(deserialize_message(data, String).data)
            if st.get("status") == "OK" and int(st.get("n_pickable", 0)) > 0:
                n += 1
    return n


@pytest.mark.launch_test
def generate_test_description():
    if not (BAG / "metadata.yaml").exists():
        pytest.skip(f"bag not found: {BAG} — run scripts/wsl_bag_record.sh")
    player = ExecuteProcess(cmd=["ros2", "bag", "play", str(BAG), "--rate", "2.0", "--delay", "3.0"], output="screen")
    executor = Node(package="pick_executor", executable="pick_executor", name="pick_executor", output="screen",
                    parameters=[{"trigger_capture": False, "execute_time_s": 0.2, "min_dist_m": 0.05}])
    return launch.LaunchDescription([executor, player, launch_testing.actions.ReadyToTest()]), \
        {"executor": executor, "player": player}


class TestBagReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node("test_bag_replay_observer")

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_one_command_per_ok_frame(self, proc_info, player):
        want = expected_commands(BAG)
        self.assertGreater(want, 0, "bag has no OK frames")
        targets, states = [], []
        self.node.create_subscription(PoseArray, "/robot/target_poses", targets.append, 50)
        self.node.create_subscription(String, "/pick_executor/state", lambda m: states.append(json.loads(m.data)), 50)
        proc_info.assertWaitForShutdown(process=player, timeout=120)      # 재생이 끝날 때까지
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.2)
        self.assertEqual(len(targets), want, f"expected {want} commands (OK frames in bag), got {len(targets)}; last state {states[-1] if states else None}")
        self.assertTrue(states and states[-1]["sent"] == want, f"counters: {states[-1] if states else None}")
        self.assertEqual(states[-1]["busy"], 0, "frames arrived while executing — bag rate too high for execute_time_s")


@launch_testing.post_shutdown_test()
class TestAfterShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15, 130, 143])
