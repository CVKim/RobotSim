"""MoveIt2 move_group 만 띄운다 (계획 전용 — 실행은 트윈이 한다).

  ros2 launch robotsim_moveit_config move_group.launch.py
  ros2 run robotsim_moveit_config plan_compare.py --ros-args -p problems:=/mnt/e/Robot_Sim/explore/twin/plan_problems.json

로봇 모델은 tools/make_ur10e_urdf.py 가 MJCF 에서 생성한 URDF 를 그대로 쓴다(twin_bridge/urdf).
실행(컨트롤러)은 붙이지 않는다: 궤적을 실제로 도는 것은 MuJoCo 트윈이고, 여기서는 '계획이 되는가·얼마나 걸리는가' 만 본다.

주의: MoveIt 설정 YAML 은 노드 파라미터 파일 형식(`node: ros__parameters:`)이 아니라 **그대로 읽어 dict 로 넘겨야** 한다.
파일 경로를 그대로 parameters 에 주면 move_group 이 'Couldn't parse params file' 로 죽는다(실제로 겪음).
"""
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def generate_launch_description():
    cfg = Path(get_package_share_directory("robotsim_moveit_config")) / "config"
    urdf = Path(get_package_share_directory("twin_bridge")) / "urdf" / "ur10e_track.urdf"

    robot_description = {"robot_description": ParameterValue(Command(["cat ", str(urdf)]), value_type=str)}
    robot_description_semantic = {"robot_description_semantic": (cfg / "ur10e_track.srdf").read_text(encoding="utf-8")}
    kinematics = {"robot_description_kinematics": load_yaml(cfg / "kinematics.yaml")}
    limits = {"robot_description_planning": load_yaml(cfg / "joint_limits.yaml")}
    ompl = load_yaml(cfg / "ompl_planning.yaml")
    planning = {"planning_pipelines": ["ompl"], "default_planning_pipeline": "ompl", "ompl": ompl}

    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
             parameters=[robot_description]),
        Node(package="moveit_ros_move_group", executable="move_group", output="screen",
             parameters=[robot_description, robot_description_semantic, kinematics, limits, planning,
                         {"allow_trajectory_execution": False,       # 실행기는 트윈 쪽에 있다
                          "publish_robot_description_semantic": True,
                          "publish_planning_scene": True,
                          "publish_geometry_updates": True,
                          "publish_state_updates": True,
                          "publish_transforms_updates": True}]),
    ])
