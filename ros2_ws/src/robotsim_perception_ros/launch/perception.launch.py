"""perception_node + rviz2.

  ros2 launch robotsim_perception_ros perception.launch.py                      # 합성 장면, 1 Hz, rviz
  ros2 launch robotsim_perception_ros perception.launch.py source:=/mnt/h/...   # 실측 세션 재생 (로컬 전용)
  ros2 launch robotsim_perception_ros perception.launch.py rate_hz:=0.0         # 트리거 모드: ros2 service call ...
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    cfg = PathJoinSubstitution([FindPackageShare("robotsim_perception_ros"), "config", "perception.rviz"])
    return LaunchDescription([
        DeclareLaunchArgument("source", default_value="synthetic",
                              description="'synthetic' or a .mim session folder / parent folder"),
        DeclareLaunchArgument("rate_hz", default_value="1.0"),
        DeclareLaunchArgument("lattice", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(
            package="robotsim_perception_ros",
            executable="perception_node",
            name="robotsim_perception",
            output="screen",
            parameters=[{
                "source": LaunchConfiguration("source"),
                "rate_hz": LaunchConfiguration("rate_hz"),
                "lattice": LaunchConfiguration("lattice"),
            }],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", cfg],
            output="screen",
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ])
