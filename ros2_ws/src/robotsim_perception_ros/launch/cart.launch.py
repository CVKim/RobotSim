"""cart_node + rviz2 (대차 고리 / 대차 프레임).

  ros2 launch robotsim_perception_ros cart.launch.py                       # 합성 대차 장면, 1 Hz, rviz
  ros2 launch robotsim_perception_ros cart.launch.py source:=/mnt/h/...    # 실측 세션 재생 (로컬 전용)
  ros2 launch robotsim_perception_ros cart.launch.py rate_hz:=0.0          # 트리거 모드
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    cfg = PathJoinSubstitution([FindPackageShare("robotsim_perception_ros"), "config", "cart.rviz"])
    return LaunchDescription([
        DeclareLaunchArgument("source", default_value="synthetic"),
        DeclareLaunchArgument("rate_hz", default_value="1.0"),
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(
            package="robotsim_perception_ros",
            executable="cart_node",
            name="robotsim_cart",
            output="screen",
            parameters=[{
                "source": LaunchConfiguration("source"),
                # value_type 을 못 박지 않으면 `rate_hz:=2` (정수) 가 int 로 전달되고 rclpy 의 정적 타입 파라미터가
                # InvalidParameterTypeException 으로 노드를 죽인다 (double 로 선언됨)
                "rate_hz": ParameterValue(LaunchConfiguration("rate_hz"), value_type=float),
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
