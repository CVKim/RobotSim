"""대차 도킹 폐루프: cart_node(합성 대차, AGV 포즈로 렌더) + dock_node(제어) + agv_sim_node(운동학 플랜트) + rviz2.

  ros2 launch robotsim_perception_ros dock.launch.py                              # 시작 hook_u −80, rim_v −450, yaw 5°
  ros2 launch robotsim_perception_ros dock.launch.py init_hook_u_mm:=120.0 init_rim_v_mm:=-480.0 init_yaw_deg:=-6.0 rviz:=false
  ros2 launch robotsim_perception_ros dock.launch.py params_file:=/경로/내_파라미터.yaml   # 기본값 파일을 통째로 갈아끼운다
  ros2 topic echo /dock/state       # robotsim_interfaces/DockState (mode, e_u_mm, e_v_mm, e_yaw_deg, docked, tf_check_mm)
  ros2 param dump /dock_node        # 지금 돌고 있는 노드의 파라미터 (config/dock_params.yaml 과 대조)

파라미터 기본값은 config/dock_params.yaml 에 있고, launch 인자(init_*, rate_hz)는 그 위에 덮어쓴다 — 뒤에 온 dict 가 이긴다.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("robotsim_perception_ros")
    cfg = PathJoinSubstitution([share, "config", "cart.rviz"])
    params = LaunchConfiguration("params_file")
    f = lambda name: ParameterValue(LaunchConfiguration(name), value_type=float)  # noqa: E731
    return LaunchDescription([
        DeclareLaunchArgument("init_hook_u_mm", default_value="-80.0"),
        DeclareLaunchArgument("init_rim_v_mm", default_value="-450.0"),
        DeclareLaunchArgument("init_yaw_deg", default_value="5.0"),
        DeclareLaunchArgument("rate_hz", default_value="0.0"),          # 0 = 트리거 모드 (dock_node 가 AGV 정지 후 촬영 요청, stop-and-go)
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("params_file", default_value=PathJoinSubstitution([share, "config", "dock_params.yaml"])),
        Node(package="robotsim_perception_ros", executable="cart_node", name="robotsim_cart", output="screen",
             parameters=[params, {"source": "synthetic_dock", "rate_hz": f("rate_hz")}]),
        Node(package="robotsim_perception_ros", executable="agv_sim_node", name="agv_sim", output="screen",
             parameters=[params, {"init_hook_u_mm": f("init_hook_u_mm"), "init_rim_v_mm": f("init_rim_v_mm"),
                                  "init_yaw_deg": f("init_yaw_deg")}]),
        Node(package="robotsim_perception_ros", executable="dock_node", name="dock_node", output="screen",
             parameters=[params]),
        Node(package="rviz2", executable="rviz2", name="rviz2", arguments=["-d", cfg], output="screen",
             condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
