"""인식 -> 제어 -> 재촬영 사이클 데모: perception_node(트리거 모드) + pick_executor(재촬영 요청) + rviz2.

  ros2 launch pick_executor pick_cycle.launch.py                 # 합성 장면 12박스: 촬영 -> 명령 -> 2초 실행 -> 재촬영 ... -> 12개 다 집으면 소진 -> DONE
  ros2 launch pick_executor pick_cycle.launch.py rviz:=false execute_time_s:=0.5
  ros2 launch pick_executor pick_cycle.launch.py source:=/mnt/h/...  # 실측 세션 재생 (로컬 전용)

인식 노드는 rate_hz:=0 (스스로 프레임을 처리하지 않음). pick_executor 가 ~/capture 를 호출할 때만 한 프레임을 처리한다.
합성 소스는 synthetic_remove_picked 로 '계획 1번 픽' 박스를 다음 프레임에서 없앤다(synthetic_pick_every 는 이때 무시된다).
끝나는 조건 둘 중 먼저 오는 것: 소스 소진(12개 다 집음, loop:=false) 또는 LAYER_EMPTY 연속 3번(pick_executor empty_retries).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    cfg = PathJoinSubstitution([FindPackageShare("pick_executor"), "config", "pick_cycle.rviz"])
    return LaunchDescription([
        DeclareLaunchArgument("source", default_value="synthetic"),
        DeclareLaunchArgument("execute_time_s", default_value="2.0"),
        DeclareLaunchArgument("min_dist_m", default_value="0.05"),
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(
            package="robotsim_perception_ros",
            executable="perception_node",
            name="robotsim_perception",
            output="screen",
            parameters=[{
                "source": LaunchConfiguration("source"),
                "rate_hz": 0.0,                 # 트리거 모드
                "loop": False,                  # 합성 소스: 12박스 다 집으면 소진 -> executor 가 DONE
                "synthetic_remove_picked": True,  # 계획 1번 픽 박스가 다음 프레임에서 사라진다 (로봇이 집어 간 것처럼)
                # 격자 보완(lattice)은 기본값 True 그대로. 합성 박스는 강도 이음새가 없어 붙은 박스가 한 덩어리로 합쳐지므로
                # 보완 없이는 첫 프레임에서 12개 중 4개만 검출된다 (오프라인 재현으로 확인).
            }],
        ),
        Node(
            package="pick_executor",
            executable="pick_executor",
            name="pick_executor",
            output="screen",
            parameters=[{
                "trigger_capture": True,
                "execute_time_s": ParameterValue(LaunchConfiguration("execute_time_s"), value_type=float),
                "min_dist_m": ParameterValue(LaunchConfiguration("min_dist_m"), value_type=float),
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
