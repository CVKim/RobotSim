"""인식 -> 제어 -> 트윈 실행 -> 재촬영 사이클: 카메라와 로봇이 모두 MuJoCo 셀 트윈이다.

  (Windows)  .venv\\Scripts\\python.exe tools/twin_server.py --arm track --boxes 12 --seed 500
  (WSL2)     ros2 launch twin_bridge twin_cycle.launch.py                 # host 생략 = WSL2 기본 게이트웨이(Windows 호스트)
             ros2 launch twin_bridge twin_cycle.launch.py host:=172.27.80.1 rviz:=false

노드 셋: perception_node(source=twin://…, 트리거 모드) + pick_executor(재촬영 요청, 완료 보고 대기) + twin_bridge(로봇 드라이버 자리).
끝나는 조건: 소스 팔레트가 비면 트윈 소스가 '소진'을 알려 pick_executor 가 DONE, 또는 남은 후보가 전부 로봇 실패로 막히면 DONE.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("twin_bridge")
    cfg = PathJoinSubstitution([share, "config", "twin_cycle.rviz"])
    urdf = PathJoinSubstitution([share, "urdf", "ur10e_track.urdf"])
    host, port = LaunchConfiguration("host"), LaunchConfiguration("port")
    # 인식 노드의 source 문자열: 'twin://HOST:PORT' (HOST 가 비면 'twin://:PORT' -> 기본 게이트웨이)
    source = PythonExpression(["'twin://' + '", host, "' + ':' + '", port, "'"])
    return LaunchDescription([
        DeclareLaunchArgument("host", default_value=""),
        DeclareLaunchArgument("port", default_value="5555"),
        DeclareLaunchArgument("execute_timeout_s", default_value="300.0"),
        DeclareLaunchArgument("min_dist_m", default_value="0.05"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("realtime_factor", default_value="1.0"),   # 트윈 TCP 경로를 시뮬 시각대로 재생 (0 = 최대 속도)
        DeclareLaunchArgument("startup_delay_s", default_value="8.0"),   # rviz2 가 뜬 뒤 첫 촬영 (캡처용; 실제 셀과 무관)
        DeclareLaunchArgument("layer_roi_mm", default_value="0.0"),      # 층 히스토그램 영역 (0 = 화면 중앙 ROI 기존; 620 = 팔레트만 — 절제에서 이득 없어 기본 꺼짐)
        DeclareLaunchArgument("temporal_prior", default_value="true"),   # 직전 프레임 층 깊이를 사전으로 (잔여 1~3개 층 점프 방지)
        DeclareLaunchArgument("use_action", default_value="false"),      # true = 픽 명령을 액션(/robot/execute_pick)으로 주고받는다
        # 트윈의 관절각(/joint_states) + URDF -> rviz 에 팔이 그려진다. URDF 는 MJCF 에서 생성한다(tools/make_ur10e_urdf.py)
        Node(package="robot_state_publisher", executable="robot_state_publisher", name="robot_state_publisher",
             output="screen",
             parameters=[{"robot_description": ParameterValue(Command(["cat ", urdf]), value_type=str)}]),
        Node(
            package="robotsim_perception_ros",
            executable="perception_node",
            name="robotsim_perception",
            output="screen",
            parameters=[{
                "source": source,
                "rate_hz": 0.0,                 # 트리거 모드: pick_executor 가 부를 때만 프레임을 받아 처리
                "loop": False,                  # 소스 팔레트가 비면 소진 -> DONE
                "synthetic_remove_picked": False,   # 집은 박스는 트윈 물리가 실제로 옮긴다
                "layer_roi_mm": ParameterValue(LaunchConfiguration("layer_roi_mm"), value_type=float),
                "temporal_prior": ParameterValue(LaunchConfiguration("temporal_prior"), value_type=bool),
            }],
        ),
        Node(
            package="pick_executor",
            executable="pick_executor",
            name="pick_executor",
            output="screen",
            parameters=[{
                "trigger_capture": True,
                # 액션 경로에서는 결과가 액션으로 오므로 완료 보고 토픽을 비운다 (두 경로로 두 번 처리하지 않게)
                "done_topic": PythonExpression(["'' if '", LaunchConfiguration("use_action"),
                                                "'.lower() == 'true' else '/robot/execution_result'"]),
                "use_action": ParameterValue(LaunchConfiguration("use_action"), value_type=bool),
                "execute_timeout_s": ParameterValue(LaunchConfiguration("execute_timeout_s"), value_type=float),
                "execute_time_s": 2.0,          # 명령을 안 낸 프레임 뒤의 재촬영 간격으로만 쓰인다
                "min_dist_m": ParameterValue(LaunchConfiguration("min_dist_m"), value_type=float),
                "watchdog_s": 60.0,             # 트윈 렌더 + 인식은 1 s 안이지만 실행 직후 프레임 요청은 큐에서 기다릴 수 있다
                "startup_delay_s": ParameterValue(LaunchConfiguration("startup_delay_s"), value_type=float),
            }],
        ),
        Node(
            package="twin_bridge",
            executable="twin_bridge",
            name="twin_bridge",
            output="screen",
            parameters=[{
                "host": host,
                "port": ParameterValue(port, value_type=int),
                "realtime_factor": ParameterValue(LaunchConfiguration("realtime_factor"), value_type=float),
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
