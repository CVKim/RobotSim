# robotsim_interfaces

노드 사이의 계약을 타입으로 적은 패키지. 여기 있는 것만 다른 패키지가 의존한다(순환 없음).

| 종류 | 이름 | 쓰는 곳 |
|---|---|---|
| 액션 | `action/ExecutePick.action` | `pick_executor`(클라이언트) ↔ `twin_bridge`(서버). 픽 한 번 = 목표 하나 |
| 메시지 | `msg/DockState.msg` | `dock_node` 가 매 프레임 발행, `/dock/state` |

## 왜 만들었나

처음에는 둘 다 `std_msgs/String` 안의 JSON 이었다. 돌아가기는 하지만 계약이 코드 주석에만 있어서

- 필드 이름·단위가 틀려도 빌드가 통과하고 런타임에 조용히 `None` 이 되며,
- `ros2 topic echo` 나 bag 재생에서 필드 단위로 읽을 수 없고,
- 픽처럼 수 초~수십 초 걸리는 작업의 **진행 상황·취소**를 표현할 자리가 없다.

액션으로 바꾸면 목표 하나에 결과 하나가 붙고, 서버가 실행 중이면 새 목표를 **거절**할 수 있으며(전에는 busy 를 따로 보고했다),
진행 중 피드백이 온다. 취소는 지금 구현이 받지 않는다 — 트윈의 한 실행은 TCP 요청 하나로 끝까지 돌아 중간에 끊을 수단이 없다.
실제 로봇 드라이버라면 그 자리에서 정지 명령을 보낸다.

## 빌드

`ament_cmake` + `rosidl_default_generators` 로 파이썬·C++ 양쪽 타입을 만든다. 인터페이스 패키지는 순수 파이썬(`ament_python`)으로는
만들 수 없다.

```bash
colcon build --packages-select robotsim_interfaces
source install/setup.bash
ros2 interface show robotsim_interfaces/action/ExecutePick
ros2 interface show robotsim_interfaces/msg/DockState
```

## 쓰는 쪽

```bash
# 픽 실행 액션 (서버: twin_bridge)
ros2 action list -t
ros2 action send_goal /robot/execute_pick robotsim_interfaces/action/ExecutePick \
  "{pre_pick: {header: {frame_id: base_link}, pose: {position: {x: 0.3, y: 0.0, z: 1.1}}}, ...}" --feedback

# 도킹 상태 (발행: dock_node)
ros2 topic echo /dock/state
```

통합 테스트: `pick_executor/test/test_action_cycle_launch.py`(가짜 액션 서버로 사이클 12회),
`robotsim_perception_ros/test/test_dock_launch.py`(DockState 로 결합 판정 + 좌표 변환 사슬 검사).
