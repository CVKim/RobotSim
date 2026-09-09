# pick_executor

인식 노드(`robotsim_perception_ros`)가 내는 픽 포즈를 받아 **로봇에게 보낼 명령**을 만들고, 실행이 끝나면
인식 노드에 **재촬영을 요청**해 인식 → 제어 → 재촬영 사이클을 닫는 노드. docs/42 6절의 실습 과제를 구현한 것이다.

```
 perception_node ──/perception/pick_poses (PoseArray)──▶ pick_executor ──▶ /robot/target_poses (PoseArray: pre-pick, pick, lift)
      ▲          ──/perception/status (String JSON)───▶      │          ──▶ /robot/trajectory   (MarkerArray)
      │                                                       │          ──▶ /pick_executor/state (String JSON)
      └──────────── ~/capture (std_srvs/Trigger) ◀────────────┘  실행(execute_time_s) 뒤 호출
```

| 항목 | 내용 |
|---|---|
| 구독 | `/perception/pick_poses` `geometry_msgs/PoseArray`(base_link), `/perception/status` `std_msgs/String`(JSON) — 둘 다 모이면 한 프레임으로 판단 |
| 퍼블리시 | `/robot/target_poses` 3점(pre-pick → pick → lift, 같은 툴 자세), `/robot/trajectory` 마커, `/pick_executor/state` 상태·카운터 |
| 서비스 클라이언트 | `/robotsim_perception/capture` — `trigger_capture:=true` 일 때 실행 완료마다 호출 |
| 파라미터 | `min_dist_m` 0.05 (직전 pick 과 이 안이면 같은 장면 → 보내지 않음), `clearance_m` 0.15, `lift_m` 0.25, `execute_time_s` 2.0, `stop_on_empty` true, `empty_retries` 2 (LAYER_EMPTY 연속 3번이면 DONE), `watchdog_s` 10, `done_topic` '' (비어 있지 않으면 이 토픽의 로봇 완료 보고 `{"ok","result",…}` 로 실행을 끝냄 — 트윈 브리지가 `/robot/execution_result` 로 냄; 박스 쪽 실패로 보고된 pick 은 막아 두고 다음 후보로, 전송 오류는 같은 명령 재시도), `execute_timeout_s` 120, `max_no_progress` 12 (명령을 못 낸 프레임이 연속 이만큼이면 DONE), `trigger_capture` false · `capture_service` `/robotsim_perception/capture` · `startup_delay_s` 3.0 (true 면 실행이 끝날 때마다 인식 노드에 다음 촬영을 요청한다 — 인식 노드를 `rate_hz:=0` 트리거 모드로 둘 때 쓴다) |
| 판단 로직 | `pick_executor/logic.py` — ROS 없이 pytest 19건 (`test/test_logic.py`). 실행 중 새 프레임은 busy 로 무시, 스탬프로 poses/status 짝 맞춤, 0 쿼터니언 거부, 로봇 실패 후보 차단, 무진전 종료 |
| 통합 테스트 | `test/test_pick_cycle_launch.py`(인식 노드 + 이 노드를 띄워 12 명령·DONE), `test_bag_replay_launch.py`(기록된 bag 재생 → OK 프레임마다 명령 1), `test_twin_cycle_launch.py`(트윈 서버 있을 때) — launch_testing, `colcon test` 가 돌린다 (docs/42 6-d) |

접근 방향은 포즈 쿼터니언에서 툴 +Z 축을 꺼낸다(인식 노드 규약: 툴 +Z = 접근). 탑다운이면 pre-pick 은 pick 의 15 cm 위, lift 는 40 cm 위.

## 실행

```bash
source /opt/ros/humble/setup.bash && source /mnt/e/Robot_Sim/ros2_ws/install/setup.bash
ros2 launch pick_executor pick_cycle.launch.py            # 합성 12박스: 촬영 → 명령 → 2초 → 재촬영 … 12개 다 집으면 소진 → DONE (+ rviz2)
ros2 topic echo /robot/target_poses --once               # 3점이 나오는지
ros2 topic echo /pick_executor/state                     # {"state": "EXECUTING", "sent": 3, "busy": 0, ...}
rqt_graph                                                # 두 노드가 토픽으로 이어진 그림 (서비스 호출은 rqt_graph 에 안 그려진다)
```

인식 노드가 이미 1 Hz 로 돌고 있다면 `ros2 run pick_executor pick_executor` 만 띄워도 된다(`trigger_capture` 기본 false).
같은 장면이 반복되면 `min_dist_m` 규칙으로 명령을 한 번만 낸다.

## 실제 셀과 다른 점

기본 launch(`pick_cycle.launch.py`)에서는 `/robot/target_poses` 를 받는 로봇 드라이버가 없고, 실행 완료는 `execute_time_s` 대기로 흉내 낸다.
`twin_bridge` 패키지의 `twin_cycle.launch.py` 로 띄우면 MuJoCo 셀 트윈의 UR10e 가 명령을 실제로 실행하고 `/robot/execution_result` 로
완료를 보고한다(`done_topic` 모드) — 실제 셀에서 MoveIt 또는 제조사 드라이버의 액션 서버가 하는 역할이다.
