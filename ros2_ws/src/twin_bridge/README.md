# twin_bridge

MuJoCo 셀 트윈(`tools/twin_server.py`, Windows 프로세스)을 ROS 그래프의 **로봇 드라이버 자리**에 놓는 노드.
pick_executor 가 낸 3점 명령을 트윈의 UR10e 가 실제로 실행하고(IK·관절 보간·흡착·목적지 배치), 결과를 보고한다.
같은 트윈이 인식 노드의 **카메라**이기도 하다(`source:=twin://…`): 로봇이 박스를 옮기면 다음 프레임에 그대로 보인다.

```
                 ┌──────────── WSL2 (ROS2 Humble) ────────────┐        ┌── Windows (.venv, MuJoCo) ──┐
 perception_node ──/perception/pick_poses──▶ pick_executor ──/robot/target_poses──▶ twin_bridge ──TCP──▶ twin_server
      ▲  ~/capture (Trigger) ◀───────────────────┘  ▲                                   │              │  UR10e + 트랙, ToF 렌더
      │                                              └──/robot/execution_result (JSON)───┘              │
      └──── source=twin://host:port (프레임 요청) ──────────────────────────────────────TCP────────────▶│
```

| 항목 | 내용 |
|---|---|
| 구독 | `/robot/target_poses` `geometry_msgs/PoseArray` (pre-pick, pick, lift; base_link) |
| 퍼블리시 | `/robot/execution_result` `std_msgs/String` JSON `{"ok","result","cycle_s","pick_err_mm","remaining","placed","cmd_stamp",…}` (실행 중 받은 명령·잘못된 명령도 `busy`/`bad_request` 로 보고) · `/robot/tcp_path` `MarkerArray`(이번 실행의 TCP 경로) · `/twin/state` JSON · TF `base_link → tcp` |
| 서비스 | `~/reset` `std_srvs/Trigger` — 트윈 장면 초기화 |
| 파라미터 | `host` ('' = WSL2 기본 게이트웨이 = Windows 호스트), `port` 5555, `frame_id` base_link, `timeout_s` 600, `state_period_s` 1.0 |
| 순수 로직 | `twin_bridge/logic.py` — 메시지 ↔ 트윈 요청/응답 변환, pytest 4건 (`test/`) |

실행은 수십 초 걸리므로(시뮬 8 s 를 실시간보다 느리게 계산) 작업 스레드에서 트윈을 부르고 그동안 노드는 spin 한다.
pick_executor 는 `done_topic:=/robot/execution_result` 로 이 보고를 기다린다(타이머 흉내 대신). 실패(`ik_unreachable`, `path_collision`,
`grasp_miss`, …)로 보고된 pick 은 pick_executor 가 막아 두고 다음 프레임에서 다음 후보를 보낸다.

## 실행

```powershell
# Windows: 트윈 서버 (UR10e + 트랙, 상층 12박스)
.venv\Scripts\python.exe tools\twin_server.py --arm track --boxes 12 --seed 500
```
```bash
# WSL2
ros2 launch twin_bridge twin_cycle.launch.py              # 인식 노드(트윈 소스, 트리거 모드) + pick_executor + twin_bridge + rviz2
ros2 topic echo /robot/execution_result                   # {"ok": true, "result": "placed", "cycle_s": 7.9, "remaining": 11, ...}
ros2 topic echo /twin/state --once
```

한 번에 끝까지(서버 기동 → 사이클 → rviz 캡처 → 서버 종료): `powershell -ExecutionPolicy Bypass -File scripts\twin_cycle_demo.ps1`.

기록(12박스, UR10e + 트랙): 프레임 13, 명령 10, 배치 10, 실패 0, 남은 2개는 층 선택 미검출(LAYER_EMPTY 3회)로 종료. 시뮬 사이클 7.5 s(6.3~10.9), 픽 위치 오차(pick_err) 중앙값 7 mm,
실행 1회 계산 1.2 s. 첫 실행에서 12개 중 6개가 놓는 순간 튕겨 나간 것을 추적해 픽 요 규약·페이로드 충돌 검사·목적지 슬롯 겹침 세 결함을 고쳤다
(docs/42 6-c, docs/41 7절). 수치: `results/ros2_twin_cycle.json`.

## 실제 셀과 다른 점

트윈 서버가 로봇 드라이버·센서 드라이버·PLC('팔레트 비움' 신호 = 정답 기준 남은 박스 0)를 한 프로세스가 대신한다.
실제로는 이 자리에 MoveIt/제조사 드라이버의 액션 서버와 센서 SDK 노드가 오고, 이 노드는 없어진다 — 바뀌지 않는 것은
pick_executor 가 완료 보고를 기다리고 실패한 후보를 막는 쪽의 로직이다.
