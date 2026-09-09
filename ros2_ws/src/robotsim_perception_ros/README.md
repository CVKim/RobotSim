# robotsim_perception_ros

`robotsim_perception`(순수 numpy/opencv 인식 패키지)을 ROS2 노드로 감싼 것. ToF 프레임을 받아
박스·6-DoF 픽 포즈(`base_link`)·판정 상태를 토픽으로 내고, rviz2 마커로 보여 준다.

```
                 ┌──────────────────────── robotsim_perception (node) ────────────────────────┐
 source ──▶ Frame ─▶ runtime.decide() ─▶ boxes / plan ─▶ pose.box_to_pick_pose(T_base_cam) ─▶ msgs.* ─┐
  synthetic |                                                                                      │
  .mim 재생  |   HealthMonitor(드리프트)                                                              ▼
  twin://    |
                                                                    /tof/points  /perception/boxes  /perception/pick_poses
                                                                    /perception/next_pick  /perception/status  /diagnostics
                                                                    TF base_link → tof_optical      ~/capture (Trigger)
```

| 토픽 / 서비스 | 타입 | 프레임 | 내용 |
|---|---|---|---|
| `/tof/points` | `sensor_msgs/PointCloud2` | `tof_optical` | XYZI (m, 강도), best-effort QoS, `cloud_stride` 로 다운샘플 |
| `/perception/boxes` | `visualization_msgs/MarkerArray` | `base_link` | CUBE 상면(초록=계획, 주황=격자 보완, 회색=저신뢰) + TEXT(id·신뢰도·치수) + ARROW(다음 픽) |
| `/perception/pick_poses` | `geometry_msgs/PoseArray` | `base_link` | 픽 순서대로. 툴 +Z = 접근(아래), X = 상면 장축(yaw) |
| `/perception/next_pick` | `geometry_msgs/PoseStamped` | `base_link` | 위 첫 번째 |
| `/perception/status` | `std_msgs/String` | — | `runtime.Decision` 한 줄 JSON + source·health·seq |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | — | 판정 상태(OK/WARN/ERROR) + 카메라 드리프트 |
| TF (static) | `base_link → tof_optical` | — | `T_base_cam` (기본: 탑다운 4.183 m 예시. 실제 값은 핸드아이 캘리브레이션) |
| `~/capture` | `std_srvs/Trigger` | — | 프레임 1장 즉시 처리. `rate_hz:=0` 이면 트리거 모드 |

## 빌드 · 실행 (WSL2 Ubuntu 22.04, ROS2 Humble)

```bash
source /opt/ros/humble/setup.bash
cd /mnt/e/Robot_Sim/ros2_ws
colcon build --symlink-install
source install/setup.bash

ros2 launch robotsim_perception_ros perception.launch.py                 # 합성 장면 1 Hz + rviz2
ros2 launch robotsim_perception_ros perception.launch.py rviz:=false rate_hz:=0.0
ros2 service call /robotsim_perception/capture std_srvs/srv/Trigger       # 트리거 모드에서 1장
ros2 topic echo /perception/status --once
colcon test --packages-select robotsim_perception_ros && colcon test-result --verbose
```

실측 세션 재생(로컬 전용): `source:=/mnt/h/<세션 폴더 또는 상위 폴더>`.
MuJoCo 셀 트윈을 카메라로 쓰려면 `source:=twin://<호스트>:5555` (Windows 쪽에서 `tools/twin_server.py` 를 띄운다 — [docs/42 6-c](../../../docs/42_ROS2_핸즈온.md)).

인식 파라미터 중 결과에 크게 작용하는 둘:

| 파라미터 | 기본값 | 뜻 |
|---|---|---|
| `temporal_prior` | false | 직전 프레임에서 고른 층 깊이를 다음 프레임 후보로 준다. 박스가 1~3개 남았을 때 층을 잘못 고르던 문제를 덮는다(트윈 폐루프 43 → 78%) |
| `layer_roi_mm` | 0 (화면 중앙 ROI) | 층 히스토그램을 팔레트 크기로 한정한다. 절제 실험에서는 단독 이득이 없었다 |

## 두 번째 노드 — `cart_node` (대차 견인 고리)

카메라가 대차 데크를 약 40° 비스듬히 보는 데이터. 탑다운 가정 대신 **데크 플레이트 평면·림·레일에서 대차 좌표계를 매 프레임 추정**
(`robotsim_perception.cart.analyze_cart`)해 동적 TF 로 낸다.

| 토픽 / 서비스 | 타입 | 프레임 | 내용 |
|---|---|---|---|
| `/tof/points` | `sensor_msgs/PointCloud2` | `tof_optical` | 위와 동일 |
| `/cart/hook_pose` | `geometry_msgs/PoseStamped` | `tof_optical` | 위치 = 고리 wall_top, 자세 = 대차 축 (도킹 목표 자세) |
| `/cart/markers` | `visualization_msgs/MarkerArray` | `cart` | 데크 판 · 림 라인(v'=0) · 레일 안쪽 벽(u'=0, u'=간격) · 고리 기둥 + 라벨 |
| `/cart/status` | `std_msgs/String` | — | `CartResult` 한 줄 JSON (`hook_cart_mm`, `rim_yaw_deg`, `rail_gap_mm`, 피팅 MAD, 지연) |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | — | OK / NO_HOOK·NO_CART_FRAME(WARN) / NO_PLANE(ERROR) + 피팅 품질 |
| TF (**dynamic**) | `tof_optical → cart` | — | 프레임마다 `R_cart, t_cart` 재추정. 상태가 OK 가 아니면 내지 않는다 |
| `~/capture` | `std_srvs/Trigger` | — | 프레임 1장 즉시 처리 |

```bash
ros2 launch robotsim_perception_ros cart.launch.py          # 합성 대차 장면(카메라 높이 400~462 mm 순환) + rviz2 (고정 프레임 = cart)
ros2 run tf2_ros tf2_echo tof_optical cart
ros2 topic echo /cart/hook_pose --once
```

## 세 번째·네 번째 노드 — `dock_node` + `agv_sim_node` (대차 도킹)

`cart_node` 가 낸 고리 포즈를 **쓰는 쪽**이다. `dock_node` 가 고리까지의 중심선을 따라 속도 명령을 내고,
`agv_sim_node` 가 차동 구동 AGV 의 운동학으로 그 명령을 적분해 새 상대 자세를 낸다. 제어 규칙 자체는 ROS 없이
`robotsim_perception/dock.py` 에 있고(단위 테스트 10건), 노드는 그것을 토픽에 연결만 한다.

| 노드 | 구독 | 발행 | 서비스 |
|---|---|---|---|
| `dock_node` | `/cart/status`(고리 평면 좌표) · `/agv/rel_pose`(정지 확인) | `/cmd_vel` `geometry_msgs/Twist` · `/dock/state` JSON | `~/capture` 클라이언트 (기본 `/robotsim_cart/capture`) |
| `agv_sim_node` | `/cmd_vel` | `/agv/rel_pose` JSON (`hook_u_mm`, `rim_v_mm`, `yaw_deg`, `moving`, `cmd_seq`) | — |

**stop-and-go**: `agv_sim_node` 는 명령 하나를 `cmd_hold_s`(0.5 s) 만 적용하고 멈춘다. `dock_node` 는 `/agv/rel_pose` 의
`moving=false` 를 보고 다음 촬영을 요청하므로 측정은 항상 정지 자세의 것이다. 연속 주행에서는 인식 지연(약 0.45 s)에
묵은 명령이 겹쳐 요가 발산했다.

| 파라미터 | 기본값 | 뜻 |
|---|---|---|
| `dock_v_mm` | −200 | 결합 자리 — 고리가 카메라 앞 200 mm, 정면(u 0), 요 0 |
| `lookahead_mm` | 250(시뮬 기본 150) | 중심선 위 추종점까지 거리 |
| `v_max_mm_s` / `w_max_rad_s` | 150 / 0.5 | 속도 상한 |
| `tol_u_mm` / `tol_v_mm` / `tol_yaw_deg` | 10 / 10 / 2 | 결합 판정 허용치 (제어기 자기 측정 기준) |
| `trigger_capture` / `capture_service` / `startup_delay_s` | true / `/robotsim_cart/capture` / 2.0 | stop-and-go 촬영 요청 |
| `watchdog_s` | 3.0 | 측정이 이만큼 끊기면 정지하고 촬영을 다시 요청 |
| `cmd_hold_s` / `cmd_timeout_s` (agv_sim) | 0.5 / 1.0 | 0 이면 연속 주행 모드 |
| `init_hook_u_mm` / `init_rim_v_mm` / `init_yaw_deg` (agv_sim) | −80 / −450 / 5 | 시작 자세 |

```bash
ros2 launch robotsim_perception_ros dock.launch.py                       # 합성 대차 + 도킹 + AGV 시뮬 + rviz2
ros2 launch robotsim_perception_ros dock.launch.py init_hook_u_mm:=120.0 init_rim_v_mm:=-480.0 init_yaw_deg:=-6.0
ros2 topic echo /dock/state --once
```

결합까지의 기록과 수치는 [docs/42 8-c](../../../docs/42_ROS2_핸즈온.md) · `results/cart_dock.json`.

## 한계

- 실로봇·센서 드라이버 없음. `source` 는 합성 또는 파일 재생이며, 실제 셀에서는 이 자리에 센서 SDK 노드가 온다.
- `T_base_cam` 기본값은 설치 예시일 뿐이다. 값이 틀리면 `/perception/pick_poses` 가 통째로 밀린다.
- 픽 포즈를 **소비하는** 쪽은 별도 패키지 `ros2_ws/src/pick_executor` 다(3점 궤적 + 재촬영 사이클). 실제 로봇 드라이버는 없다.
- 도킹의 AGV 는 운동학 플랜트다. 바퀴 미끄러짐·가감속 한계·실제 통신 지연은 없고, 대차 장면도 합성이다.
