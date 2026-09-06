# robotsim_perception_ros

`robotsim_perception`(순수 numpy/opencv 인식 패키지)을 ROS2 노드로 감싼 것. ToF 프레임을 받아
박스·6-DoF 픽 포즈(`base_link`)·판정 상태를 토픽으로 내고, rviz2 마커로 보여 준다.

```
                 ┌──────────────────────── robotsim_perception (node) ────────────────────────┐
 source ──▶ Frame ─▶ runtime.decide() ─▶ boxes / plan ─▶ pose.box_to_pick_pose(T_base_cam) ─▶ msgs.* ─┐
  synthetic |                                                                                      │
  .mim 재생  |   HealthMonitor(드리프트)                                                              ▼
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

## 한계

- 실로봇·센서 드라이버 없음. `source` 는 합성 또는 파일 재생이며, 실제 셀에서는 이 자리에 센서 SDK 노드가 온다.
- `T_base_cam` 기본값은 설치 예시일 뿐이다. 값이 틀리면 `/perception/pick_poses` 가 통째로 밀린다.
- 픽 포즈를 **소비하는** 로봇 제어 노드는 이 패키지에 없다 (docs/42 의 실습 과제).
