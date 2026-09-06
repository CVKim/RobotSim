# 아키텍처

## 1. 한 장 요약

![architecture](../assets/architecture.svg)

- 생성: `python tools/make_architecture_diagram.py` → `assets/architecture.svg` + `docs/diagrams/architecture.excalidraw`
  (같은 스펙에서 둘 다 생성하므로 어긋나지 않는다. 수치가 바뀌면 생성기의 텍스트만 고치고 재실행)
- 손으로 다듬으려면 `architecture.excalidraw` 를 [excalidraw.com](https://excalidraw.com) 에서 연다

## 2. 인식 파이프라인 (무학습 기하) — `robotsim_perception/geometry.py`

```mermaid
flowchart TD
  A["ToF 프레임 X·Y·D·I (mm)"] --> B["중앙 ROI 깊이 히스토그램<br/>후보 피크 최대 8개"]
  B --> C["층 마스크 (±40mm) + 깊이 그래디언트 필터"]
  A --> D["강도(I) 채널 Canny 에지<br/>= 박스 이음새 (D 와 픽셀 정합)"]
  C --> E["마스크 − 에지 → 연결요소<br/>(유효 픽셀만 — 센티넬 오염 방지)"]
  D --> E
  E --> F["직사각형성·종횡비 필터 → X/Y mm 맵 minAreaRect 치수"]
  F --> G["박스별 신뢰도<br/>충전율·평면 RMS·직사각형성·SKU 부합"]
  G --> H["층 선택: 신뢰 박스가 있는 층 중 최근접"]
  H --> I["격자(방향·치수·피치) 추정 → 결손 셀 보완 'inferred'"]
  I --> J["평면 피팅 → 중심·법선·기울기 (pose.py)<br/>T_base_cam → 6-DoF 픽 포즈"]
  J --> K["runtime.decide(): OK / RETAKE / LOW_CONFIDENCE / LAYER_EMPTY / NO_SURFACE"]
```

## 3. 좌표계

| 프레임 | 정의 | 어디서 |
|---|---|---|
| `tof_optical` | 카메라 광학. x 오른쪽, y 아래, z 전방(깊이). 실측 `.mim` 의 X/Y/D 와 동일 | `frame.py`, ROS `/tof/points` |
| `base_link` | 로봇 베이스. `T_base_cam`(4×4, mm)으로 변환. 기본값은 탑다운 설치 예시 (카메라 높이 4.183 m, R = diag(1,−1,−1)) | `pose.py`, ROS TF static |
| 트윈 월드 | MuJoCo. `ToF_X = world_X`, `ToF_Y = −world_Y`, `ToF_D = CAM_H − world_Z` | `sim/cell_twin.py` |
| `cart` (대차) | 대차 구조물 프레임. 원점 = 플레이트 림 라인 × 왼쪽 레일 안쪽 벽, u' 림 방향, v' 림에 수직(카메라 쪽 +), h 플레이트 법선. **프레임마다 재추정**, `tof_optical` 의 자식 동적 TF | `cart.py` `_cart_frame`, ROS `msgs.cart_transform` |

실제 `T_base_cam` 값은 핸드아이 캘리브레이션이 필요하다(하드웨어). 값이 틀리면 픽 포즈가 통째로 밀리므로
`runtime.HealthMonitor` 가 정적 배경 깊이 편차로 드리프트를 감시한다.

## 4. ROS2 그래프 — `ros2_ws/src/robotsim_perception_ros`

```mermaid
flowchart LR
  SRC["source<br/>synthetic | .mim 재생<br/>(실셀: 센서 드라이버)"] --> N["perception_node<br/>decide() + HealthMonitor"]
  N -->|PointCloud2| P["/tof/points"]
  N -->|MarkerArray| M["/perception/boxes"]
  N -->|PoseArray base_link| Q["/perception/pick_poses"]
  N -->|String JSON| S["/perception/status"]
  N -->|DiagnosticArray| D["/diagnostics"]
  N -.->|TF static| T["base_link → tof_optical"]
  PLC["시퀀서 / PLC 역할"] -->|std_srvs/Trigger| N
  P --> R["rviz2"]
  M --> R
  Q --> R
  Q --> X["pick_executor<br/>(docs/42 실습 과제 — 직접 작성)"]
```

대차(카트) 모드는 별도 노드다. 카메라가 데크를 약 40° 비스듬히 보므로 탑다운 가정을 쓸 수 없고,
좌표계를 **대차 구조물(플레이트 평면 · 림 · 레일)** 에서 매 프레임 추정해 동적 TF 로 낸다:

```mermaid
flowchart LR
  SRC2["source<br/>synthetic_cart | .mim 재생"] --> C["cart_node<br/>cart.analyze_cart()"]
  C -->|PointCloud2 tof_optical| P2["/tof/points"]
  C -->|PoseStamped tof_optical<br/>위치 = 고리, 자세 = 대차 축| HP["/cart/hook_pose"]
  C -->|MarkerArray cart| CM["/cart/markers<br/>데크 판 · 림 · 레일 · 고리"]
  C -->|String JSON| CS["/cart/status"]
  C -->|DiagnosticArray| CD["/diagnostics"]
  C -.->|TF 동적| T2["tof_optical → cart"]
  P2 --> R2["rviz2"]
  CM --> R2
  HP --> R2
```
