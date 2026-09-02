# 파이프라인 아키텍처

> GitHub이 Mermaid를 직접 렌더링합니다. 편집용 손그림 버전: [`diagrams/pipeline.excalidraw`](diagrams/pipeline.excalidraw) (excalidraw.com에서 열기)

## 1. 전체 맵 — 데이터에서 데모까지

```mermaid
flowchart LR
  subgraph DATA["실데이터 (비공개, 이적재 공정)"]
    MIM["ToF .mim (Matrox MIL/TIFF)<br/>X·Y·D·I 640×480, mm"]
    RGBI["RGB 5MP"]
  end

  subgraph PERC["인식 · 분석 (tools/)"]
    LOADER["mim_loader.py<br/>TIFF 디코드 + valid mask"]
    TOP["binpick_topface.py<br/>층 깊이 검출 + I채널 에지로 박스 분리<br/>148박스, 293×219mm ±3%"]
    PICKP["binpick_pickpoints.py<br/>평면 피팅 픽포인트 + 슬롯 추적<br/>픽 순서 자동 복원"]
    HOOK["hook_analysis.py<br/>RANSAC 평면 + 후크 클러스터<br/>4/4 검출, 높이 반복성 1.2mm"]
    NOISE["tof_noise_study.py<br/>정적픽셀 12.7만개 실측<br/>σ(mm) = 180.3 × I^-0.805"]
  end

  subgraph RL["시뮬레이션 · 강화학습"]
    ENV["palletize_env.py<br/>실측 치수 시드 heightmap 환경<br/>DBL 휴리스틱 56.6박스/59.8%"]
    PPO["palletize_train.py<br/>MaskablePPO 1.5M steps<br/>64.9박스/68.5% (+14.7%)"]
  end

  subgraph DEMO["통합 데모 (인식 ↔ 정책)"]
    PICKN["pick_next.py<br/>최소 동선 픽 제안<br/>실제 순서 대비 top-2 81%"]
    PLACE["transfer_place_v3.py<br/>목적지 스택 위 배치 제안"]
  end

  MIM --> LOADER
  LOADER --> TOP --> PICKP
  LOADER --> HOOK
  LOADER --> NOISE
  TOP -- "치수 분포 293×219×283" --> ENV --> PPO
  NOISE -. "합성데이터 노이즈 주입 (예정 T5)" .-> ENV
  PICKP --> PICKN
  PICKP -- "적재 상태 이식" --> PLACE
  PPO --> PLACE
```

## 2. 인식 파이프라인 상세 (무학습 기하)

```mermaid
flowchart TD
  A["ToF 프레임 D·I·X·Y"] --> B["중앙 ROI 깊이 히스토그램<br/>최대 피크 = 상면 층 깊이"]
  B --> C["층 마스크 (±40mm)"]
  A --> D["강도(I) 채널 Canny 에지<br/>= 박스 이음새 (D와 픽셀 정합)"]
  C --> E["마스크 − 에지 → 연결요소"]
  D --> E
  E --> F["직사각형성·종횡비 필터"]
  F --> G["X/Y 좌표맵(mm) minAreaRect<br/>→ 실측 치수 L×W"]
  F --> H["박스별 3D 평면 피팅(SVD)<br/>→ 픽포인트: 중심+법선+기울기"]
  H --> I["기준 레이아웃 슬롯 매칭<br/>→ 세션 차분 = 실제 픽 순서"]
```

## 3. 이적재 사이클 — 데이터로 복원·검증된 흐름

```mermaid
flowchart LR
  S["소스 팔레트<br/>(박스 12→3개 감소)"] -- "① PICK NEXT<br/>최소 동선 박스 선택" --> R["로봇 이송<br/>(동선 mm 산출)"]
  R -- "② PLACE<br/>PPO 배치 제안" --> DST["목적지 스택<br/>(770→1260mm 성장)"]
  DST -- "만재 시 반출 (높이 0 복귀)" --> OUT["완성 팔레트"]
  GT["세션 시계열<br/>(30 캡처)"] -. "검증: 실제 픽 순위 평균 1.69<br/>스택 성장 계단 = 질량 보존" .-> S
```

## 4. 학습 인프라 (RTX 3080 10GB ×2)

```mermaid
flowchart TD
  subgraph GPU0["GPU 0"]
    P["LeRobot PushT Diffusion 263M<br/>(파이프라인 검증용)"]
  end
  subgraph GPU1["GPU 1"]
    S1["팔레타이징 MaskablePPO ✅ 완료"] --> S2["SmolVLA-450M 파인튜닝<br/>BS2, VRAM 피크 4.7GB"]
  end
  M["Monitor 감시<br/>(마일스톤·에러·완료 알림)"] --- GPU0
  M --- GPU1
  N["교훈: torch 2.9+ Win 리그레션→2.8 고정<br/>심링크 폴백 패치, extras 일괄 설치"] -.-> M
```
