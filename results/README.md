# results — 공개 집계 결과

README 의 수치를 검증할 수 있도록 실험 산출 JSON 을 모아 둔 디렉터리.
생성: `python tools/publish_results.py` (원본은 `explore/`·`runs/`, 둘 다 비공개).

회사 원본 데이터(깊이맵·좌표·RGB)는 포함하지 않으며, 세션 폴더명(현장 타임스탬프)은
`frame_01`..`frame_34` 으로 익명화했다 (docs/03 공개 정책).

| 파일 | 내용 |
|---|---|
| [detector_robustness.json](detector_robustness.json) | 기하 검출 v1 vs v2 강건성 (실측 30프레임, 교란 격자) |
| [twin_detect_accuracy.json](twin_detect_accuracy.json) | 셀 트윈 절대 정답 기반 검출 정확도 (48장면) |
| [twin_closed_loop.json](twin_closed_loop.json) | 셀 트윈 인식->제어 폐루프 (oracle vs 인식 구동) |
| [sim2real_reeval.json](sim2real_reeval.json) | 학습 모델 v2 정답 재평가 (홀드아웃 24장) |
| [sim2real_gap.json](sim2real_gap.json) | sim2real 사다리 (합성 전용 -> 실측) |
| [sim2real_cotrain.json](sim2real_cotrain.json) | 합성 + 실측 6장 co-training |
| [sdg_tool_compare.json](sdg_tool_compare.json) | SDG 툴 비교 (Isaac Replicator vs BlenderProc) |
| [seg_vs_geometric.json](seg_vs_geometric.json) | 학습 세그멘테이션 vs 기하 검출 (열화 그리드 포함) |
| [hook_repeatability.json](hook_repeatability.json) | 대차 후크 위치 반복성 (정렬 방식 4종) |
| [tof_noise_model.json](tof_noise_model.json) | ToF 시간 노이즈 실측 모델 |
| [palletize_rl.json](palletize_rl.json) | 팔레타이징 MaskablePPO vs DBL 휴리스틱 |
| [imitation_bc.json](imitation_bc.json) | 모방학습 BC/DART (가상 Franka) |
| [palletize_multiseed.json](palletize_multiseed.json) | 팔레타이징 RL 다중 시드 + action mask ablation (시드 3개씩) |
