# 팔레타이징 RL 다중 시드 + action mask ablation
# 기존 결과의 약점: 단일 런·시드 미설정 -> 학습 재현 미보장, 런 간 분산 미보고, 계획된 ablation 미이행.
# 이 스크립트는 시드 3개(마스크 O) + 시드 3개(마스크 X)를 순차 학습하고 각 런의 eval.json 을 남긴다.
$py = "E:\Robot_Sim\.venv\Scripts\python.exe"
$root = "E:\Robot_Sim\runs\palletize_seeds"
New-Item -ItemType Directory -Force -Path $root | Out-Null

foreach ($seed in 0, 1, 2) {
    $out = Join-Path $root "mask_s$seed"
    if (Test-Path (Join-Path $out "eval.json")) { Write-Output "skip mask_s$seed"; continue }
    Write-Output "=== training mask seed $seed ==="
    & $py E:\Robot_Sim\tools\palletize_train.py --steps 1500000 --seed $seed --device cuda:1 --out $out
}
foreach ($seed in 0, 1, 2) {
    $out = Join-Path $root "nomask_s$seed"
    if (Test-Path (Join-Path $out "eval.json")) { Write-Output "skip nomask_s$seed"; continue }
    Write-Output "=== training no-mask seed $seed ==="
    & $py E:\Robot_Sim\tools\palletize_train.py --steps 1500000 --seed $seed --device cuda:1 --out $out --no-mask
}
Write-Output "=== aggregating ==="
& $py E:\Robot_Sim\tools\palletize_aggregate.py
Write-Output "ALL DONE"
