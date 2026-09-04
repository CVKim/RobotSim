# 팔레타이징 RL 다중 시드 + action mask ablation
#
# 기존 결과의 약점: 단일 런·시드 미설정 -> 학습 재현 미보장, 런 간 분산 미보고, 계획된 ablation 미이행.
# 시드 3개(마스크 O) + 시드 3개(마스크 X)를 순차 학습하고 각 런의 eval.json 을 남긴다.
#
# 스텝 수 800k: 원 런(1.5M)의 학습곡선이 500k 이후 플래토라 800k 면 같은 수준에 도달하며
#               6런을 현실적인 시간에 끝낼 수 있다. 비교 시 스텝 수를 함께 보고할 것.
# 진행 상황은 runs/palletize_seeds/progress.txt, 런별 출력은 각 폴더의 train.log 에 남는다.

$ErrorActionPreference = "Continue"
$py   = "E:\Robot_Sim\.venv\Scripts\python.exe"
$train = "E:\Robot_Sim\tools\palletize_train.py"
$agg   = "E:\Robot_Sim\tools\palletize_aggregate.py"
$root = "E:\Robot_Sim\runs\palletize_seeds"
$prog = Join-Path $root "progress.txt"

New-Item -ItemType Directory -Force -Path $root | Out-Null
"START $(Get-Date -Format s)" | Out-File -FilePath $prog -Encoding utf8

foreach ($variant in "mask", "nomask") {
    foreach ($seed in 0, 1, 2) {
        $tag = "${variant}_s${seed}"
        $out = Join-Path $root $tag
        if (Test-Path (Join-Path $out "eval.json")) {
            "SKIP  $tag $(Get-Date -Format s)" | Out-File -FilePath $prog -Append -Encoding utf8
            continue
        }
        New-Item -ItemType Directory -Force -Path $out | Out-Null
        "BEGIN $tag $(Get-Date -Format s)" | Out-File -FilePath $prog -Append -Encoding utf8
        $log = Join-Path $out "train.log"
        if ($variant -eq "mask") {
            & $py $train --steps 800000 --seed $seed --device cuda:1 --out $out *>&1 |
                Out-File -FilePath $log -Encoding utf8
        } else {
            & $py $train --steps 800000 --seed $seed --device cuda:1 --out $out --no-mask *>&1 |
                Out-File -FilePath $log -Encoding utf8
        }
        "END   $tag exit=$LASTEXITCODE $(Get-Date -Format s)" | Out-File -FilePath $prog -Append -Encoding utf8
    }
}

"AGGREGATE $(Get-Date -Format s)" | Out-File -FilePath $prog -Append -Encoding utf8
& $py $agg *>&1 | Out-File -FilePath (Join-Path $root "aggregate.log") -Encoding utf8
"ALL DONE $(Get-Date -Format s)" | Out-File -FilePath $prog -Append -Encoding utf8
