# 팔레타이징 RL — 한 변형(mask 또는 nomask)의 시드 3개를 순차 학습.
# 두 변형을 GPU 0/1 에 나눠 병렬로 띄우기 위해 분리했다 (scripts/palletize_launch.ps1 참조).
#
# 스텝 500k: 원 런(1.5M)의 학습곡선이 500k 부근에서 플래토에 도달한다(docs/21 #8).
#            6런을 하룻밤 안에 끝내기 위한 선택이며, 비교 시 스텝 수를 함께 보고할 것.
param(
    [Parameter(Mandatory = $true)][ValidateSet("mask", "nomask")][string]$Variant,
    [Parameter(Mandatory = $true)][string]$Device,
    [int]$Steps = 500000
)

$ErrorActionPreference = "Continue"
$py    = "E:\Robot_Sim\.venv\Scripts\python.exe"
$train = "E:\Robot_Sim\tools\palletize_train.py"
$root  = "E:\Robot_Sim\runs\palletize_seeds"
$prog  = Join-Path $root "progress_$Variant.txt"

New-Item -ItemType Directory -Force -Path $root | Out-Null
"START $Variant device=$Device steps=$Steps $(Get-Date -Format s)" |
    Out-File -FilePath $prog -Encoding utf8

foreach ($seed in 0, 1, 2) {
    $tag = "${Variant}_s${seed}"
    $out = Join-Path $root $tag
    if (Test-Path (Join-Path $out "eval.json")) {
        "SKIP  $tag $(Get-Date -Format s)" | Out-File -FilePath $prog -Append -Encoding utf8
        continue
    }
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    "BEGIN $tag $(Get-Date -Format s)" | Out-File -FilePath $prog -Append -Encoding utf8
    $log = Join-Path $out "train.log"
    if ($Variant -eq "mask") {
        & $py $train --steps $Steps --seed $seed --device $Device --out $out *>&1 |
            Out-File -FilePath $log -Encoding utf8
    } else {
        & $py $train --steps $Steps --seed $seed --device $Device --out $out --no-mask *>&1 |
            Out-File -FilePath $log -Encoding utf8
    }
    "END   $tag exit=$LASTEXITCODE $(Get-Date -Format s)" | Out-File -FilePath $prog -Append -Encoding utf8
}
"DONE $Variant $(Get-Date -Format s)" | Out-File -FilePath $prog -Append -Encoding utf8
