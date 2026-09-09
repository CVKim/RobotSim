# 트윈 연결 사이클 데모 한 번에: Windows 에서 트윈 서버를 띄우고, WSL2 의 ROS2 그래프(인식 + pick_executor + twin_bridge + rviz2)를
# 끝까지 돌린 뒤 서버를 내린다. 산출물: assets/ros2_twin_cycle_rviz.png, explore/ros2/twin_cycle.log(+_results.jsonl),
# explore/twin/twin_server.jsonl (트윈 쪽 프레임·실행 기록).
#   powershell -ExecutionPolicy Bypass -File scripts\twin_cycle_demo.ps1 [-Arm track|fixed|none] [-Boxes 12] [-Seed 500] [-Port 5555]
param(
    [string]$Arm = "track",
    [int]$Boxes = 12,
    [int]$Seed = 500,
    [int]$Port = 5555,
    [int]$MaxWaitS = 1500,
    [string]$UseAction = "false"      # true = 픽 명령을 액션(/robot/execute_pick)으로
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$pidfile = Join-Path $root "explore\twin\twin_server.pid"
$serverLog = Join-Path $root "explore\twin\twin_server.log"
New-Item -ItemType Directory -Force (Join-Path $root "explore\twin") | Out-Null

$server = Start-Process -FilePath $py -ArgumentList @("tools\twin_server.py", "--arm", $Arm, "--boxes", $Boxes, "--seed", $Seed,
    "--port", $Port, "--pidfile", $pidfile) -WorkingDirectory $root -PassThru -NoNewWindow `
    -RedirectStandardOutput $serverLog -RedirectStandardError (Join-Path $root "explore\twin\twin_server.err")
try {
    $ok = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 1
        if ((Test-Path $serverLog) -and (Select-String -Path $serverLog -Pattern "listening" -Quiet)) { $ok = $true; break }
    }
    if (-not $ok) { throw "twin server did not start (see $serverLog)" }
    Get-Content $serverLog | Select-Object -First 2
    # WSL 쪽은 기본 게이트웨이로 Windows 호스트를 찾는다 (host 인자 비움)
    wsl -d Ubuntu-22.04 -e bash /mnt/e/Robot_Sim/scripts/wsl_twin_cycle_demo.sh "-" "/mnt/e/Robot_Sim/assets/ros2_twin_cycle_rviz.png" "/mnt/e/Robot_Sim/explore/ros2/twin_cycle.log" $MaxWaitS $UseAction
}
finally {
    if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }
    Write-Host "--- twin server log tail:"
    Get-Content $serverLog -Tail 15
}
