# PushT 체크포인트 일괄 평가 — 학습 완료 후 실행 (GPU0)
# 산출: runs/pusht_eval/<step>/eval_info.json (성공률 포함)
$venv = "E:\Robot_Sim\.venv\Scripts"
$env:CUDA_VISIBLE_DEVICES = "0"
$ckRoot = "E:\Robot_Sim\runs\pusht_diffusion\checkpoints"

Get-ChildItem $ckRoot -Directory | Sort-Object Name | ForEach-Object {
    $step = $_.Name
    $model = Join-Path $_.FullName "pretrained_model"
    $out = "E:\Robot_Sim\runs\pusht_eval\$step"
    Write-Host "=== eval checkpoint $step ==="
    & "$venv\lerobot-eval.exe" `
        --policy.path=$model `
        --env.type=pusht `
        --eval.n_episodes=50 `
        --eval.batch_size=10 `
        --eval.use_async_envs=false `
        --output_dir=$out `
        --policy.device=cuda *>> "E:\Robot_Sim\runs\pusht_sweep_full.log"
}
