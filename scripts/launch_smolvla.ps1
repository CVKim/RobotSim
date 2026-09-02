# SmolVLA 파인튜닝 (GPU1) — 데이터셋 카메라 이름 매핑 포함
$env:CUDA_VISIBLE_DEVICES = "1"
$env:WANDB_MODE = "offline"
Start-Process -WindowStyle Hidden -FilePath "E:\Robot_Sim\.venv\Scripts\lerobot-train.exe" `
  -ArgumentList @(
    "--policy.path=E:\Robot_Sim\models\smolvla_base",
    "--policy.push_to_hub=false",
    "--policy.empty_cameras=1",
    '--rename_map={\"observation.images.side\":\"observation.images.camera1\",\"observation.images.up\":\"observation.images.camera2\"}',
    "--dataset.repo_id=lerobot/svla_so101_pickplace",
    "--batch_size=2", "--steps=20000", "--save_freq=10000", "--log_freq=200",
    "--output_dir=E:\Robot_Sim\runs\smolvla_so101",
    "--policy.device=cuda", "--wandb.enable=false"
  ) `
  -RedirectStandardOutput "E:\Robot_Sim\runs\smolvla_train.log" `
  -RedirectStandardError "E:\Robot_Sim\runs\smolvla_train.err"
Write-Host "smolvla relaunched with rename_map + empty_cameras=1"
