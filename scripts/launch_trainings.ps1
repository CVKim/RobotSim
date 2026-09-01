# 두 GPU에 학습 분산 실행 (독립 프로세스로 — 셸 종료와 무관하게 지속)
# GPU0: PushT Diffusion Policy (LeRobot)  /  GPU1: 팔레타이징 MaskablePPO
$venv = "E:\Robot_Sim\.venv\Scripts"
$env:WANDB_MODE = "offline"

New-Item -ItemType Directory -Force E:\Robot_Sim\runs\pusht_diffusion | Out-Null
New-Item -ItemType Directory -Force E:\Robot_Sim\runs\palletize_ppo | Out-Null

# --- GPU0: PushT diffusion ---
$env:CUDA_VISIBLE_DEVICES = "0"
Start-Process -WindowStyle Hidden -FilePath "$venv\lerobot-train.exe" `
  -ArgumentList @(
    "--policy.type=diffusion", "--dataset.repo_id=lerobot/pusht", "--env.type=pusht",
    "--batch_size=64", "--steps=100000", "--env_eval_freq=20000", "--save_freq=20000",
    "--output_dir=E:\Robot_Sim\runs\pusht_diffusion", "--policy.device=cuda",
    "--wandb.enable=false"
  ) `
  -RedirectStandardOutput "E:\Robot_Sim\runs\pusht_train.log" `
  -RedirectStandardError "E:\Robot_Sim\runs\pusht_train.err"

# --- GPU1: 팔레타이징 PPO ---
$env:CUDA_VISIBLE_DEVICES = "1"
Start-Process -WindowStyle Hidden -FilePath "$venv\python.exe" `
  -ArgumentList @("E:\Robot_Sim\tools\palletize_train.py", "--steps", "1500000", "--device", "cuda") `
  -RedirectStandardOutput "E:\Robot_Sim\runs\palletize_train.log" `
  -RedirectStandardError "E:\Robot_Sim\runs\palletize_train.err"

Write-Host "launched: pusht(GPU0) + palletize(GPU1). logs in E:\Robot_Sim\runs\*\train.log"
