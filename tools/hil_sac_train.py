# -*- coding: utf-8 -*-
"""가상환경 제어모델 학습 (JD: '가상환경에서 제어 모델을 학습하여 적용').

gym-hil(MuJoCo Franka) PandaPickCubeBase-v0 를 SAC로 학습 → 성공률 평가 → 저장.
이후 이 정책을 전문가로 데모를 수집해 모방학습(BC)으로 증류하는 사이클의 1단계.
실행: python tools/hil_sac_train.py --steps 400000 --device cuda
"""
import argparse
import json
from pathlib import Path

import gym_hil  # noqa: F401 (env 등록)
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

ENV_ID = "gym_hil/PandaPickCubeBase-v0"


def make_env():
    env = gym.make(ENV_ID, image_obs=False, reward_type="dense")
    return Monitor(env)


def evaluate(model, n=50):
    env = make_env()
    succ, rets = 0, []
    for _ in range(n):
        obs, _ = env.reset()
        done, ret = False, 0.0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            ret += r
            done = term or trunc
        succ += bool(info.get("succeed", False))
        rets.append(ret)
    env.close()
    return succ / n, float(np.mean(rets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=r"E:\Robot_Sim\runs\hil_sac")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = make_env()
    print("obs space:", env.observation_space)
    print("act space:", env.action_space)
    policy = "MultiInputPolicy" if isinstance(env.observation_space, gym.spaces.Dict) else "MlpPolicy"

    model = SAC(policy, env, device=args.device, verbose=1,
                buffer_size=400_000, batch_size=512, learning_starts=5_000,
                train_freq=1, gradient_steps=1, tensorboard_log=None)
    model.learn(total_timesteps=args.steps,
                callback=CheckpointCallback(100_000, str(out / "ckpt"), "sac"),
                log_interval=20)
    model.save(str(out / "final"))

    sr, ret = evaluate(model)
    res = {"env": ENV_ID, "steps": args.steps, "success_rate": sr, "mean_return": ret}
    (out / "eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(res)


if __name__ == "__main__":
    main()
