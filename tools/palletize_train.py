# -*- coding: utf-8 -*-
"""팔레타이징 배치 RL 학습 — MaskablePPO (sb3-contrib).

관측: heightmap(44x44)/1500 + 현재 박스 (L,W,H)/1500 → flatten concat (1939,)
액션: Discrete(44*44*2) — PalletizeEnv의 (x,y,rot) ravel 인덱스와 동일 규약
마스크: PalletizeEnv.action_mask().ravel()
실행:  python tools/palletize_train.py --steps 1500000 --device cuda:1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from palletize_env import GRID, PalletizeEnv  # noqa: E402

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402


class PalletizeGym(gym.Env):
    """PalletizeEnv → gymnasium 어댑터 (MaskablePPO action_masks 프로토콜)."""

    def __init__(self, seed=None):
        super().__init__()
        self.env = PalletizeEnv(max_boxes=120)
        self._next_seed = seed
        self.action_space = spaces.Discrete(GRID * GRID * 2)
        self.observation_space = spaces.Box(
            0.0, 2.0, shape=(GRID * GRID + 3,), dtype=np.float32)

    def _obs(self):
        hm = self.env.heightmap.astype(np.float32).ravel() / 1500.0
        box = np.asarray(self.env.box, dtype=np.float32) / 1500.0
        return np.concatenate([hm, box])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        s = seed if seed is not None else self._next_seed
        self.env.reset(seed=s)
        if self._next_seed is not None:
            self._next_seed += 1
        return self._obs(), {}

    def step(self, action):
        _, reward, done, info = self.env.step(int(action))
        return self._obs(), float(reward), bool(done), False, info

    def action_masks(self):
        return self.env.action_mask().ravel()


def evaluate(model, n_episodes=100):
    utils, boxes = [], []
    for seed in range(n_episodes):
        genv = PalletizeGym()
        obs, _ = genv.reset(seed=seed)
        done = False
        while not done:
            mask = genv.action_masks()
            if not mask.any():
                break
            act, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, _, done, _, _ = genv.step(act)
        utils.append(genv.env.utilization())
        boxes.append(genv.env.n_placed)
    return {
        "ppo_utilization_mean": float(np.mean(utils)),
        "ppo_utilization_std": float(np.std(utils)),
        "ppo_boxes_mean": float(np.mean(boxes)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=r"E:\Robot_Sim\runs\palletize_ppo")
    ap.add_argument("--seed", type=int, default=0, help="학습 시드 (모델 초기화 + 환경 스트림)")
    ap.add_argument("--no-mask", action="store_true",
                    help="ablation: action mask 없이 학습 (무효 배치를 정책이 고를 수 있음)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 시드 고정: 기존 런은 시드를 주지 않아 학습 재현이 보장되지 않았고 런 간 분산도 알 수 없었다.
    env = Monitor(PalletizeGym(seed=1000 + 100 * args.seed), filename=str(out / "monitor.csv"))
    if args.no_mask:
        # ablation: 마스크를 전부 True 로 → 정책이 무효 배치도 선택 가능 (환경이 거절)
        env.env.action_masks = lambda: np.ones(GRID * GRID * 2, dtype=bool)
    model = MaskablePPO(
        "MlpPolicy", env, device=args.device, verbose=0, seed=args.seed,
        n_steps=2048, batch_size=512, learning_rate=3e-4, ent_coef=0.01,
        policy_kwargs=dict(net_arch=[512, 256]),
    )
    model.learn(total_timesteps=args.steps,
                callback=CheckpointCallback(250_000, str(out / "ckpt"), "ppo"))
    model.save(str(out / "final"))

    res = evaluate(model)
    res.update({"seed": args.seed, "steps": args.steps, "action_mask": not args.no_mask})
    (out / "eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
