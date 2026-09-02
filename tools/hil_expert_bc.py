# -*- coding: utf-8 -*-
"""모방학습 풀사이클: 스크립티드 전문가 데모 수집 → BC 학습 → 성공률 평가.

배경: dense SAC(500k)은 접근까지만 학습하고 파지 완성 실패(성공률 0) — 탐사 한계.
사람 개입 대신 화이트박스 스크립티드 전문가(접근→하강→파지→리프트 상태기계)로
데모를 수집해 BC로 증류한다. 데모 수(10/30/60) ablation 포함.
실행: python tools/hil_expert_bc.py
"""
import json
from pathlib import Path

import gym_hil  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

OUT = Path(r"E:\Robot_Sim\runs\hil_bc")
OUT.mkdir(parents=True, exist_ok=True)
ENV_ID = "gym_hil/PandaPickCubeBase-v0"


def make_env():
    return gym.make(ENV_ID, image_obs=False, reward_type="dense")


def flat_obs(obs):
    return np.concatenate([obs["agent_pos"], obs["environment_state"]]).astype(np.float32)


class ScriptedExpert:
    """EE 델타 상태기계: above → descend → grasp → lift."""

    def __init__(self, env):
        self.d = env.unwrapped._data
        self.phase = 0

    def reset(self):
        self.phase = 0

    def act(self):
        # 주의: mocap_pos는 실제 EE가 아니라 '명령 타깃' — 작은 델타로 서서히 이동
        # 그리퍼 ctrl: 0=열림, 255=닫힘 → 델타 음수=열기, 양수=닫기
        ee = self.d.mocap_pos[0]
        block = self.d.sensor("block_pos").data
        a = np.zeros(7, dtype=np.float32)
        step = 0.02
        if self.phase == 0:      # 블록 상공으로 (열고 접근)
            tgt = block + [0, 0, 0.10]
            a[:3] = np.clip(tgt - ee, -step, step)
            a[6] = -0.5
            if np.linalg.norm(tgt - ee) < 0.015:
                self.phase = 1
        elif self.phase == 1:    # 하강
            tgt = block + [0, 0, 0.012]
            a[:3] = np.clip(tgt - ee, -step, step)
            a[6] = -0.5
            if abs(ee[2] - tgt[2]) < 0.006:
                self.phase = 2
                self.t = 0
        elif self.phase == 2:    # 파지 (닫기)
            a[6] = 0.5
            self.t += 1
            if self.t > 10:
                self.phase = 3
        else:                    # 리프트 (닫힘 유지)
            a[2] = step
            a[6] = 0.5
        return a


def collect_demos(n_target, max_tries=200, max_steps=250):
    env = make_env()
    expert = ScriptedExpert(env)
    demos, tries, succ = [], 0, 0
    while len(demos) < n_target and tries < max_tries:
        tries += 1
        obs, _ = env.reset()
        expert.reset()
        traj = []
        for _ in range(max_steps):
            a = expert.act()
            traj.append((flat_obs(obs), a.copy()))
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        if info.get("succeed", False):
            demos.append(traj)
            succ += 1
    env.close()
    return demos, succ / max(tries, 1)


class BCPolicy(nn.Module):
    def __init__(self, obs_dim=21, act_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim), nn.Tanh())

    def forward(self, x):
        return self.net(x)


def train_bc(demos, device, epochs=200):
    X = torch.tensor(np.array([o for tr in demos for o, _ in tr]), device=device)
    Y = torch.tensor(np.array([a for tr in demos for _, a in tr]), device=device)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    pol = BCPolicy(X.shape[1]).to(device)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    for ep in range(epochs):
        idx = torch.randperm(len(X), device=device)
        for i in range(0, len(X), 512):
            b = idx[i:i + 512]
            loss = nn.functional.mse_loss(pol((X[b] - mu) / sd), Y[b])
            opt.zero_grad(), loss.backward(), opt.step()
    return pol, mu.cpu().numpy(), sd.cpu().numpy(), float(loss.item())


def eval_policy(pol, mu, sd, device, n=50, max_steps=250):
    env = make_env()
    succ = 0
    for _ in range(n):
        obs, _ = env.reset()
        for _ in range(max_steps):
            x = (flat_obs(obs) - mu[0]) / sd[0]
            with torch.no_grad():
                a = pol(torch.tensor(x, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        succ += bool(info.get("succeed", False))
    env.close()
    return succ / n


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    demos, expert_sr = collect_demos(60)
    print(f"expert: {len(demos)} demos collected, success_rate={expert_sr:.2f}")

    results = {"expert_success_rate": round(expert_sr, 3), "bc": {}}
    for n in [10, 30, 60]:
        pol, mu, sd, loss = train_bc(demos[:n], device)
        sr = eval_policy(pol, mu, sd, device)
        results["bc"][str(n)] = {"success_rate": round(sr, 3), "final_mse": round(loss, 5)}
        print(f"BC({n} demos): success_rate={sr:.2f}")
        torch.save(pol.state_dict(), OUT / f"bc_{n}.pt")

    (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(results)


if __name__ == "__main__":
    main()
