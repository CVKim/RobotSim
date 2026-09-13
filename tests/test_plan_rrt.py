# -*- coding: utf-8 -*-
"""관절 공간 경로 계획 (sim/plan_rrt.py) — 가짜 팔로 논리를 검사한다. mujoco·회사 데이터 불필요.

충돌 판정은 진짜 트윈이 하지만, 여기서는 '가운데 벽이 있는 2차원 관절 공간' 같은 간단한 세계를 만들어
계획기 자체의 성질을 본다: 길을 찾는가, 막히면 None 인가, 단축이 충돌을 만들지 않는가."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))

import plan_rrt  # noqa: E402


class FakeArm:
    """관절 2개짜리 세계. blocked(q) 가 True 면 충돌. path_collides 는 구간을 잘게 나눠 검사한다."""

    def __init__(self, blocked, lo=(-1.0, -1.0), hi=(1.0, 1.0)):
        self.blocked = blocked
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)

    def contacts(self, q, allowed_gids=()):
        return ["hit"] if self.blocked(np.asarray(q, float)) else []

    def path_collides(self, a, b, n=24, allowed_gids=(), max_step_rad=0.02):
        a, b = np.asarray(a, float), np.asarray(b, float)
        steps = max(n, int(np.ceil(float(np.max(np.abs(b - a))) / max_step_rad)))
        for t in np.linspace(0.0, 1.0, steps + 1):
            if self.blocked(a * (1 - t) + b * t):
                return ["hit"]
        return None

    def segment_time(self, a, b):
        return float(np.max(np.abs(np.asarray(b, float) - np.asarray(a, float))))


def wall_with_gap(q):
    """x=0 부근이 벽인데 y > 0.6 에 통로가 있다."""
    return abs(q[0]) < 0.12 and q[1] < 0.6


def solid_wall(q):
    return abs(q[0]) < 0.12


def test_straight_line_when_free():
    arm = FakeArm(lambda q: False)
    p = plan_rrt.plan(arm, [-0.8, 0.0], [0.8, 0.0], seed=0)
    assert p is not None and len(p) == 2          # 막힌 게 없으면 두 점짜리 직선


def test_finds_way_around_the_wall():
    arm = FakeArm(wall_with_gap)
    start, goal = [-0.8, 0.0], [0.8, 0.0]
    assert arm.path_collides(start, goal) is not None     # 직선은 막힌다
    p = plan_rrt.plan(arm, start, goal, seed=1, step=0.15, max_iters=4000)
    assert p is not None, "통로가 있는데 못 찾았다"
    assert all(not arm.blocked(q) for q in p)
    assert all(arm.path_collides(a, b) is None for a, b in zip(p[:-1], p[1:]))
    assert max(q[1] for q in p) > 0.6                     # 통로를 통해 돌아갔다


def test_returns_none_when_sealed():
    arm = FakeArm(solid_wall)
    assert plan_rrt.plan(arm, [-0.8, 0.0], [0.8, 0.0], seed=2, step=0.15, max_iters=600) is None


def test_rejects_colliding_endpoints():
    arm = FakeArm(solid_wall)
    assert plan_rrt.plan(arm, [0.0, 0.0], [0.8, 0.0], seed=0) is None      # 시작이 충돌
    assert plan_rrt.plan(arm, [-0.8, 0.0], [0.05, 0.0], seed=0) is None    # 목표가 충돌


def test_shortcut_shortens_without_collision():
    arm = FakeArm(wall_with_gap)
    p = plan_rrt.plan(arm, [-0.8, 0.0], [0.8, 0.0], seed=3, step=0.15, max_iters=4000)
    assert p is not None
    s = plan_rrt.shortcut(arm, p, seed=3)
    assert plan_rrt.path_cost(s) <= plan_rrt.path_cost(p) + 1e-9
    assert all(arm.path_collides(a, b) is None for a, b in zip(s[:-1], s[1:]))


def test_resample_keeps_endpoints_and_limits_step():
    p = [np.array([0.0, 0.0]), np.array([1.0, 0.0])]
    r = plan_rrt.resample(p, max_step=0.1)
    assert np.allclose(r[0], p[0]) and np.allclose(r[-1], p[-1])
    assert max(float(np.abs(b - a).max()) for a, b in zip(r[:-1], r[1:])) <= 0.1 + 1e-9
