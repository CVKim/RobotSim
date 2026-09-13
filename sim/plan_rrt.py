# -*- coding: utf-8 -*-
"""관절 공간 경로 계획 (RRT-Connect + 단축 다듬기) — 장애물을 **돌아가는** 이동.

지금까지 팔 실행기는 TCP 직선 이동만 했다. 직선 위에 설비가 있으면 그 픽을 포기했고, 문서에도
'장애물을 돌아가는 경로 계획은 없다' 고 적어 두었다. 여기서 그 구멍을 메운다.

  plan(arm, q_start, q_goal)          -> 관절 경유점 목록 또는 None
  shortcut(arm, path)                 -> 무작위 구간을 직선으로 당겨 길이를 줄인다
  path_cost(path)                     -> 관절 이동량 합 (rad). 계획 품질 비교용

충돌 판정은 시뮬레이터가 한다(sim/arm.py 의 contacts / path_collides). 즉 이 모듈은 '어떤 자세가 안전한가' 를
직접 정의하지 않고, 트윈이 이미 쓰는 판정을 그대로 불러 쓴다 — 실행기와 계획기가 같은 기준을 보게 하려는 것이다.

RRT-Connect: 시작·목표에서 트리를 하나씩 키우고 서로를 향해 뻗어(connect) 만나면 끝낸다. 단방향 RRT 보다
좁은 통로에서 빨리 붙는다. 무작위라서 같은 문제도 매번 다른 경로가 나오므로, 쓰는 쪽은 시드를 고정해 재현한다.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence

import numpy as m_np  # noqa: N813  (numpy 를 np 로 쓰되 이름 충돌 방지)
np = m_np


def path_cost(path: Sequence[Sequence[float]]) -> float:
    """관절 이동량 합 (rad). 짧을수록 좋은 경로."""
    P = [np.asarray(q, float) for q in path]
    return float(sum(float(np.abs(b - a).sum()) for a, b in zip(P[:-1], P[1:])))


class _Tree:
    def __init__(self, root):
        self.nodes = [np.asarray(root, float)]
        self.parent = [-1]

    def nearest(self, q) -> int:
        d = [float(np.linalg.norm(n - q)) for n in self.nodes]
        return int(np.argmin(d))

    def add(self, parent: int, q) -> int:
        self.nodes.append(np.asarray(q, float))
        self.parent.append(parent)
        return len(self.nodes) - 1

    def path_to(self, idx: int) -> List[np.ndarray]:
        out = []
        while idx >= 0:
            out.append(self.nodes[idx])
            idx = self.parent[idx]
        return out[::-1]


def _steer(a, b, step: float):
    d = b - a
    n = float(np.linalg.norm(d))
    return b.copy() if n <= step else a + d * (step / n)


def plan(arm, q_start, q_goal, *, step: float = 0.25, max_iters: int = 3000, seed: int = 0,
         allowed_gids: Sequence[int] = (), collides: Optional[Callable] = None,
         goal_bias: float = 0.1) -> Optional[List[np.ndarray]]:
    """q_start -> q_goal 의 충돌 없는 관절 경로. 못 찾으면 None.

    collides(q) 를 주면 그것으로 자세 검사를 한다(기본: arm.contacts 가 비어 있어야 통과).
    구간 검사는 arm.path_collides 로 한다 — 실행기가 쓰는 것과 같은 판정이다."""
    q0 = np.asarray(q_start, float)
    q1 = np.asarray(q_goal, float)
    lo, hi = np.asarray(arm.lo, float), np.asarray(arm.hi, float)
    if collides is None:
        def collides(q):
            return bool(arm.contacts(q, allowed_gids=allowed_gids))
    if collides(q0) or collides(q1):
        return None
    if not arm.path_collides(q0, q1, allowed_gids=allowed_gids):
        return [q0, q1]                                    # 직선이 되면 그게 최선이다

    rng = np.random.default_rng(seed)
    ta, tb = _Tree(q0), _Tree(q1)
    for k in range(max_iters):
        q_rand = (q1 if rng.random() < goal_bias and k % 2 == 0 else rng.uniform(lo, hi))
        i = ta.nearest(q_rand)
        q_new = _steer(ta.nodes[i], q_rand, step)
        q_new = np.clip(q_new, lo, hi)
        if collides(q_new) or arm.path_collides(ta.nodes[i], q_new, allowed_gids=allowed_gids):
            ta, tb = tb, ta
            continue
        ia = ta.add(i, q_new)
        # 반대 트리를 q_new 쪽으로 끝까지 뻗는다 (connect)
        j = tb.nearest(q_new)
        q_cur = tb.nodes[j]
        jb = j
        while True:
            q_next = _steer(q_cur, q_new, step)
            q_next = np.clip(q_next, lo, hi)
            if collides(q_next) or arm.path_collides(q_cur, q_next, allowed_gids=allowed_gids):
                break
            jb = tb.add(jb, q_next)
            q_cur = q_next
            if float(np.linalg.norm(q_cur - q_new)) < 1e-6:
                pa, pb = ta.path_to(ia), tb.path_to(jb)
                path = pa + pb[::-1][1:]
                # ta 가 시작 트리인지 확인해 방향을 맞춘다
                if float(np.linalg.norm(path[0] - q0)) > 1e-6:
                    path = path[::-1]
                return path
        ta, tb = tb, ta
    return None


def shortcut(arm, path, *, iters: int = 120, seed: int = 0, allowed_gids: Sequence[int] = ()) -> List[np.ndarray]:
    """무작위 두 지점을 직선으로 이어 보며 경로를 줄인다. RRT 경로는 지그재그가 심해 이 단계가 꼭 필요하다."""
    P = [np.asarray(q, float) for q in path]
    if len(P) < 3:
        return P
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        if len(P) < 3:
            break
        i = int(rng.integers(0, len(P) - 2))
        j = int(rng.integers(i + 2, len(P)))
        if not arm.path_collides(P[i], P[j], allowed_gids=allowed_gids):
            P = P[:i + 1] + P[j:]
    return P


def resample(path, max_step: float = 0.08) -> List[np.ndarray]:
    """구간이 길면 잘게 나눈다 (위치 서보로 따라가기 좋게)."""
    P = [np.asarray(q, float) for q in path]
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(math.ceil(float(np.abs(b - a).max()) / max_step)))
        for t in range(1, n + 1):
            out.append(a + (b - a) * (t / n))
    return out
