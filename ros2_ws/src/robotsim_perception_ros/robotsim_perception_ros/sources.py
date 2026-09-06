# -*- coding: utf-8 -*-
"""프레임 소스 — 노드가 어디서 ToF 프레임을 받을지.

  SyntheticSource : robotsim_perception.synthetic 으로 디팔레타이징 장면을 생성 (회사 데이터 불필요).
                    4x3 격자, 실측 SKU·깊이·피치. 프레임마다 박스가 하나씩 줄어드는 시퀀스.
  SessionSource   : 실측 .mim 세션 폴더들을 순서대로 재생 (로컬 전용, 공개 금지 데이터).
  실제 셀에서는 이 자리에 센서 SDK 드라이버(또는 depth 토픽 구독)가 들어간다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from robotsim_perception import load_frame
from robotsim_perception.synthetic import SynthBox, make_frame

SKU = (293.0, 219.0, 283.0)
PITCH = (300.0, 225.0)
TOP_DEPTH_MM = 2970.0            # 실측 2층 상면 깊이


class SyntheticSource:
    def __init__(self, seed: int = 0, noise_mm: float = 3.0, loop: bool = True, pick_every: int = 1):
        """pick_every: 몇 프레임마다 박스를 하나 뺄지 (1 = 매 프레임, 실제 셀의 1픽/사이클에 해당).
        rviz 데모·스크린샷에서는 8 정도로 두면 장면이 천천히 비워진다."""
        self.rng = np.random.default_rng(seed)
        self.noise_mm = noise_mm
        self.loop = loop
        self.pick_every = max(int(pick_every), 1)
        self._reset()

    def _reset(self):
        cells = [(c, r) for r in range(3) for c in range(4)]
        self.remaining = [cells[i] for i in self.rng.permutation(len(cells))]
        self.k = 0

    def next(self):
        if not self.remaining:
            if not self.loop:
                return None
            self._reset()
        boxes = []
        for (c, r) in self.remaining:
            x = (c - 1.5) * PITCH[0] + float(self.rng.normal(0, 3.0))
            y = (r - 1.0) * PITCH[1] + float(self.rng.normal(0, 3.0))
            boxes.append(SynthBox((x, y), depth_mm=TOP_DEPTH_MM, yaw_deg=float(self.rng.normal(0, 1.0))))
        frame = make_frame(boxes, noise_mm=self.noise_mm, seed=int(self.rng.integers(0, 1 << 30)))
        self.k += 1
        if self.k % self.pick_every == 0:
            self.remaining.pop()              # 픽 진행: 박스 하나 줄어든다
        return frame, f"synthetic#{self.k}"


class SessionSource:
    def __init__(self, root: str, loop: bool = True):
        root_p = Path(root)
        if (root_p / "4_tof_D.mim").exists():
            self.sessions = [root_p]
        else:
            self.sessions = sorted(d for d in root_p.iterdir() if d.is_dir() and (d / "4_tof_D.mim").exists())
        if not self.sessions:
            raise FileNotFoundError(f"no .mim sessions under {root}")
        self.loop = loop
        self.i = 0

    def next(self):
        if self.i >= len(self.sessions):
            if not self.loop:
                return None
            self.i = 0
        d = self.sessions[self.i]
        self.i += 1
        return load_frame(d), d.name


def make_source(spec: str, seed: int = 0, loop: bool = True, pick_every: int = 1):
    if spec in ("", "synthetic"):
        return SyntheticSource(seed=seed, loop=loop, pick_every=pick_every)
    return SessionSource(spec, loop=loop)
