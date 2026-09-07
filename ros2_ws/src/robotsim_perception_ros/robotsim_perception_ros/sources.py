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
        self._last_cells_xy = [((c - 1.5) * PITCH[0], (r - 1.0) * PITCH[1]) for (c, r) in self.remaining]
        self.k += 1
        if self.auto_pop and self.k % self.pick_every == 0:
            self.remaining.pop()              # 픽 진행: 박스 하나 줄어든다 (어느 박스인지는 임의)
        return frame, f"synthetic#{self.k}"

    auto_pop = True                           # False 면 remove_nearest() 로만 박스가 줄어든다 (실제 셀처럼 '집은 박스'가 사라짐)

    def remove_nearest(self, x_mm: float, y_mm: float, max_dist_mm: float = 200.0) -> bool:
        """카메라 좌표 (x, y) 에 가장 가까운 남은 셀을 제거. 로봇이 집은 박스가 다음 프레임에서 사라지는 것을 흉내 낸다."""
        if not self.remaining:
            return False
        d = [np.hypot((c - 1.5) * PITCH[0] - x_mm, (r - 1.0) * PITCH[1] - y_mm) for (c, r) in self.remaining]
        i = int(np.argmin(d))
        if d[i] > max_dist_mm:
            return False
        self.remaining.pop(i)
        return True


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


class TwinSource:
    """MuJoCo 셀 트윈(tools/twin_server.py, Windows 프로세스)을 카메라로 쓴다. spec = 'twin://host:port' (host 생략 시 WSL2 기본
    게이트웨이 = Windows 호스트). 프레임마다 트윈을 렌더해 받으므로, 로봇(트윈 안의 UR10e)이 박스를 옮기면 다음 프레임에 그대로 반영된다.
    소스 팔레트가 비면(정답 기준 — 실제 셀의 '팔레트 비움' PLC 신호에 해당) None 을 돌려 노드가 '소스 소진'을 알린다."""

    def __init__(self, spec: str, loop: bool = False, timeout_s: float = 600.0):
        from robotsim_perception.twin_link import TwinClient
        self.client = TwinClient.from_spec(spec, timeout_s=timeout_s)   # 실행 중에는 응답이 늦어진다(한 픽 시뮬 수십 초)
        self.loop = loop
        self.info = None          # 연결은 첫 next() 에서 (서버가 늦게 떠도 노드가 죽지 않게 — 오류는 노드가 SOURCE_ERROR 로 처리)
        self.k = 0

    def next(self):
        """프레임 하나. 서버 연결/응답 오류는 OSError/RuntimeError 로 올라가고, 노드가 SOURCE_ERROR 상태로 바꿔 보고한다."""
        from robotsim_perception.frame import Frame
        if self.info is None:
            self.info = self.client.hello()
        meta, f = self.client.frame()
        if int(meta.get("remaining", 1)) == 0 and not self.loop:
            return None
        self.k += 1
        name = f"{meta.get('name', 'twin')} (remaining {meta.get('remaining', '?')}, placed {meta.get('placed', '?')})"
        return Frame(f["X"], f["Y"], f["D"], f["I"], source=name), name


def make_source(spec: str, seed: int = 0, loop: bool = True, pick_every: int = 1):
    if spec in ("", "synthetic"):
        return SyntheticSource(seed=seed, loop=loop, pick_every=pick_every)
    if spec.startswith("twin"):
        return TwinSource(spec, loop=loop)
    return SessionSource(spec, loop=loop)


# ------------------------------------------------------------------ 대차(카트) 소스

class SyntheticCartSource:
    """합성 대차 장면. 프레임마다 카메라 높이·대차 yaw 를 바꿔 '카메라 좌표는 변해도 대차 프레임 좌표는 같다'를 보여 준다.
    실측 4세션의 카메라 높이 400~463 mm 범위를 따른다."""
    HEIGHTS = (400.0, 420.0, 440.0, 462.0)
    YAWS = (0.0, 4.0, -3.0, 2.0)

    def __init__(self, seed: int = 0, loop: bool = True, noise_mm: float = 2.5):
        from robotsim_perception.synthetic_cart import make_cart_frame
        self._make = make_cart_frame
        self.seed, self.loop, self.noise_mm = int(seed), loop, float(noise_mm)
        self.k = 0

    def next(self):
        n = len(self.HEIGHTS)
        if self.k >= n and not self.loop:
            return None
        i = self.k % n
        frame, _gt = self._make(seed=self.seed + self.k, cam_height_mm=self.HEIGHTS[i],
                                cart_yaw_deg=self.YAWS[i], noise_mm=self.noise_mm)
        self.k += 1
        return frame, f"synthetic_cart#{self.k} (h={self.HEIGHTS[i]:.0f} yaw={self.YAWS[i]:+.0f})"


class DockingCartSource:
    """도킹 시뮬용 합성 대차: AGV 시뮬(agv_sim_node)이 주는 상대 포즈(hook_u, rim_v, yaw)로 프레임마다 장면을 렌더한다.
    set_pose() 로 최신 포즈를 넣어 두면 next() 가 그 포즈의 프레임을 만든다 (프레임마다 다른 노이즈 시드)."""

    def __init__(self, seed: int = 0, noise_mm: float = 2.5, init=(-80.0, -450.0, 5.0)):
        from robotsim_perception.dock import RelPose
        self.rp = RelPose(*init)
        self.seed, self.noise_mm, self.k = int(seed), float(noise_mm), 0

    def set_pose(self, hook_u_mm: float, rim_v_mm: float, yaw_deg: float):
        from robotsim_perception.dock import RelPose
        self.rp = RelPose(float(hook_u_mm), float(rim_v_mm), float(yaw_deg))

    def next(self):
        from robotsim_perception.synthetic_cart import make_cart_frame
        self.k += 1
        frame, _gt = make_cart_frame(noise_mm=self.noise_mm, seed=self.seed + self.k, **self.rp.render_kwargs())
        return frame, f"dock#{self.k} (hook_u={self.rp.hook_u_mm:.0f} rim_v={self.rp.rim_v_mm:.0f} yaw={self.rp.yaw_deg:+.1f})"


def make_cart_source(spec: str, seed: int = 0, loop: bool = True):
    if spec in ("", "synthetic"):
        return SyntheticCartSource(seed=seed, loop=loop)
    if spec == "synthetic_dock":
        return DockingCartSource(seed=seed)
    return SessionSource(spec, loop=loop)
