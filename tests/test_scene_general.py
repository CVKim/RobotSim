# -*- coding: utf-8 -*-
"""트윈 씬 일반화 — 박스별 치수(혼합 SKU)와 기운 카메라가 정답 좌표와 맞는지. mujoco 필요, 회사 데이터 불필요."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT))

from cell_scene import CAM_H, build_xml, full_layout  # noqa: E402
from cell_twin import camera_transform, ground_truth, settle  # noqa: E402
from robotsim_perception.pose import topdown_camera_transform  # noqa: E402

MIXED = [(0.293, 0.219, 0.283), (0.245, 0.185, 0.240), (0.200, 0.150, 0.200), (0.270, 0.205, 0.160)]


def build(sku=None, tilt=0.0, layers=1, seed=3):
    xml, _ = build_xml(full_layout(layers), seed=seed, sku=sku, cam_tilt_deg=tilt)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    settle(m, d, 1200)
    return m, d


def test_topdown_camera_transform_unchanged():
    """카메라를 안 기울이면 모델에서 만든 변환이 기존 탑다운 값과 같아야 한다 (회귀)."""
    m, d = build()
    assert np.allclose(camera_transform(m, d), topdown_camera_transform(CAM_H * 1000.0), atol=1e-6)


def test_ground_truth_two_paths_agree():
    """정답 좌표는 기존 탑다운 공식과 새 변환 경로가 같은 값을 줘야 한다."""
    m, d = build()
    a = ground_truth(m, d, top_only=False)
    b = ground_truth(m, d, top_only=False, T_base_cam=camera_transform(m, d))
    assert len(a) == len(b) and a
    for x, y in zip(a, b):
        assert abs(x["center_mm"][0] - y["center_mm"][0]) < 1e-3
        assert abs(x["center_mm"][1] - y["center_mm"][1]) < 1e-3
        assert abs(x["top_d_mm"] - y["top_d_mm"]) < 1e-3


def test_mixed_sku_dims_come_from_the_model():
    """박스마다 다른 치수를 넣으면 정답 치수도 그대로 나와야 한다 (예전에는 BOX 상수를 썼다)."""
    layout = full_layout(1)
    sku = [MIXED[i % len(MIXED)] for i in range(len(layout))]
    xml, _ = build_xml(layout, seed=5, sku=sku)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    settle(m, d, 1200)
    got = sorted(round(b["dims_mm"][0]) for b in ground_truth(m, d, top_only=False))
    want = sorted(round(max(s[0], s[1]) * 1000) for s in sku)
    assert got == want, (got, want)


def test_tilted_camera_sees_the_pallet():
    """카메라를 기울여도 팔레트는 화면 안에 있고, 정답 깊이는 기울기만큼 퍼진다."""
    m0, d0 = build()
    m1, d1 = build(tilt=20.0)
    g0 = ground_truth(m0, d0, top_only=False, T_base_cam=camera_transform(m0, d0))
    g1 = ground_truth(m1, d1, top_only=False, T_base_cam=camera_transform(m1, d1))
    assert len(g0) == len(g1)
    spread0 = max(b["top_d_mm"] for b in g0) - min(b["top_d_mm"] for b in g0)
    spread1 = max(b["top_d_mm"] for b in g1) - min(b["top_d_mm"] for b in g1)
    assert spread0 < 30.0, spread0             # 탑다운: 같은 층은 같은 깊이
    # 20도면 박스 중심들이 앞뒤로 155 mm 퍼진다(중심 간 거리 0.45 m × sin20). 검출기의 층 허용폭이 ±40 mm 이므로
    # 한 번에 다 볼 수 없다 — '층 = 같은 깊이' 가 깨지는 지점이다.
    assert spread1 > 100.0, spread1
