# -*- coding: utf-8 -*-
"""대차 도킹 시뮬 (합성 대차 장면, 회사 데이터 불필요).

1. RelPose -> 합성 장면 정답이 RelPose.hook_plane() 과 같다 (좌표 규약이 맞다).
2. 운동학 step 의 부호: 전진하면 림이 다가오고, 좌회전하면 대차 요가 줄어드는 방향(yaw −ω dt).
3. 정답 측정(oracle)으로 제어기가 여러 시작 자세에서 수렴한다.
4. 인식(analyze_cart)을 넣은 폐루프도 수렴한다 (노이즈 2.5 mm, 시작 3개)."""
import math

import numpy as np
import pytest

from robotsim_perception.cart import analyze_cart
from robotsim_perception.dock import DockController, RelPose, measurement_from_result, simulate, step
from robotsim_perception.synthetic_cart import make_cart_frame


def test_relpose_matches_synthetic_ground_truth():
    for rp in (RelPose(-80.0, -285.0, 0.0), RelPose(60.0, -450.0, 6.0), RelPose(-120.0, -380.0, -7.5)):
        _, gt = make_cart_frame(noise_mm=0.0, **rp.render_kwargs())
        u, v = rp.hook_plane()
        assert abs(gt["hook_plane_mm"][0] - u) < 1e-6 and abs(gt["hook_plane_mm"][1] - v) < 1e-6, (rp, gt["hook_plane_mm"])
        assert abs(gt["hook_cart_mm"][0] - 36.0) < 1e-6         # 고리는 왼쪽 레일에서 36 mm (실측 규약)


def test_kinematics_signs():
    rp = RelPose(0.0, -500.0, 0.0)
    fwd = step(rp, 100.0, 0.0, 1.0)
    assert abs(fwd.rim_v_mm - (-400.0)) < 1e-6 and abs(fwd.hook_u_mm) < 1e-9        # 전진 100 mm -> 림이 100 mm 가까워진다
    turned = step(rp, 0.0, 0.1, 1.0)
    assert abs(turned.yaw_deg - (-math.degrees(0.1))) < 1e-6                          # 좌회전 -> 대차가 시계방향으로 보인다
    assert abs(turned.rim_v_mm - rp.rim_v_mm) < 1e-9
    # 대차가 6도 돌아가 있을 때 전진하면 고리의 u 도 sin(6°) 비율로 변한다
    rp2 = RelPose(0.0, -500.0, 6.0)
    m = step(rp2, 100.0, 0.0, 1.0)
    assert abs(m.hook_u_mm - 100 * math.sin(math.radians(6.0))) < 1e-6


def _oracle(rp):
    u, v = rp.hook_plane()
    return u, v, rp.yaw_deg


@pytest.mark.parametrize("start", [RelPose(-80.0, -285.0, 0.0), RelPose(150.0, -600.0, 8.0), RelPose(-140.0, -520.0, -8.0),
                                   RelPose(40.0, -350.0, 5.0), RelPose(0.0, -150.0, 0.0)])
def test_controller_converges_with_oracle(start):
    ctrl = DockController()
    r = simulate(start, ctrl, _oracle, dt_s=0.5, max_steps=80)
    assert r["success"] and r["truth_docked"], r
    e = r["errors"]
    assert abs(e["e_u"]) < ctrl.tol_u_mm and abs(e["e_v"]) < ctrl.tol_v_mm and abs(e["e_yaw"]) < ctrl.tol_yaw_deg


def _perception(seed_box):
    def measure(rp):
        seed_box[0] += 1
        frame, _ = make_cart_frame(noise_mm=2.5, seed=seed_box[0], **rp.render_kwargs())
        return measurement_from_result(analyze_cart(frame))
    return measure


@pytest.mark.parametrize("start", [RelPose(-80.0, -285.0, 0.0), RelPose(120.0, -480.0, 6.0), RelPose(-100.0, -450.0, -6.0)])
def test_closed_loop_with_perception_converges(start):
    ctrl = DockController()
    r = simulate(start, ctrl, _perception([0]), dt_s=0.5, max_steps=80)
    assert r["success"], r
    e = r["errors"]                                     # 정답 기준 최종 오차: 인식 오차만큼 허용치를 넘을 수 있어 여유를 둔다
    assert abs(e["e_u"]) < 25 and abs(e["e_v"]) < 25 and abs(e["e_yaw"]) < 4.0, e
    assert r["misses"] <= 2
