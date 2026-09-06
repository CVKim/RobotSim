# -*- coding: utf-8 -*-
"""대차 고리 검출 + 대차 프레임 — 합성 장면 절대 정답 검사 (회사 데이터 불필요)."""
import numpy as np
import pytest

from robotsim_perception.cart import analyze_cart, cam_to_plane
from robotsim_perception.synthetic_cart import make_cart_frame, HOOK_H_MM


@pytest.fixture(scope="module")
def scene():
    frame, gt = make_cart_frame(seed=1)
    res, masks = analyze_cart(frame, return_masks=True)
    return frame, gt, res, masks


def test_scene_is_plausible(scene):
    frame, gt, res, masks = scene
    assert frame.valid.mean() > 0.6
    assert masks["plate"].sum() > 30000                        # 플레이트가 화면을 지배
    assert masks["wall"].sum() > 200


def test_plane_matches_gt(scene):
    frame, gt, res, masks = scene
    assert res.status == "OK", res.reason
    n_est = np.asarray(res.plane["normal_toward_camera"])
    assert float(n_est @ np.asarray(gt["normal_cam"])) > 0.9999          # 법선 < 0.8°
    assert abs(res.plane["camera_height_mm"] - gt["cam_height_mm"]) < 3.0
    assert res.plane["plate_mad_mm"] < 4.0


def test_hook_position_absolute(scene):
    frame, gt, res, masks = scene
    err = np.asarray(res.hook_cam_mm) - np.asarray(gt["hook_cam_mm"])
    assert np.linalg.norm(err) < 6.0, (res.hook_cam_mm, gt["hook_cam_mm"])
    u, v, h = res.hook_plane_mm
    assert abs(u - gt["hook_plane_mm"][0]) < 4.0
    assert abs(v - gt["hook_plane_mm"][1]) < 4.0
    assert abs(h - HOOK_H_MM) < 6.0


def test_cart_frame_absolute(scene):
    frame, gt, res, masks = scene
    assert abs(res.cart["rim_yaw_deg"]) < 1.0
    assert abs(res.cart["rail_gap_mm"] - gt["rail_gap_mm"]) < 6.0
    u, v, h = res.hook_cart_mm
    assert abs(u - gt["hook_cart_mm"][0]) < 5.0
    assert abs(v - gt["hook_cart_mm"][1]) < 4.0
    # R,t 로 카메라 좌표를 옮기면 hook_cart 와 같아야 한다 (일관성)
    R, t = np.asarray(res.R_cart), np.asarray(res.t_cart)
    p = R @ np.asarray(res.hook_cam_mm) + t
    assert np.allclose(p, res.hook_cart_mm, atol=0.05)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)


def test_cart_yaw_recovered():
    frame, gt = make_cart_frame(seed=2, cart_yaw_deg=6.0)
    res = analyze_cart(frame)
    assert res.status == "OK", res.reason
    assert abs(abs(res.cart["rim_yaw_deg"]) - 6.0) < 1.0
    u, v, h = res.hook_cart_mm                                  # 대차 프레임 좌표는 yaw 와 무관해야 한다
    assert abs(u - gt["hook_cart_mm"][0]) < 5.0 and abs(v) < 4.0


@pytest.mark.parametrize("yaw", [-8.0, 8.0])
@pytest.mark.parametrize("normal", [(0.03, 0.64, -0.77), (0.05, 0.70, -0.71)])
def test_cart_yaw_sweep_over_camera_height(yaw, normal):
    """회귀: |yaw| > 5.7° 에서 림 창(+30 mm)에 잘린 먼 빈이 가짜 수평선을 만들어 yaw≈0·hook v' 25~31 mm 오차를
    상태 OK 로 내던 결함(검토에서 발견; 카메라 높이 418~422 mm 에서만 터지는 쌍안정). 부호까지 검사한다."""
    for hc in (380.0, 400.0, 418.0, 420.0, 422.0, 440.0, 480.0):
        frame, gt = make_cart_frame(seed=0, cam_height_mm=hc, cart_yaw_deg=yaw, normal_cam=normal, noise_mm=0.0)
        res = analyze_cart(frame)
        assert res.status == "OK", (hc, yaw, normal, res.reason)
        assert abs(res.cart["rim_yaw_deg"] - yaw) < 1.0, (hc, yaw, normal, res.cart["rim_yaw_deg"])
        err = np.asarray(res.hook_cart_mm) - np.asarray(gt["hook_cart_mm"])
        assert np.linalg.norm(err[:2]) < 5.0, (hc, yaw, normal, res.hook_cart_mm, gt["hook_cart_mm"])


def test_degraded_frames_return_status_not_exception():
    """회귀: 창 안 강도가 균일하거나 벽 아래가 잘리면 Otsu 전경이 없어 배경(라벨 0)을 벽으로 잡고
    빈 배열 percentile 로 IndexError 가 났다(검토에서 발견). 상태로 돌아와야 한다."""
    f, _ = make_cart_frame(seed=1)
    f.I[:] = 8000.0
    r = analyze_cart(f)
    assert r.status in ("NO_HOOK", "NO_CART_FRAME"), r.status
    for cut in (238, 260, 283):
        f2, gt = make_cart_frame(seed=1)
        f2.valid[cut:, :] = False
        r2 = analyze_cart(f2)
        assert r2.status in ("OK", "NO_HOOK", "NO_CART_FRAME"), (cut, r2.status)
        if r2.status == "OK":
            assert np.all(np.isfinite(r2.hook_cart_mm)) and np.all(np.isfinite(r2.t_cart))


def test_repeatability_across_camera_height():
    """카메라 높이·시점이 달라도 대차 프레임 좌표는 같아야 한다 (실측 4세션 구조 프레임 3.85 mm RMS 의 합성판)."""
    got = []
    for k, hc in enumerate((400.0, 420.0, 440.0, 462.0)):
        frame, gt = make_cart_frame(seed=10 + k, cam_height_mm=hc)
        res = analyze_cart(frame)
        assert res.status == "OK", res.reason
        got.append(res.hook_cart_mm)
    got = np.asarray(got)
    rms = float(np.sqrt(np.mean(np.sum((got - got.mean(axis=0)) ** 2, axis=1))))
    assert rms < 4.0, got


def test_statuses_without_hook_or_plane():
    frame, gt = make_cart_frame(seed=3)
    # 고리 없음: 고리 영역 픽셀을 무효화 -> NO_HOOK (평면은 유지)
    res, masks = analyze_cart(frame, return_masks=True)
    wall = masks["wallcrown"]
    ys, xs = np.where(wall)
    y0, y1, x0, x1 = ys.min() - 8, ys.max() + 8, xs.min() - 8, xs.max() + 8
    f2 = make_cart_frame(seed=3)[0]
    f2.valid[y0:y1, x0:x1] = False
    r2 = analyze_cart(f2)
    assert r2.status == "NO_HOOK" and r2.plane is not None
    # 거의 다 무효 -> NO_PLANE
    f3 = make_cart_frame(seed=3)[0]
    f3.valid[:] = False
    assert analyze_cart(f3).status == "NO_PLANE"


def test_log_line_json(scene):
    import json
    frame, gt, res, masks = scene
    d = json.loads(res.log_line())
    assert d["status"] == "OK" and "R_cart" not in d and len(d["hook_cart_mm"]) == 3
    assert cam_to_plane(gt["frame"], gt["hook_cam_mm"]) == pytest.approx(gt["hook_plane_mm"], abs=1e-6)
