# -*- coding: utf-8 -*-
"""합성 프레임 단위 테스트: 치수·기울기·센티넬·빈 프레임·노이즈."""
import math

import numpy as np
import pytest

from robotsim_perception import (Frame, SENTINEL_D, SENTINEL_XY, detect_boxes, detect_top_layer,
                                 valid_mask)
from robotsim_perception.synthetic import SynthBox, make_frame

SKU = (293.0, 219.0, 283.0)
DEPTH = 1500.0  # 박스 상면 ~98x73 px (fx=500). 실측 거리(2970mm, ~49x37px)는 test_real_depth_grid 에서.


def two_boxes(**kw):
    return [SynthBox((-250.0, 0.0), depth_mm=DEPTH, **kw),
            SynthBox((250.0, 100.0), depth_mm=DEPTH, yaw_deg=90.0, **kw)]


def rel_err(dims):
    return max(abs(dims[0] - SKU[0]) / SKU[0], abs(dims[1] - SKU[1]) / SKU[1])


def test_two_flat_boxes_dims_within_5pct():
    fr = make_frame(two_boxes())
    top = detect_top_layer(fr)
    assert top.ok and abs(top.depth_mm - DEPTH) < 10
    boxes = detect_boxes(fr, sku=SKU)
    assert len(boxes) == 2
    for b in boxes:
        assert rel_err(b.dims_mm) < 0.05, b.dims_mm
        assert abs(b.center_mm[2] - DEPTH) < 5
        assert b.tilt_deg < 0.5
        assert b.plane_rms_mm < 1.0
        assert 0.7 < b.confidence <= 1.0
        assert b.normal[2] < 0  # 카메라 쪽을 향함
    by_x = sorted(boxes, key=lambda b: b.center_mm[0])
    assert abs(by_x[0].center_mm[0] + 250) < 10 and abs(by_x[0].center_mm[1]) < 10
    assert abs(by_x[1].center_mm[0] - 250) < 10 and abs(by_x[1].center_mm[1] - 100) < 10
    # 픽셀 중심이 이미지 안쪽, rect_px 크기가 mm 치수/(mm/px) 와 일치 (mm/px = D/f = 3)
    for b in boxes:
        assert 0 <= b.center_px[0] < 640 and 0 <= b.center_px[1] < 480
        assert abs(max(b.rect_px[1]) * 3.0 - b.dims_mm[0]) < 12


def test_gaussian_noise_8mm_keeps_two_boxes():
    fr = make_frame(two_boxes(), noise_mm=8.0, seed=1)
    boxes = detect_boxes(fr, sku=SKU)
    assert len(boxes) == 2
    for b in boxes:
        assert rel_err(b.dims_mm) < 0.05, b.dims_mm
        assert 6.0 < b.plane_rms_mm < 10.0  # 주입한 σ 를 평면 잔차로 회수
        assert b.tilt_deg < 1.0


def test_tilted_box_normal_and_tilt_deg():
    tilt = 5.0
    boxes_spec = [SynthBox((-250.0, 0.0), depth_mm=DEPTH, slope_x=math.tan(math.radians(tilt))),
                  SynthBox((250.0, 100.0), depth_mm=DEPTH, yaw_deg=90.0)]
    boxes = detect_boxes(make_frame(boxes_spec), sku=SKU)
    assert len(boxes) == 2
    tilted = min(boxes, key=lambda b: b.center_mm[0])
    flat = max(boxes, key=lambda b: b.center_mm[0])
    assert abs(tilted.tilt_deg - tilt) < 0.3
    assert flat.tilt_deg < 0.3
    # 평면 Z = z0 + s(X-x0) 의 카메라향 법선 ∝ (s, 0, -1)
    assert tilted.normal[0] > 0 and abs(tilted.normal[1]) < 0.02 and tilted.normal[2] < 0
    assert abs(np.linalg.norm(tilted.normal) - 1) < 1e-3
    assert rel_err(tilted.dims_mm) < 0.05


def test_real_depth_grid_six_boxes():
    """실측 거리(2970mm, ~49x37px)에서 7mm 간격 2x3 격자 → 6개. 경계 침식 편향은 ~-3~-5% (문서화)."""
    grid = [SynthBox((x, y)) for x in (-300.0, 0.0, 300.0) for y in (-115.0, 115.0)]
    boxes = detect_boxes(make_frame(grid, noise_mm=3.0), sku=SKU)
    assert len(boxes) == 6
    for b in boxes:
        assert rel_err(b.dims_mm) < 0.08, b.dims_mm
        assert b.dims_mm[0] < SKU[0] and b.dims_mm[1] < SKU[1]  # 과소 편향 방향 고정
        assert b.confidence > 0.7


def test_touching_boxes_split_by_intensity_seam():
    """맞닿은 박스는 깊이 단차가 없어 강도(I) 에지로만 분리된다 (체커보드 강도)."""
    xs, ys = (-293.0, 0.0, 293.0), (-109.5, 109.5)
    grid = [SynthBox((x, y), intensity=(900.0 if (i + j) % 2 == 0 else 450.0))
            for i, x in enumerate(xs) for j, y in enumerate(ys)]
    boxes = detect_boxes(make_frame(grid), sku=SKU)
    assert len(boxes) == 6
    for b in boxes:
        assert rel_err(b.dims_mm) < 0.10, b.dims_mm


def test_sentinel_region_is_invalid_and_ignored():
    fr = make_frame(two_boxes(), invalid_rect_px=(0, 0, 200, 120))
    blk = (slice(0, 120), slice(0, 200))
    assert not fr.valid[blk].any()
    assert np.all(fr.D[blk] > SENTINEL_D) and np.all(np.abs(fr.X[blk]) > SENTINEL_XY)
    assert np.array_equal(fr.valid, valid_mask(fr.X, fr.Y, fr.D))
    assert fr.n_valid == 640 * 480 - 200 * 120
    boxes = detect_boxes(fr, sku=SKU)
    assert len(boxes) == 2 and all(rel_err(b.dims_mm) < 0.05 for b in boxes)


def test_sentinel_block_over_a_box_removes_only_that_box():
    fr = make_frame(two_boxes(), invalid_rect_px=(0, 0, 300, 480))  # 왼쪽 박스(중심 u≈237) 전부 덮음
    boxes = detect_boxes(fr, sku=SKU)
    assert len(boxes) == 1
    assert abs(boxes[0].center_mm[0] - 250) < 10 and rel_err(boxes[0].dims_mm) < 0.05


def test_zero_depth_and_nonfinite_are_invalid():
    fr = make_frame(two_boxes())
    D = fr.D.copy()
    D[10:20, 10:20] = 0.0
    D[30:40, 30:40] = np.nan
    D[50:60, 50:60] = np.inf
    fr2 = Frame(fr.X, fr.Y, D, fr.I)
    assert not fr2.valid[10:20, 10:20].any()
    assert not fr2.valid[30:40, 30:40].any()
    assert not fr2.valid[50:60, 50:60].any()
    assert len(detect_boxes(fr2, sku=SKU)) == 2


@pytest.mark.parametrize("kind", ["all_sentinel", "zeros", "nan"])
def test_empty_frame_yields_zero_boxes(kind):
    if kind == "all_sentinel":
        fr = make_frame([], floor="invalid")
    elif kind == "zeros":
        z = np.zeros((480, 640), np.float32)
        fr = Frame(z, z, z, z)
    else:
        z = np.full((480, 640), np.nan, np.float32)
        fr = Frame(z, z, z, z)
    assert fr.n_valid == 0
    top = detect_top_layer(fr)
    assert not top.ok and top.n_px == 0
    assert detect_boxes(fr, sku=SKU) == []


def test_flat_floor_without_boxes_is_filtered_by_sku_tol_or_confidence():
    """박스 없는 평평한 바닥: 원 알고리즘은 바닥 전체를 후보 1개로 잡는다 → SKU 허용/신뢰도로 걸러진다."""
    fr = make_frame([], floor="flat")
    raw = detect_boxes(fr, sku=SKU)
    assert len(raw) <= 1
    if raw:
        assert raw[0].dims_mm[0] > 2 * SKU[0]
        assert raw[0].confidence <= 0.5  # 치수 항 0
    assert detect_boxes(fr, sku=SKU, sku_tol=0.5) == []


def test_frame_validation_and_session_roundtrip():
    with pytest.raises(ValueError):
        Frame(np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 5)), np.zeros((4, 4)))
    fr = make_frame(two_boxes())
    sess = fr.as_session()
    fr2 = Frame.from_session(sess, source="x")
    assert fr2.source == "x" and fr2.n_valid == fr.n_valid
    with pytest.raises(KeyError):
        Frame.from_session({"D": fr.D})
