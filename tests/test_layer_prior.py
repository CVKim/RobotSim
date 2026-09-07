# -*- coding: utf-8 -*-
"""층 선택 보강 두 가지 (합성 프레임, 회사 데이터 불필요).

1. layer_roi_mm — 층 히스토그램을 팔레트 영역(카메라 XY)으로 한정: 팔레트 밖에 카메라에 더 가까운 물체(팔·설비)가 있어도
   층 피크가 그쪽으로 넘어가지 않는다.
2. prior_top_mm — 시간 사전: 박스가 1~2개만 남아 그 층의 픽셀 질량이 바닥에 눌려 후보 피크에 못 들고 SKU 신뢰(strong)도 못 받을 때,
   직전 프레임의 층 깊이를 후보로 넣어 그 층의 박스를 돌려준다. 사전 층에 박스가 없으면(층이 비었다) 기존 규칙으로 돌아간다.
기본값(둘 다 None)은 기존 동작과 같아야 한다."""
import numpy as np

from robotsim_perception.detect import detect_layer_and_boxes, detect_top_layer
from robotsim_perception.geometry import detect_boxes_v2, find_top_layer, layer_roi_mask, top_layer_candidates
from robotsim_perception.runtime import Thresholds, decide
from robotsim_perception.synthetic import SynthBox, make_frame

TOP = 2970.0
FLOOR = 3253.0


def _sess(fr):
    return {"X": fr.X, "Y": fr.Y, "D": fr.D, "I": fr.I}


def _with_arm_like_object(fr, depth_mm=2700.0):
    """팔레트 밖(X 700~900 mm, 화면 중앙 ROI 안)에 카메라에 더 가까운 큰 평면을 그린다 — 팔·설비 대용."""
    m = (fr.X > 650) & (fr.X < 950) & (np.abs(fr.Y) < 900) & fr.valid
    D = fr.D.copy()
    D[m] = depth_mm
    fr.D = D
    return fr


def test_layer_roi_ignores_objects_outside_the_pallet():
    boxes = [SynthBox((x, y), depth_mm=TOP) for x in (-150.0, 150.0) for y in (-112.5, 112.5)]
    fr = _with_arm_like_object(make_frame(boxes))              # 기본 바닥은 기울어져 있어(실측처럼) 단일 피크를 만들지 않는다
    sess = _sess(fr)
    roi = layer_roi_mask(sess, 620.0)
    plain = find_top_layer(fr.D, fr.valid)
    with_roi = find_top_layer(fr.D, fr.valid, roi_mask=roi)
    assert abs(plain - 2700.0) < 15, plain                     # 화면 중앙 ROI: 팔레트 밖의 가까운 큰 물체가 '상면' 피크가 된다
    assert abs(with_roi - TOP) < 15, with_roi                  # 팔레트 ROI: 박스 상면
    assert all(abs(c - 2700.0) > 15 for c in top_layer_candidates(fr.D, fr.valid, k=8, roi_mask=roi))
    assert abs(detect_top_layer(fr, roi_mm=620.0).depth_mm - TOP) < 15
    # v2 최종 결과: ROI 없이도 fallback 이 박스 층을 찾아 4개, ROI 를 주면 처음부터 박스 층 — 둘 다 4개 (회귀 확인)
    for kw in ({}, {"layer_roi_mm": 620.0}):
        top_d, _, det = detect_boxes_v2(sess, **kw)
        assert len(det) == 4 and abs(top_d - TOP) < 15, (kw, top_d, len(det))


def _sparse_top_scene(top_box=True):
    """가득 찬 아래층(12개, 3253 mm) 위에 SKU 에서 벗어난 작은 상면 하나(2970 mm) — 침식된 잔여 박스의 대용(strong 판정 못 받음).
    make_frame 은 나중 박스가 앞의 박스를 덮으므로 위층 박스를 마지막에 그린다(가림)."""
    lower = [SynthBox(((c - 1.5) * 300.0, (r - 1.0) * 225.0), depth_mm=FLOOR) for r in range(3) for c in range(4)]
    top = [SynthBox((150.0, 0.0), size_mm=(230.0, 170.0), depth_mm=TOP)] if top_box else []
    return make_frame(lower + top)


def test_temporal_prior_recovers_a_single_weak_box_over_a_full_layer():
    fr = _sparse_top_scene()
    sess = _sess(fr)
    top_d0, _, det0 = detect_boxes_v2(sess)
    assert abs(top_d0 - FLOOR) < 15 and len(det0) >= 9, (top_d0, len(det0))      # 기존: 아래층으로 점프 (docs/41 5절의 실패 모드 재현)
    dbg = {}
    top_d1, _, det1 = detect_boxes_v2(sess, prior_top_mm=TOP, debug=dbg)
    assert abs(top_d1 - TOP) < 15 and len(det1) == 1 and dbg["layer_rule"] == "prior", (top_d1, len(det1), dbg)
    assert abs(det1[0]["center_mm"][0] - 150.0) < 30 and abs(det1[0]["center_mm"][1]) < 30
    # 사전 층에 박스가 없으면(마지막 박스를 집은 뒤) 기존 규칙으로 돌아가 아래층을 낸다 — 예외 없이
    top_d2, _, det2 = detect_boxes_v2(_sess(_sparse_top_scene(top_box=False)), prior_top_mm=TOP)
    assert abs(top_d2 - FLOOR) < 15 and len(det2) >= 12, (top_d2, len(det2))   # 격자 보완이 가장자리 셀을 더 채울 수 있다


def test_decide_reports_selected_layer_and_uses_prior():
    fr = _sparse_top_scene()
    th_off = Thresholds(lattice=True, min_confidence=0.3, source_roi_mm=620.0)
    th_on = Thresholds(lattice=True, min_confidence=0.3, source_roi_mm=620.0, temporal_prior=True, layer_roi_mm=620.0)
    d_off = decide(fr, th_off, prior_top_mm=TOP)                   # temporal_prior=False 면 사전을 무시한다
    assert abs(d_off.top_depth_mm - FLOOR) < 15 and d_off.n_boxes >= 9, d_off.log_line()
    d_on = decide(fr, th_on, prior_top_mm=TOP)
    assert d_on.n_boxes == 1 and abs(d_on.top_depth_mm - TOP) < 15, d_on.log_line()
    top_d, boxes = detect_layer_and_boxes(fr, lattice=True, prior_top_mm=TOP)
    assert len(boxes) == 1 and abs(top_d - TOP) < 15
