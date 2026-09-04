# -*- coding: utf-8 -*-
"""런타임 판정 계층(runtime.decide / HealthMonitor) 테스트 — 합성 프레임만 사용."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robotsim_perception.frame import Frame  # noqa: E402
from robotsim_perception.runtime import (Decision, HealthMonitor,  # noqa: E402
                                         Thresholds, decide)
from robotsim_perception.synthetic import SynthBox, make_frame  # noqa: E402

SKU = (293.0, 219.0, 283.0)


def _scene(n=2, depth=1500.0, noise=0.0, seed=0):
    boxes = [SynthBox((-250 + 500 * i, 0), depth_mm=depth) for i in range(n)]
    return make_frame(boxes, noise_mm=noise, seed=seed)


def test_ok_status_with_boxes():
    d = decide(_scene(2), Thresholds(lattice=False), sku=SKU)
    assert d.status == "OK" and d.ok
    assert d.n_boxes >= 1 and d.n_pickable >= 1
    assert d.plan and len(d.plan) == d.n_pickable
    assert [s.order for s in d.plan] == list(range(1, len(d.plan) + 1))   # order 는 1부터
    assert d.top_depth_mm is not None and d.latency_ms > 0


def test_retake_when_frame_mostly_invalid():
    fr = _scene(2)
    D = fr.D.copy()
    rng = np.random.default_rng(0)
    D[rng.random(D.shape) < 0.9] = 16383.75          # 대부분 무효화
    bad = Frame(X=fr.X, Y=fr.Y, D=D, I=fr.I)
    d = decide(bad, Thresholds(min_valid_frac=0.25, lattice=False), sku=SKU)
    assert d.status == "RETAKE"
    assert "valid pixels" in d.reason


def test_no_surface_when_all_invalid():
    fr = _scene(1)
    D = np.full_like(fr.D, 16383.75)
    d = decide(Frame(X=fr.X, Y=fr.Y, D=D, I=fr.I),
               Thresholds(min_valid_frac=0.0, lattice=False), sku=SKU)
    assert d.status in ("NO_SURFACE", "RETAKE", "LAYER_EMPTY")


def test_layer_empty_on_flat_ground():
    """박스 없는 평평한 면 — SKU 필터가 걸러 LAYER_EMPTY 가 되어야 한다."""
    fr = make_frame([], noise_mm=0.0, seed=0)
    d = decide(fr, Thresholds(sku_tol=0.35, lattice=False), sku=SKU)
    assert d.status in ("LAYER_EMPTY", "NO_SURFACE", "LOW_CONFIDENCE")
    assert d.n_pickable == 0


def test_low_confidence_gate():
    """신뢰도 임계를 1.0 으로 올리면 어떤 박스도 통과하지 못한다."""
    d = decide(_scene(2), Thresholds(min_confidence=1.01, lattice=False), sku=SKU)
    assert d.status == "LOW_CONFIDENCE"
    assert d.n_boxes >= 1 and d.n_pickable == 0
    assert d.best_confidence is not None


def test_source_roi_filters_far_boxes():
    fr = _scene(2)
    wide = decide(fr, Thresholds(lattice=False), sku=SKU)
    narrow = decide(fr, Thresholds(lattice=False, source_roi_mm=50.0), sku=SKU)
    assert narrow.n_boxes <= wide.n_boxes
    assert narrow.status in ("LAYER_EMPTY", "OK", "LOW_CONFIDENCE")


def test_thresholds_roundtrip(tmp_path):
    th = Thresholds(min_valid_frac=0.3, min_confidence=0.7, lattice=False)
    p = tmp_path / "th.json"
    th.to_json(p)
    back = Thresholds.from_json(p)
    assert back.min_valid_frac == 0.3 and back.min_confidence == 0.7
    assert back.lattice is False
    # 알 수 없는 키가 있어도 무시하고 로드되어야 한다 (설정 파일 버전 차이 대응)
    d = json.loads(p.read_text(encoding="utf-8"))
    d["future_option"] = 1
    p.write_text(json.dumps(d), encoding="utf-8")
    assert Thresholds.from_json(p).min_confidence == 0.7


def test_decision_serializable_and_logline():
    d = decide(_scene(2), Thresholds(lattice=False), sku=SKU)
    js = json.dumps(d.to_dict())          # 전부 순수 파이썬 타입이어야 함
    assert '"status"' in js
    line = json.loads(d.log_line())
    assert line["status"] == d.status and "latency_ms" in line


def test_health_monitor_detects_shift():
    fr = _scene(2)
    hm = HealthMonitor()
    assert hm.check(fr)["status"] == "NO_REFERENCE"
    n = hm.set_reference(fr)
    assert n > 0
    assert hm.check(fr)["status"] == "OK"          # 같은 프레임 -> 드리프트 0
    moved = Frame(X=fr.X, Y=fr.Y, D=fr.D + 40.0, I=fr.I)   # 카메라가 40mm 밀린 상황
    res = hm.check(moved)
    assert res["status"] == "FAIL"
    assert res["drift_mm"] > 25.0
