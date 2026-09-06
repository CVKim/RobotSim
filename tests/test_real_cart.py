# -*- coding: utf-8 -*-
"""실측 대차 세션 파리티 테스트 (로컬 전용). tools/local_paths.DAECHA_DIR 또는 환경변수 DAECHA_DIR 이 없으면 skip.

robotsim_perception.cart.analyze_cart 가 tools/hook_analysis_v2.py 의 공개 결과
(results/hook_repeatability.json, wall_top 추정)와 수치 일치하는지 검사한다.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _sessions():
    tools = REPO_ROOT / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    try:
        from local_paths import DAECHA_DIR
    except ImportError:
        DAECHA_DIR = os.environ.get("DAECHA_DIR")
    if not DAECHA_DIR or not Path(DAECHA_DIR).is_dir():
        return []
    return sorted(p for p in Path(DAECHA_DIR).iterdir() if (p / "4_tof_D.mim").exists())


SESSIONS = _sessions()
pytestmark = pytest.mark.skipif(not SESSIONS, reason="실측 대차 데이터 없음 (로컬 전용)")


def test_parity_with_published_hook_results():
    from robotsim_perception import load_frame
    from robotsim_perception.cart import analyze_cart
    ref = json.loads((REPO_ROOT / "results" / "hook_repeatability.json").read_text(encoding="utf-8"))
    ref_sessions = ref["sessions"]
    assert len(ref_sessions) == len(SESSIONS)
    cart = []
    for sp, rs in zip(SESSIONS, ref_sessions):
        res = analyze_cart(load_frame(sp))
        assert res.status == "OK", (sp.name, res.reason)
        assert np.allclose(res.hook_cam_mm, rs["estimates"]["wall_top"]["cam"], atol=0.05)
        assert np.allclose(res.hook_cart_mm, rs["estimates"]["wall_top"]["cart"], atol=0.05)
        assert abs(res.plane["camera_height_mm"] - rs["plane"]["camera_height_above_plate_mm"]) < 0.2
        assert res.latency_ms < 1500
        cart.append(res.hook_cart_mm)
    cart = np.asarray(cart)
    rms = float(np.sqrt(np.mean(np.sum((cart - cart.mean(axis=0)) ** 2, axis=1))))
    # 공개 수치(results/hook_repeatability.json): 구조 프레임(cart_frame) 3.85 mm, 데크 ICP 정련(icp_deck) 3.16 mm.
    # 이 모듈은 ICP 없이 구조 프레임만 쓰므로 3.85 가 기준이다. 검토에서 3.2 로 잘못 적었던 것을 바로잡음.
    assert 3.6 < rms < 4.1, rms
