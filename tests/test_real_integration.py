# -*- coding: utf-8 -*-
"""실측 ToF 세션 통합 테스트. tools/local_paths.py (또는 BINPICK_DIR 환경변수) 가 없으면 skip.

- 첫 세션에서 박스 검출·치수·기울기·지연 검사
- tools/binpick_topface.detect_boxes 와의 수치 파리티 (동일 검출 집합)
- CLI 실행으로 JSON/PNG 생성
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS = REPO_ROOT / "tools"


def _binpick_dir():
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    try:
        from local_paths import BINPICK_DIR  # noqa: WPS433
    except ImportError:
        BINPICK_DIR = os.environ.get("BINPICK_DIR")
    if not BINPICK_DIR or not Path(BINPICK_DIR).is_dir():
        return None
    sessions = sorted(d for d in Path(BINPICK_DIR).iterdir() if d.is_dir() and (d / "4_tof_D.mim").exists())
    return sessions[0] if sessions else None


def _all_sessions():
    """BINPICK_DIR 하위 전체 세션 (SESSION 의 부모에서 유도 — BINPICK_DIR 은 함수 지역)."""
    if SESSION is None:
        return []
    root = Path(SESSION).parent
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "4_tof_D.mim").exists())


SESSION = _binpick_dir()
pytestmark = pytest.mark.skipif(SESSION is None, reason="BINPICK_DIR not available (local_paths.py or env)")


@pytest.fixture(scope="module")
def result():
    from robotsim_perception import run_session
    return run_session(SESSION)


def test_first_session_boxes_and_dims(result):
    boxes = result.boxes
    assert result.frame.shape == (480, 640) and 0.3 < result.frame.valid.mean() < 1.0
    assert result.top_layer.ok and 2000 < result.top_layer.depth_mm < 4500
    assert len(boxes) >= 1
    for b in boxes:
        assert abs(b.dims_mm[0] - 293) < 40 and abs(b.dims_mm[1] - 219) < 40, b.dims_mm
        assert b.tilt_deg < 5.0 and b.plane_rms_mm < 20.0
        assert 0.0 <= b.confidence <= 1.0 and b.normal[2] < 0
        assert abs(b.center_mm[2] - result.top_layer.depth_mm) < 40
    assert result.latency_ms < 1000
    assert len(result.plan) == len(boxes) and [s.order for s in result.plan] == list(range(1, len(boxes) + 1))


def test_parity_with_tools_detect_boxes(result):
    from mim_loader import load_session
    from binpick_topface import detect_boxes as tools_detect
    sess = load_session(SESSION)
    top_ref, _, ref = tools_detect(sess)
    assert abs(result.top_layer.depth_mm - top_ref) < 1e-6
    assert len(ref) == len(result.boxes)
    for a, b in zip(ref, result.boxes):
        assert a["dims_mm"] == b.dims_mm
        assert a["area_px"] == b.area_px
        assert abs(a["depth_mm"] - b.depth_mm) < 1e-6
        assert abs(a["rect_px"][0][0] - b.rect_px[0][0]) < 1e-3


def test_cli_on_real_session(tmp_path, result):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONIOENCODING="utf-8")
    out_json, out_png = tmp_path / "real.json", tmp_path / "real.png"
    r = subprocess.run([sys.executable, "-m", "robotsim_perception", "run", str(SESSION),
                        "--json", str(out_json), "--overlay", str(out_png), "--quiet"],
                       cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    d = json.loads(out_json.read_text(encoding="utf-8"))
    assert d["schema_version"] == "1.0" and d["n_boxes"] == len(result.boxes)
    assert d["latency_ms"] > 0 and "load" in d["latency_breakdown_ms"]
    assert out_png.exists() and out_png.stat().st_size > 10000


def test_parity_all_sessions_v1_and_v2():
    """전 세션 파리티 — 패키지와 tools 가 v1/v2 모두 같은 검출 수를 내야 한다.

    이전 파리티 테스트는 세션 1개만 검사해, 알고리즘이 두 곳에 복제돼 있던 시절
    센티넬 오염 버그 수정이 한쪽에만 적용된 불일치(148 vs 152)를 놓쳤다.
    이제 알고리즘은 robotsim_perception.geometry 한 곳이지만, 배선이 어긋나면 여기서 잡힌다.
    """
    from binpick_topface import detect_boxes as tools_v1, detect_boxes_v2 as tools_v2
    from mim_loader import load_session
    from robotsim_perception import load_frame
    from robotsim_perception.detect import detect_boxes as pkg

    sessions = _all_sessions()
    assert sessions, "no sessions"
    n1 = n2 = p1 = p2 = 0
    for d in sessions:
        sess, frame = load_session(d), load_frame(d)
        a, b = len(tools_v1(sess)[2]), len(tools_v2(sess)[2])
        c, e = len(pkg(frame)), len(pkg(frame, lattice=True))
        assert (a, b) == (c, e), f"{d.name}: tools v1/v2 ({a},{b}) != package ({c},{e})"
        n1, n2, p1, p2 = n1 + a, n2 + b, p1 + c, p2 + e
    assert (n1, n2) == (p1, p2)
    assert n1 == 152, f"v1 baseline changed: {n1} != 152"
    assert n2 == 167, f"v2 baseline changed: {n2} != 167"
