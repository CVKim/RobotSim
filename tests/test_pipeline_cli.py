# -*- coding: utf-8 -*-
"""파이프라인 JSON 스키마 + .mim 세션 라운드트립 + CLI (subprocess) 테스트."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from robotsim_perception import (SCHEMA_VERSION, __version__, load_frame, render_overlay, run_frame,
                                 run_session)
from robotsim_perception.synthetic import SynthBox, make_frame, write_session

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPTH = 1500.0
DEST = (-1000.0, 0.0)


def _frame():
    return make_frame([SynthBox((-250.0, 0.0), depth_mm=DEPTH),
                       SynthBox((250.0, 100.0), depth_mm=DEPTH, yaw_deg=90.0)], noise_mm=2.0)


def _cli(*args, cwd=None):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-m", "robotsim_perception", *args], cwd=cwd or REPO_ROOT,
                          capture_output=True, text=True, env=env)


def test_run_frame_result_and_json_schema():
    res = run_frame(_frame(), dest_xy_mm=DEST)
    d = res.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION == "1.0"
    assert d["package_version"] == __version__
    assert d["n_boxes"] == len(d["boxes"]) == 2
    assert d["latency_ms"] >= 0 and d["latency_ms"] < 5000
    for k in ("top_layer", "detect_boxes", "pick_plan", "total"):
        assert k in d["latency_breakdown_ms"]
    assert d["frame"]["shape"] == [480, 640] and d["frame"]["valid_px"] == 480 * 640
    assert abs(d["top_layer"]["depth_mm"] - DEPTH) < 10
    box = d["boxes"][0]
    for k in ("id", "center_mm", "dims_mm", "tilt_deg", "normal", "rect_px", "confidence", "depth_mm",
              "area_px", "center_px", "plane_rms_mm", "fill"):
        assert k in box, k
    assert set(box["rect_px"]) == {"center", "size", "angle"}
    plan = d["pick_plan"]
    assert plan["dest_xy_mm"] == list(DEST) and len(plan["steps"]) == 2
    assert [s["order"] for s in plan["steps"]] == [1, 2]
    # 왼쪽 박스(x=-250)가 목적지에 더 가깝고 다른 열 → 먼저 픽
    first = next(b for b in d["boxes"] if b["id"] == plan["steps"][0]["box_id"])
    assert first["center_mm"][0] < 0
    s = json.dumps(d)  # 직렬화 가능 (numpy 타입 없음)
    assert json.loads(s)["schema_version"] == "1.0"


def test_min_confidence_filter_reindexes():
    res = run_frame(_frame(), min_confidence=0.99)
    assert res.boxes == [] and res.plan == []
    res = run_frame(_frame(), min_confidence=0.5)
    assert [b.id for b in res.boxes] == list(range(len(res.boxes)))


def test_mim_session_roundtrip_and_run_session(tmp_path):
    fr = _frame()
    sess = tmp_path / "synthetic_session"
    write_session(fr, sess)
    assert sorted(p.name for p in sess.iterdir()) == ["2_tof_X.mim", "3_tof_Y.mim", "4_tof_D.mim", "5_tof_I.mim"]
    fr2 = load_frame(sess)
    for k in ("X", "Y", "D", "I"):
        assert np.array_equal(getattr(fr, k), getattr(fr2, k)), k
    assert fr2.source == str(sess)
    res = run_session(sess, dest_xy_mm=DEST)
    assert len(res.boxes) == 2 and "load" in res.latency_breakdown_ms
    assert res.to_dict()["source"] == str(sess)
    with pytest.raises(FileNotFoundError):
        load_frame(tmp_path / "nope")
    (tmp_path / "partial").mkdir()
    with pytest.raises(FileNotFoundError):
        load_frame(tmp_path / "partial")


def test_render_overlay_writes_png(tmp_path):
    fr = _frame()
    res = run_frame(fr, dest_xy_mm=DEST)
    out = tmp_path / "sub" / "overlay.png"
    img = render_overlay(fr, res, out)
    assert img.shape == (480, 640, 3) and out.exists() and out.stat().st_size > 1000
    import cv2
    loaded = cv2.imdecode(np.fromfile(str(out), np.uint8), cv2.IMREAD_COLOR)
    assert loaded.shape == (480, 640, 3)


def test_cli_run_json_and_overlay(tmp_path):
    sess = tmp_path / "sess"
    write_session(_frame(), sess)
    out_json, out_png = tmp_path / "out.json", tmp_path / "out.png"
    r = _cli("run", str(sess), "--json", str(out_json), "--overlay", str(out_png),
             "--dest", "-1000", "0", "--sku", "293", "219", "283")
    assert r.returncode == 0, r.stderr
    assert "boxes     : 2" in r.stdout and "latency" in r.stdout and "pick plan" in r.stdout
    d = json.loads(out_json.read_text(encoding="utf-8"))
    assert d["schema_version"] == "1.0" and d["n_boxes"] == 2 and d["latency_ms"] >= 0
    assert d["pick_plan"]["dest_xy_mm"] == [-1000.0, 0.0] and d["sku_mm"] == [293.0, 219.0, 283.0]
    assert out_png.exists() and out_png.stat().st_size > 1000


def test_cli_missing_dir_and_version(tmp_path):
    r = _cli("run", str(tmp_path / "missing"))
    assert r.returncode == 2 and "error" in r.stderr
    r = _cli("version")
    assert r.returncode == 0 and __version__ in r.stdout and SCHEMA_VERSION in r.stdout
    r = _cli("run", str(tmp_path), "--dest", "1", "2", "3")
    assert r.returncode == 2  # argparse 오류 (인자 수 불일치)
