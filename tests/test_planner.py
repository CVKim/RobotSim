# -*- coding: utf-8 -*-
"""열 스캔 픽 계획 규칙 (tools/pick_next.py) 단위 테스트."""
import math

from robotsim_perception import pick_plan, PickStep

DEST = (-1000.0, 0.0)


def _boxes(xy_list, z=2970.0):
    return [{"center_mm": (x, y, z)} for x, y in xy_list]


def _dist(xy):
    return math.hypot(xy[0] - DEST[0], xy[1] - DEST[1])


def test_column_scan_nearest_column_farthest_first():
    # 열 A (x=-500) 3개, 열 B (x=0) 2개, 열 C (x=500) 1개
    pts = [(-500, 0), (-500, -300), (-500, 300), (0, -300), (0, 300), (500, 0)]
    steps = pick_plan(_boxes(pts), DEST, col_tol_mm=80)
    assert [s.order for s in steps] == list(range(1, 7))
    assert sorted(s.box_id for s in steps) == list(range(6))
    ids = [s.box_id for s in steps]
    # 목적지 최근접은 (-500,0) → 열 A → 가장 먼 것부터: (-500,-300)/( -500,300) 은 동거리 → 낮은 id 우선
    assert ids[0] == 1 and ids[1] == 2 and ids[2] == 0
    assert set(ids[3:5]) == {3, 4} and ids[5] == 5
    for s in steps:
        assert isinstance(s, PickStep)
        assert abs(s.dist_mm - _dist(pts[s.box_id])) < 0.1
        assert s.anchor_id in range(6) and s.column_size >= 1
        assert s.center_mm == tuple(float(v) for v in (pts[s.box_id][0], pts[s.box_id][1], 2970.0))
    assert steps[0].anchor_id == 0 and steps[0].column_size == 3


def test_column_tolerance_controls_grouping():
    pts = [(-500, 0), (-440, 300)]  # dx = 60mm
    wide = pick_plan(_boxes(pts), DEST, col_tol_mm=80)
    narrow = pick_plan(_boxes(pts), DEST, col_tol_mm=50)
    assert [s.box_id for s in wide] == [1, 0]      # 같은 열 → 먼 것(1) 먼저
    assert [s.box_id for s in narrow] == [0, 1]    # 다른 열 → 그리디 최근접
    assert wide[0].column_size == 2 and narrow[0].column_size == 1


def test_tiny_tolerance_degenerates_to_greedy_nearest():
    pts = [(300, 0), (-700, 0), (-200, 200), (100, -400)]
    steps = pick_plan(_boxes(pts), DEST, col_tol_mm=1e-6)
    d = [_dist(p) for p in pts]
    assert [s.box_id for s in steps] == sorted(range(4), key=lambda i: d[i])


def test_empty_and_single():
    assert pick_plan([], DEST) == []
    s = pick_plan(_boxes([(0, 0)]), DEST)
    assert len(s) == 1 and s[0].order == 1 and s[0].box_id == 0 and s[0].anchor_id == 0


def test_accepts_objects_with_id_and_center_mm():
    class B:
        def __init__(self, i, c):
            self.id, self.center_mm = i, c
    boxes = [B(10, (-500, 0, 1)), B(20, (-500, 300, 1))]
    steps = pick_plan(boxes, DEST)
    assert [s.box_id for s in steps] == [20, 10]
    assert steps[0].to_dict()["center_mm"] == [-500.0, 300.0, 1.0]
