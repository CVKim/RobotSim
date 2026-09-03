# -*- coding: utf-8 -*-
"""robotsim_perception — ToF 빈피킹 인식 파이프라인 패키지 (순수 numpy/opencv).

공개 API:
  load_frame(session_dir)            -> Frame
  detect_top_layer(frame)            -> TopLayer (depth_mm, mask)
  detect_boxes(frame, sku=...)       -> list[Box]
  pick_plan(boxes, dest_xy_mm)       -> list[PickStep]  (열 스캔 규칙)
  run_frame(frame, ...) / run_session(session_dir, ...) -> Result (JSON 직렬화 가능)
  render_overlay(frame, result, out_path)

알고리즘은 tools/binpick_topface.py · tools/binpick_pickpoints.py · tools/pick_next.py 와
동일하며(수치 파리티 테스트로 보증), 외부 의존 없이 설치 가능하도록 최소 함수만 복제했다.
"""
from .frame import Frame, load_frame, valid_mask, SENTINEL_D, SENTINEL_XY
from .detect import Box, TopLayer, detect_top_layer, detect_boxes, DEFAULT_SKU
from .planner import PickStep, pick_plan, DEFAULT_DEST_XY_MM, DEFAULT_COL_TOL_MM
from .pipeline import Result, run_frame, run_session, SCHEMA_VERSION
from .render import render_overlay

__version__ = "0.1.0"

__all__ = [
    "Frame", "load_frame", "valid_mask", "SENTINEL_D", "SENTINEL_XY",
    "Box", "TopLayer", "detect_top_layer", "detect_boxes", "DEFAULT_SKU",
    "PickStep", "pick_plan", "DEFAULT_DEST_XY_MM", "DEFAULT_COL_TOL_MM",
    "Result", "run_frame", "run_session", "SCHEMA_VERSION",
    "render_overlay", "__version__",
]
