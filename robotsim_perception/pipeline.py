# -*- coding: utf-8 -*-
"""프레임 → 최상층 → 박스 → 픽 계획 전체 파이프라인 + JSON 직렬화 (schema_version, latency_ms)."""
from __future__ import annotations

import datetime as _dt
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .detect import DEFAULT_SKU, Box, TopLayer, detect_boxes, detect_top_layer
from .frame import Frame, load_frame
from .planner import DEFAULT_COL_TOL_MM, DEFAULT_DEST_XY_MM, PickStep, pick_plan

SCHEMA_VERSION = "1.0"


@dataclass
class Result:
    frame: Frame
    top_layer: TopLayer
    boxes: list                       # list[Box]
    plan: list                        # list[PickStep]
    sku: Optional[tuple]
    dest_xy_mm: tuple
    col_tol_mm: float
    latency_ms: float                 # 인식 연산 시간 (최상층+검출+계획, 디스크 로드 제외)
    latency_breakdown_ms: dict = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> dict:
        from . import __version__
        fr = self.frame
        return {
            "schema_version": SCHEMA_VERSION,
            "package_version": __version__,
            "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": self.source or fr.source,
            "frame": {"shape": [int(v) for v in fr.shape], "valid_px": fr.n_valid,
                      "valid_frac": round(float(fr.valid.mean()), 4) if fr.valid.size else 0.0},
            "sku_mm": [float(v) for v in self.sku] if self.sku is not None else None,
            "top_layer": {"depth_mm": (round(float(self.top_layer.depth_mm), 1)
                                       if self.top_layer.ok else None),
                          "tol_mm": self.top_layer.tol_mm, "n_px": self.top_layer.n_px},
            "n_boxes": len(self.boxes),
            "boxes": [b.to_dict() for b in self.boxes],
            "pick_plan": {"rule": "column-scan (nearest-column, farthest-first)",
                          "dest_xy_mm": [float(v) for v in self.dest_xy_mm],
                          "col_tol_mm": float(self.col_tol_mm),
                          "steps": [s.to_dict() for s in self.plan]},
            "latency_ms": round(float(self.latency_ms), 2),
            "latency_breakdown_ms": {k: round(float(v), 2) for k, v in self.latency_breakdown_ms.items()},
        }

    def to_json(self, path=None, indent: int = 1) -> str:
        s = json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(s)
        return s


def run_frame(frame: Frame, sku: Optional[Sequence[float]] = DEFAULT_SKU, dest_xy_mm=DEFAULT_DEST_XY_MM,
              col_tol_mm: float = DEFAULT_COL_TOL_MM, tol_mm: float = 40, min_area_px: int = 700,
              sku_tol: Optional[float] = None, min_confidence: float = 0.0,
              lattice: bool = False, load_ms: Optional[float] = None) -> Result:
    """메모리 상의 Frame 에 대해 전체 파이프라인 실행."""
    t0 = time.perf_counter()
    top = detect_top_layer(frame, tol_mm=tol_mm)
    t1 = time.perf_counter()
    boxes = detect_boxes(frame, sku=sku, tol_mm=tol_mm, min_area_px=min_area_px, sku_tol=sku_tol,
                         top_layer=None if lattice else top, lattice=lattice)
    if min_confidence > 0:
        boxes = [b for b in boxes if b.confidence >= min_confidence]
        for k, b in enumerate(boxes):
            b.id = k
    t2 = time.perf_counter()
    plan = pick_plan(boxes, dest_xy_mm=dest_xy_mm, col_tol_mm=col_tol_mm)
    t3 = time.perf_counter()
    breakdown = {"top_layer": (t1 - t0) * 1e3, "detect_boxes": (t2 - t1) * 1e3, "pick_plan": (t3 - t2) * 1e3}
    if load_ms is not None:
        breakdown["load"] = load_ms
    breakdown["total"] = sum(breakdown.values())
    return Result(frame=frame, top_layer=top, boxes=boxes, plan=plan,
                  sku=tuple(float(v) for v in sku) if sku is not None else None,
                  dest_xy_mm=tuple(float(v) for v in dest_xy_mm), col_tol_mm=float(col_tol_mm),
                  latency_ms=(t3 - t0) * 1e3, latency_breakdown_ms=breakdown, source=frame.source)


def run_session(session_dir, **kwargs) -> Result:
    """세션 폴더(.mim) 로드 후 run_frame. latency_breakdown_ms 에 load 포함."""
    t0 = time.perf_counter()
    frame = load_frame(session_dir)
    load_ms = (time.perf_counter() - t0) * 1e3
    res = run_frame(frame, load_ms=load_ms, **kwargs)
    res.source = str(session_dir)
    return res
