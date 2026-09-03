# -*- coding: utf-8 -*-
"""픽 순서 계획: 열 스캔 규칙 (tools/pick_next.py).

실제 픽 순서 마이닝 결과(n=16 전이): 그리디 최근접 50% → 열 스캔 규칙 81% 일치.
규칙: 목적지(DEST)에 가장 가까운 박스가 속한 열(|dx| < col_tol_mm) 에서
      목적지에서 가장 먼 박스부터 픽한다. 이를 남은 박스가 없을 때까지 반복해 전체 순서를 만든다.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

# explore/pickpoints/pick_order_eval.json 의 dest_xy (목적지 스택 ROI 중심, 카메라 좌표 mm)
DEFAULT_DEST_XY_MM = (-1191.4, -38.3)
DEFAULT_COL_TOL_MM = 80.0


@dataclass
class PickStep:
    order: int            # 1-based 픽 순서
    box_id: int           # detect_boxes 가 부여한 Box.id
    center_mm: tuple      # 박스 상면 중심 (X, Y, Z)
    dist_mm: float        # 목적지까지 XY 평면 거리
    anchor_id: int        # 이 스텝에서 열을 정한 목적지 최근접 박스 id
    column_size: int      # 열에 포함된 (당시 남은) 박스 수
    reason: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["center_mm"] = [float(v) for v in self.center_mm]
        return d


def _xy(box):
    c = box.center_mm if hasattr(box, "center_mm") else box["center_mm"]
    return float(c[0]), float(c[1])


def pick_plan(boxes: Sequence, dest_xy_mm=DEFAULT_DEST_XY_MM, col_tol_mm: float = DEFAULT_COL_TOL_MM) -> list:
    """boxes(list[Box] 또는 center_mm 키를 가진 dict) → 열 스캔 규칙 순서의 list[PickStep]."""
    dx0, dy0 = float(dest_xy_mm[0]), float(dest_xy_mm[1])
    ids = [getattr(b, "id", i) for i, b in enumerate(boxes)]
    xy = [_xy(b) for b in boxes]
    dist = [float(np.hypot(x - dx0, y - dy0)) for x, y in xy]
    centers = [tuple(float(v) for v in (b.center_mm if hasattr(b, "center_mm") else b["center_mm"]))
               for b in boxes]

    remaining = list(range(len(boxes)))
    steps = []
    while remaining:
        anchor = min(remaining, key=lambda i: (dist[i], i))
        col = [i for i in remaining if abs(xy[i][0] - xy[anchor][0]) < col_tol_mm]
        pick = max(col, key=lambda i: (dist[i], -i))
        steps.append(PickStep(
            order=len(steps) + 1, box_id=int(ids[pick]), center_mm=centers[pick],
            dist_mm=round(dist[pick], 1), anchor_id=int(ids[anchor]), column_size=len(col),
            reason="nearest-column farthest-first",
        ))
        remaining.remove(pick)
    return steps
