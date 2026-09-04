# -*- coding: utf-8 -*-
"""런타임 판정 계층 — 상위 제어기(PLC/로봇)가 바로 분기할 수 있는 상태를 낸다.

지금까지 이 패키지는 박스 목록만 반환했다. 실제 셀에서는 그것만으로 부족하다.
"박스 0개"가 층이 비어서인지, 프레임이 나빠서인지, 인식이 실패해서인지 구분하지 못하면
제어기는 멈추거나 잘못 진행한다. 이 모듈이 그 구분을 담당한다.

상태(Status)
    OK              집을 박스가 있고 신뢰도도 충분 -> pick_plan 실행
    RETAKE          프레임 품질 미달(유효 픽셀 부족) -> 재촬영 요청
    LOW_CONFIDENCE  박스는 찾았으나 최선 후보가 임계 미만 -> 재촬영 또는 사람 호출
    LAYER_EMPTY     최상층 탐색은 됐으나 박스 0개 -> 다음 층/다음 팔레트로
    NO_SURFACE      최상층 자체를 못 찾음(빈 팔레트·센서 이상) -> 상위 판단 필요

설계 원칙: 이 모듈은 판정만 하고 로봇을 움직이지 않는다. 모든 임계는 Thresholds 로 노출해
현장에서 조정하고 로그로 추적할 수 있게 한다(코드에 상수를 박아 두지 않는다).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .detect import DEFAULT_SKU, Box, detect_boxes, detect_top_layer
from .frame import Frame
from .planner import DEFAULT_COL_TOL_MM, DEFAULT_DEST_XY_MM, pick_plan


@dataclass
class Thresholds:
    """현장 조정 대상. from_json/to_json 으로 설정 파일화."""
    min_valid_frac: float = 0.25        # 프레임 유효 픽셀 비율 하한
    min_confidence: float = 0.55        # 픽 후보 신뢰도 하한
    max_tilt_deg: float = 12.0          # 이보다 기울면 석션 부적합
    min_boxes_for_ok: int = 1
    sku_tol: Optional[float] = 0.35     # SKU 대비 치수 허용(병합·조각 제거). None 이면 끔
    lattice: bool = True                # v2 격자 보완 사용 여부
    source_roi_mm: Optional[float] = None   # 소스 팔레트 반경(카메라 XY). None 이면 제한 없음

    @classmethod
    def from_json(cls, path) -> "Thresholds":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_json(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=1, ensure_ascii=False),
                              encoding="utf-8")


@dataclass
class Decision:
    status: str
    reason: str
    n_boxes: int = 0
    n_pickable: int = 0
    valid_frac: float = 0.0
    top_depth_mm: Optional[float] = None
    best_confidence: Optional[float] = None
    boxes: list = field(default_factory=list)
    plan: list = field(default_factory=list)
    latency_ms: float = 0.0
    thresholds: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["boxes"] = [b.to_dict() for b in self.boxes]
        d["plan"] = [asdict(s) for s in self.plan]
        return d

    def log_line(self) -> str:
        """구조화 로그 한 줄 (사고 후 재현용 — 프레임별로 남긴다)."""
        return json.dumps({
            "status": self.status, "reason": self.reason, "n_boxes": self.n_boxes,
            "n_pickable": self.n_pickable, "valid_frac": round(self.valid_frac, 3),
            "top_depth_mm": self.top_depth_mm, "best_conf": self.best_confidence,
            "latency_ms": round(self.latency_ms, 1),
        }, ensure_ascii=False)


def decide(frame: Frame, th: Optional[Thresholds] = None,
           sku: Optional[Sequence[float]] = DEFAULT_SKU,
           dest_xy_mm=DEFAULT_DEST_XY_MM, col_tol_mm: float = DEFAULT_COL_TOL_MM) -> Decision:
    """프레임 하나 -> 실행 가능한 판정."""
    th = th or Thresholds()
    t0 = time.perf_counter()
    valid_frac = float(frame.valid.mean())
    thr = asdict(th)

    if valid_frac < th.min_valid_frac:
        return Decision(status="RETAKE",
                        reason=f"valid pixels {valid_frac:.1%} < {th.min_valid_frac:.0%}",
                        valid_frac=valid_frac, thresholds=thr,
                        latency_ms=(time.perf_counter() - t0) * 1e3)

    top = detect_top_layer(frame)
    if not top.ok:
        return Decision(status="NO_SURFACE", reason="top layer not found",
                        valid_frac=valid_frac, thresholds=thr,
                        latency_ms=(time.perf_counter() - t0) * 1e3)

    boxes = detect_boxes(frame, sku=sku, sku_tol=th.sku_tol, lattice=th.lattice)
    if th.source_roi_mm is not None:
        r = float(th.source_roi_mm)
        boxes = [b for b in boxes
                 if abs(b.center_mm[0]) < r and abs(b.center_mm[1]) < r]
        for i, b in enumerate(boxes):
            b.id = i

    top_d = round(float(top.depth_mm), 1)
    if not boxes:
        return Decision(status="LAYER_EMPTY", reason="no box on top layer",
                        valid_frac=valid_frac, top_depth_mm=top_d, thresholds=thr,
                        latency_ms=(time.perf_counter() - t0) * 1e3)

    pickable = [b for b in boxes
                if b.confidence >= th.min_confidence and b.tilt_deg <= th.max_tilt_deg]
    best = max((b.confidence for b in boxes), default=0.0)
    if len(pickable) < th.min_boxes_for_ok:
        return Decision(status="LOW_CONFIDENCE",
                        reason=(f"best confidence {best:.2f} < {th.min_confidence:.2f} "
                                f"or tilt > {th.max_tilt_deg}deg"),
                        n_boxes=len(boxes), n_pickable=0, valid_frac=valid_frac,
                        top_depth_mm=top_d, best_confidence=round(best, 3), boxes=boxes,
                        thresholds=thr, latency_ms=(time.perf_counter() - t0) * 1e3)

    plan = pick_plan(pickable, dest_xy_mm=dest_xy_mm, col_tol_mm=col_tol_mm)
    return Decision(status="OK", reason="ready", n_boxes=len(boxes), n_pickable=len(pickable),
                    valid_frac=valid_frac, top_depth_mm=top_d, best_confidence=round(best, 3),
                    boxes=boxes, plan=plan, thresholds=thr,
                    latency_ms=(time.perf_counter() - t0) * 1e3)


class HealthMonitor:
    """정적 배경 기준 드리프트 감시 — 카메라 이동·오염·조명 변화 탐지.

    셀의 고정 구조물(컨베이어·프레임)은 매 프레임 같은 깊이여야 한다. 기준 프레임에서
    안정 픽셀을 고르고, 이후 프레임에서 그 픽셀들의 깊이 편차 중앙값을 본다.
    캘리브레이션이 틀어지면 픽 위치가 통째로 밀리므로, 검출 성공률만 봐서는 늦는다.
    """

    def __init__(self, drift_warn_mm: float = 8.0, drift_fail_mm: float = 25.0,
                 roi_frac: float = 0.25):
        self.ref_D: Optional[np.ndarray] = None
        self.ref_mask: Optional[np.ndarray] = None
        self.drift_warn_mm = drift_warn_mm
        self.drift_fail_mm = drift_fail_mm
        self.roi_frac = roi_frac

    def set_reference(self, frame: Frame):
        """기준 프레임 등록 — 중앙 ROI(팔레트) **밖**의 유효 픽셀만 정적 배경으로 본다."""
        h, w = frame.D.shape
        outside = np.ones((h, w), bool)
        outside[int(h * self.roi_frac):int(h * (1 - self.roi_frac)),
                int(w * self.roi_frac):int(w * (1 - self.roi_frac))] = False
        self.ref_mask = outside & frame.valid
        self.ref_D = frame.D.copy()
        return int(self.ref_mask.sum())

    def check(self, frame: Frame) -> dict:
        if self.ref_D is None:
            return {"status": "NO_REFERENCE", "drift_mm": None, "n_px": 0}
        m = self.ref_mask & frame.valid
        n = int(m.sum())
        if n < 500:
            return {"status": "INSUFFICIENT", "drift_mm": None, "n_px": n}
        d = np.abs(frame.D[m] - self.ref_D[m])
        drift = float(np.median(d))
        status = ("OK" if drift < self.drift_warn_mm
                  else "WARN" if drift < self.drift_fail_mm else "FAIL")
        return {"status": status, "drift_mm": round(drift, 2),
                "p95_mm": round(float(np.percentile(d, 95)), 2), "n_px": n}
