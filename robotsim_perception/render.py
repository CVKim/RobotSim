# -*- coding: utf-8 -*-
"""검출·픽 계획 오버레이 PNG (depth 컬러맵 + 최상층 틴트 + 박스 + 픽 순서)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .frame import Frame


def _dest_px(frame: Frame, dest_xy_mm, max_dist_mm: float = 100.0):
    """목적지 mm 좌표에 가장 가까운 유효 픽셀 (거리 max_dist_mm 이내일 때만)."""
    if frame.n_valid == 0:
        return None
    d2 = (frame.X - dest_xy_mm[0]) ** 2 + (frame.Y - dest_xy_mm[1]) ** 2
    d2 = np.where(frame.valid, d2, np.inf)
    k = int(np.argmin(d2))
    if not np.isfinite(d2.flat[k]) or d2.flat[k] > max_dist_mm ** 2:
        return None
    v, u = divmod(k, frame.shape[1])
    return (int(u), int(v))


def render_overlay(frame: Frame, result, out_path=None) -> np.ndarray:
    """Result → BGR 이미지. out_path 를 주면 저장(부모 폴더 생성)."""
    D, valid = frame.D, frame.valid
    img = np.full((*frame.shape, 3), 20, dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(D[valid], [2, 98])
        norm = np.clip((D - lo) / max(hi - lo, 1), 0, 1)
        img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_BONE)
        img[~valid] = (20, 20, 20)
    mask = result.top_layer.mask
    if mask.any():
        img[mask] = (0.55 * img[mask] + 0.45 * np.array([0, 140, 255])).astype(np.uint8)

    order = {s.box_id: s.order for s in result.plan}
    for b in result.boxes:
        pts = b.corners_px().astype(np.int32)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        cx, cy = b.center_px
        cv2.drawMarker(img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
        cv2.putText(img, f"{b.id}", (cx - 8, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, f"{b.tilt_deg:.1f}d c{b.confidence:.2f}", (cx - 24, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)
        if b.id in order:
            cv2.putText(img, f"P{order[b.id]}", (cx - 14, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (60, 220, 255), 2)

    dest_px = _dest_px(frame, result.dest_xy_mm)
    if dest_px is not None:
        cv2.drawMarker(img, dest_px, (60, 220, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.putText(img, "DEST", (dest_px[0] - 20, dest_px[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (60, 220, 255), 1)
    if result.plan:
        first = next(b for b in result.boxes if b.id == result.plan[0].box_id)
        px = first.center_px
        cv2.circle(img, px, 26, (60, 220, 255), 2)
        if dest_px is not None:
            cv2.arrowedLine(img, px, dest_px, (60, 220, 255), 2, tipLength=0.06)

    top = result.top_layer
    top_txt = f"{top.depth_mm:.0f}mm" if top.ok else "n/a"
    cv2.putText(img, f"top layer @ {top_txt}, {len(result.boxes)} boxes, {result.latency_ms:.0f} ms",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(out_path.suffix or ".png", img)
        if not ok:
            raise RuntimeError(f"encode failed: {out_path}")
        buf.tofile(str(out_path))  # 비ASCII 경로 대응
    return img
