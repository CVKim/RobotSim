# -*- coding: utf-8 -*-
"""빈피킹 v2: 박스 상면 픽포인트(중심+법선) + 픽킹 진행 슬롯 추적.

- 픽포인트: 각 박스 상면 포인트(X,Y,D)에 평면 최소자승 피팅 → 중심 + 법선.
  석션 그리퍼 접근 벡터 = -법선. 법선의 수직축 대비 기울기(deg)도 산출(틀어진 박스 감지).
- 슬롯 추적: 같은 층의 '가장 찬 세션'을 기준 레이아웃으로 삼아, 각 세션의 박스를
  중심 거리로 슬롯에 매칭 → 세션별 점유/빈 슬롯 → 픽킹 순서 복원.
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from binpick_topface import detect_boxes, render_overlay  # noqa: E402
from mim_loader import load_session  # noqa: E402

try:
    from local_paths import BINPICK_DIR
except ImportError:
    BINPICK_DIR = os.environ["BINPICK_DIR"]

OUT = Path(r"E:\Robot_Sim\explore\pickpoints")
OUT.mkdir(parents=True, exist_ok=True)


def fit_plane(pts):
    """pts (N,3) mm → (centroid, unit normal, rms residual mm)."""
    c = pts.mean(axis=0)
    q = pts - c
    _, s, vt = np.linalg.svd(q, full_matrices=False)
    n = vt[2]
    if n[2] > 0:  # 카메라 쪽(-Z)을 향하도록 통일 (D축이 카메라에서 멀어지는 방향)
        n = -n
    rms = float(np.sqrt(np.mean((q @ n) ** 2)))
    return c, n, rms


def analyze_session(sess_dir):
    sess = load_session(sess_dir)
    top_d, mask, boxes = detect_boxes(sess)
    X, Y, D = sess["X"], sess["Y"], sess["D"]
    picks = []
    for b in boxes:
        pts_px = cv2.boxPoints(b["rect_px"]).astype(np.int32)
        comp = np.zeros(D.shape, dtype=np.uint8)
        cv2.fillPoly(comp, [pts_px], 1)
        m = (comp > 0) & (np.abs(D - b["depth_mm"]) < 40)
        if m.sum() < 100:
            continue
        pts = np.stack([X[m], Y[m], D[m]], axis=-1)
        c, n, rms = fit_plane(pts)
        tilt = float(np.degrees(np.arccos(min(abs(n[2]), 1.0))))
        picks.append({
            "center_mm": [round(float(v), 1) for v in c],
            "normal": [round(float(v), 4) for v in n],
            "tilt_deg": round(tilt, 2),
            "plane_rms_mm": round(rms, 2),
            "dims_mm": b["dims_mm"],
            "center_px": [int(pts_px[:, 0].mean()), int(pts_px[:, 1].mean())],
        })
    return sess, top_d, mask, boxes, picks


def render_picks(sess, top_d, mask, boxes, picks, path):
    render_overlay(sess, top_d, mask, boxes, path)  # 기본 오버레이 먼저
    img = cv2.imread(str(path))
    for p in picks:
        cx, cy = p["center_px"]
        cv2.drawMarker(img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(img, f'{p["tilt_deg"]:.1f}d', (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)
    cv2.imwrite(str(path), img)


def main():
    root = Path(BINPICK_DIR)
    sessions = sorted(d for d in root.iterdir() if d.is_dir())
    all_res = {}
    for s in sessions:
        sess, top_d, mask, boxes, picks = analyze_session(s)
        all_res[s.name] = {"top_layer_mm": round(float(top_d), 1), "picks": picks}
        if picks:
            render_picks(sess, top_d, mask, boxes, picks, OUT / f"{s.name}_picks.png")

    # --- 슬롯 추적: 층별로 가장 박스 많은 세션을 기준 레이아웃으로 ---
    def layer_key(v):
        return int(round(v["top_layer_mm"] / 100.0))

    layers = {}
    for name, v in all_res.items():
        if v["picks"]:
            layers.setdefault(layer_key(v), []).append(name)

    slot_report = {}
    for lk, names in layers.items():
        ref_name = max(names, key=lambda n: len(all_res[n]["picks"]))
        ref = [p["center_mm"][:2] for p in all_res[ref_name]["picks"]]
        for name in sorted(names):
            occ = []
            for slot_i, rc in enumerate(ref):
                d = min(
                    (np.hypot(p["center_mm"][0] - rc[0], p["center_mm"][1] - rc[1])
                     for p in all_res[name]["picks"]), default=1e9)
            # 슬롯 점유 = 기준 중심 150mm 이내에 박스 존재
                occ.append(bool(d < 150))
            slot_report[name] = {"layer": lk, "ref": ref_name,
                                 "occupied": occ, "n_empty": occ.count(False)}

    json.dump({"sessions": all_res, "slots": slot_report},
              open(OUT / "pickpoints.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)

    tilts = [p["tilt_deg"] for v in all_res.values() for p in v["picks"]]
    rms = [p["plane_rms_mm"] for v in all_res.values() for p in v["picks"]]
    print("picks total:", len(tilts))
    print("tilt deg: mean %.2f max %.2f" % (np.mean(tilts), np.max(tilts)))
    print("plane rms mm: mean %.2f p95 %.2f" % (np.mean(rms), np.percentile(rms, 95)))
    empty_seq = [(n, v["n_empty"]) for n, v in sorted(slot_report.items())]
    print("empty-slot progression:", empty_seq[:12])


if __name__ == "__main__":
    main()
