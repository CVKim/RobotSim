# -*- coding: utf-8 -*-
"""실용성 검증 1: 기하 인식 파이프라인의 (a) 지연시간 (b) 열화 강건성.

(b) 실측 30프레임에 교란을 단계적으로 주입하고 검출 수/치수 안정성을 기준선(148박스)과 비교:
    - 추가 무효 픽셀(스펙클) 5/15/30%
    - 깊이 가우시안 노이즈 σ 5/10/20mm
    - 강도 채널 대비 저하(에지 약화) x0.5/x0.25
    - 무효 블롭(대면적 결손) 5/15개
산출: explore/robustness/robustness.json + degradation.png
"""
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from binpick_topface import detect_boxes  # noqa: E402
from mim_loader import SENTINEL_D, load_session  # noqa: E402

try:
    from local_paths import BINPICK_DIR
except ImportError:
    import os
    BINPICK_DIR = os.environ["BINPICK_DIR"]

OUT = Path(r"E:\Robot_Sim\explore\robustness")
OUT.mkdir(parents=True, exist_ok=True)


def perturb(sess, kind, level, rng):
    s = {k: v.copy() for k, v in sess.items() if k != "rgb"}
    D, I = s["D"], s["I"]
    h, w = D.shape
    if kind == "speckle":          # 추가 무효 픽셀 비율
        m = rng.random(D.shape) < level
        D[m] = SENTINEL_D + 0.75
    elif kind == "gauss":          # 깊이 가우시안 노이즈 (mm)
        valid = (D > 0) & (D < SENTINEL_D)
        D[valid] += rng.normal(0, level, valid.sum())
    elif kind == "contrast":       # 강도 대비 저하 (에지 약화)
        s["I"] = (I * level).astype(I.dtype)
    elif kind == "blobs":          # 대면적 결손
        blob = np.zeros((h, w), np.uint8)
        for _ in range(int(level)):
            cv2.ellipse(blob, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                        (int(rng.integers(10, 60)), int(rng.integers(10, 60))),
                        float(rng.uniform(0, 180)), 0, 360, 1, -1)
        D[blob > 0] = SENTINEL_D + 0.75
    return s


def count_boxes(sess):
    _, _, boxes = detect_boxes(sess)
    dims = np.array([b["dims_mm"] for b in boxes]) if boxes else np.zeros((0, 2))
    return len(boxes), dims


def main():
    sessions = sorted(d for d in Path(BINPICK_DIR).iterdir() if d.is_dir())
    loaded = [load_session(s) for s in sessions]

    # (a) 지연시간
    t0 = time.perf_counter()
    base_counts, base_dims = [], []
    for sess in loaded:
        n, dims = count_boxes(sess)
        base_counts.append(n)
        base_dims.append(dims)
    lat_ms = (time.perf_counter() - t0) / len(loaded) * 1000
    base_total = int(sum(base_counts))
    base_L = np.concatenate([d[:, 0] for d in base_dims if len(d)]).mean()
    print(f"latency: {lat_ms:.1f} ms/frame (CPU, incl. all stages) | baseline boxes={base_total} L={base_L:.1f}mm")

    # (b) 열화
    grid = {"speckle": [0.05, 0.15, 0.30], "gauss": [5, 10, 20],
            "contrast": [0.5, 0.25, 0.1], "blobs": [5, 15, 30]}
    rng = np.random.default_rng(0)
    res = {"latency_ms_per_frame": round(lat_ms, 1), "baseline_boxes": base_total,
           "baseline_L_mm": round(float(base_L), 1), "degradation": {}}
    for kind, levels in grid.items():
        res["degradation"][kind] = []
        for lv in levels:
            tot, Ls = 0, []
            for sess in loaded:
                n, dims = count_boxes(perturb(sess, kind, lv, rng))
                tot += n
                if len(dims):
                    Ls.append(dims[:, 0].mean())
            rec = {"level": lv, "boxes": tot, "recall_vs_base": round(tot / base_total, 3),
                   "L_mm": round(float(np.mean(Ls)), 1) if Ls else None}
            res["degradation"][kind].append(rec)
            print(f"  {kind}={lv}: boxes={tot} ({rec['recall_vs_base']:.0%}), L={rec['L_mm']}")

    (OUT / "robustness.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), dpi=130)
    titles = {"speckle": "extra invalid px (%)", "gauss": "depth noise sigma (mm)",
              "contrast": "intensity contrast (x)", "blobs": "invalid blobs (#)"}
    for ax, (kind, recs) in zip(axes, res["degradation"].items()):
        xs = [r["level"] * (100 if kind == "speckle" else 1) for r in recs]
        ys = [r["recall_vs_base"] * 100 for r in recs]
        ax.plot(xs, ys, "o-", color="#d97706", lw=2)
        ax.axhline(100, color="#9ca3af", ls="--", lw=1)
        ax.set_ylim(0, 110)
        ax.set_xlabel(titles[kind])
        ax.set_ylabel("detected boxes vs baseline (%)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Geometric detector robustness on 30 real frames (baseline {base_total} boxes, {lat_ms:.0f} ms/frame CPU)")
    fig.tight_layout()
    fig.savefig(OUT / "degradation.png")
    print("saved")


if __name__ == "__main__":
    main()
