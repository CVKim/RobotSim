# -*- coding: utf-8 -*-
"""SDG 툴 비교: BlenderProc(noisy) vs Isaac Replicator(noisy) — 동일 실측 홀드아웃(24장) mAP50.
BlenderProc 모델은 기존 학습 결과(train_noisy/best.pt) 재사용, Isaac은 동일 레시피로 학습.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(r"E:\Robot_Sim\explore")
OUT = ROOT / "sim2real"
HOLD = ROOT / "real_holdout" / "data.yaml"


def main():
    res = {}
    m_bp = YOLO(str(OUT / "train_noisy" / "weights" / "best.pt"))
    res["blenderproc_noisy"] = round(float(m_bp.val(data=str(HOLD), split="val", device=0,
                                                    verbose=False).box.map50), 4)
    m_is = YOLO("yolo11n.pt")
    m_is.train(data=str(ROOT / "isaac_yolo_noisy" / "data.yaml"), epochs=60, imgsz=640,
               device=0, batch=16, project=str(OUT), name="train_isaac_noisy",
               verbose=False, plots=False, workers=0)
    res["isaac_noisy"] = round(float(m_is.val(data=str(HOLD), split="val", device=0,
                                              verbose=False).box.map50), 4)
    (OUT / "tool_compare.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(res)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=130)
    names = ["BlenderProc\n(physics drop)", "Isaac Replicator\n(static placement)"]
    vals = [res["blenderproc_noisy"], res["isaac_noisy"]]
    bars = ax.bar(names, vals, color=["#d97706", "#15803d"], width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("mAP@50 on REAL ToF holdout (24)")
    ax.set_ylim(0, 1.05)
    ax.set_title("SDG tool comparison, same recipe (sensor-noise sim, no real frames)")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "tool_compare.png")


if __name__ == "__main__":
    main()
