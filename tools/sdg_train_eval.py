# -*- coding: utf-8 -*-
"""T5 sim2real 갭 측정: 합성(clean/noisy) 학습 → 합성 val + 실측 val 평가.

산출: explore/sim2real/gap.json + gap_chart.png
  gap = 합성 val mAP − 실측 val mAP  (노이즈 주입이 갭을 줄이는지 ablation)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(r"E:\Robot_Sim\explore")
OUT = ROOT / "sim2real"
OUT.mkdir(parents=True, exist_ok=True)

results = {}
for variant in ["clean", "noisy"]:
    data = ROOT / f"sdg_yolo_{variant}" / "data.yaml"
    model = YOLO("yolo11n.pt")
    model.train(data=str(data), epochs=60, imgsz=640, device=0, batch=16,
                project=str(OUT), name=f"train_{variant}", verbose=False,
                plots=False, workers=2)
    syn = model.val(data=str(data), split="val", device=0, verbose=False)
    real = model.val(data=str(ROOT / "real_eval" / "data.yaml"), split="val",
                     device=0, verbose=False)
    results[variant] = {
        "synth_map50": round(float(syn.box.map50), 4),
        "synth_map5095": round(float(syn.box.map), 4),
        "real_map50": round(float(real.box.map50), 4),
        "real_map5095": round(float(real.box.map), 4),
        "gap_map50": round(float(syn.box.map50 - real.box.map50), 4),
    }
    print(variant, results[variant])

(OUT / "gap.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

fig, ax = plt.subplots(figsize=(7, 4.4), dpi=130)
x = range(2)
labels = ["clean synth", "noisy synth\n(measured ToF traits)"]
syn_v = [results[v]["synth_map50"] for v in ["clean", "noisy"]]
real_v = [results[v]["real_map50"] for v in ["clean", "noisy"]]
ax.bar([i - 0.18 for i in x], syn_v, 0.36, label="synthetic val", color="#93c5fd")
ax.bar([i + 0.18 for i in x], real_v, 0.36, label="REAL ToF val", color="#d97706")
for i, v in enumerate(["clean", "noisy"]):
    ax.text(i + 0.18, real_v[i] + 0.02, f'{real_v[i]:.2f}', ha="center", fontsize=11, fontweight="bold")
    ax.text(i, max(syn_v[i], real_v[i]) + 0.09, f'gap {results[v]["gap_map50"]:+.2f}',
            ha="center", fontsize=10, color="#374151")
ax.set_xticks(list(x)), ax.set_xticklabels(labels)
ax.set_ylabel("mAP@50 (box detection)"), ax.set_ylim(0, 1.15)
ax.set_title("sim2real gap: synthetic-only training evaluated on REAL factory ToF")
ax.grid(alpha=0.25, axis="y"), ax.legend()
fig.tight_layout(), fig.savefig(OUT / "gap_chart.png")
print("saved:", OUT / "gap_chart.png")
