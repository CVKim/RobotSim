# -*- coding: utf-8 -*-
"""팔레타이징 RL 다중 시드·ablation 집계 -> results/palletize_multiseed.json + 차트.

기존 보고(64.9박스 / 활용률 68.5%)는 시드 미설정 단일 런이라 학습 재현과 런 간 분산을 알 수 없었다.
여기서는 시드 3개 x (action mask 유/무)를 모아 평균 +- 표준편차와 DBL 휴리스틱 대비를 함께 낸다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "palletize_seeds"
DBL = {"boxes_mean": 56.6, "boxes_std": 4.1, "utilization_mean": 0.598}


def collect(prefix):
    rows = []
    for d in sorted(RUNS.glob(f"{prefix}_s*")):
        f = d / "eval.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        rows.append(r)
    return rows


def stat(rows, key):
    v = np.array([r[key] for r in rows], float)
    return {"mean": round(float(v.mean()), 3), "std": round(float(v.std(ddof=1)), 3) if len(v) > 1 else 0.0,
            "n": len(v), "values": [round(float(x), 3) for x in v]}


def main():
    out = {"baseline_dbl": DBL, "variants": {}}
    for prefix, label in (("mask", "MaskablePPO (action mask)"), ("nomask", "PPO (mask 제거 ablation)")):
        rows = collect(prefix)
        if not rows:
            print(f"  {prefix}: 결과 없음 (학습 미완료)")
            continue
        out["variants"][prefix] = {
            "label": label,
            "boxes": stat(rows, "ppo_boxes_mean"),
            "utilization": stat(rows, "ppo_utilization_mean"),
        }
        b = out["variants"][prefix]["boxes"]
        u = out["variants"][prefix]["utilization"]
        print(f"  {label}: {b['mean']:.1f} +- {b['std']:.1f} 박스 (n={b['n']}), "
              f"활용률 {u['mean']:.3f} +- {u['std']:.3f}")

    if "mask" in out["variants"]:
        m = out["variants"]["mask"]["boxes"]["mean"]
        out["vs_dbl_boxes_pct"] = round(100 * (m - DBL["boxes_mean"]) / DBL["boxes_mean"], 1)
        print(f"  DBL 대비: {out['vs_dbl_boxes_pct']:+.1f}%")
    if "mask" in out["variants"] and "nomask" in out["variants"]:
        a = out["variants"]["mask"]["boxes"]["mean"]
        b = out["variants"]["nomask"]["boxes"]["mean"]
        out["action_mask_effect_boxes"] = round(a - b, 2)
        print(f"  action mask 효과: {a - b:+.2f} 박스")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "palletize_multiseed.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels, means, stds = ["DBL heuristic"], [DBL["boxes_mean"]], [DBL["boxes_std"]]
        for prefix, name in (("mask", "PPO + action mask"), ("nomask", "PPO no mask")):
            if prefix in out["variants"]:
                labels.append(name)
                means.append(out["variants"][prefix]["boxes"]["mean"])
                stds.append(out["variants"][prefix]["boxes"]["std"])
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
        cols = ["#9ca3af", "#2563eb", "#d97706"][:len(means)]
        ax.bar(labels, means, yerr=stds, capsize=5, color=cols, width=0.55)
        for i, (m_, s_) in enumerate(zip(means, stds)):
            ax.text(i, m_ + s_ + 0.8, f"{m_:.1f}", ha="center", fontsize=11, fontweight="bold")
        ax.set_ylabel("boxes placed (100 eval episodes)")
        ax.set_title("Palletizing RL: multi-seed mean ± sd (3 seeds each)")
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(ROOT / "assets" / "palletize_multiseed.png")
        print("saved chart")
    except Exception as e:
        print("chart skipped:", e)
    print("saved", ROOT / "results" / "palletize_multiseed.json")


if __name__ == "__main__":
    main()
