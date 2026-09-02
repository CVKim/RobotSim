# -*- coding: utf-8 -*-
"""PushT 체크포인트 스윕 결과 → 학습량-성공률 곡선."""
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []
for f in sorted(glob.glob(r"E:\Robot_Sim\runs\pusht_eval\*\eval_info.json")):
    step = Path(f).parent.name
    d = json.load(open(f))
    m = d["per_task"][0]["metrics"]
    succ = 100.0 * sum(m["successes"]) / len(m["successes"])
    avg_max = sum(m["max_rewards"]) / len(m["max_rewards"])
    rows.append((step, succ, avg_max))

for r in rows:
    print(r)

named = [(s, a, b) for s, a, b in rows if s != "last" and a is not None]
xs = [int(s) // 1000 for s, _, _ in named]
sr = [v for _, v, _ in named]
mr = [v * 100 for _, _, v in named]

fig, ax = plt.subplots(figsize=(8, 4.4), dpi=130)
ax.plot(xs, sr, "o-", color="#d97706", lw=2, label="success rate (%)")
ax2 = ax.twinx()
ax2.plot(xs, mr, "s--", color="#1d4ed8", alpha=0.7, label="avg max coverage (%)")
ax.set_xlabel("training steps (k)")
ax.set_ylabel("success rate (%) - 50 episodes")
ax2.set_ylabel("avg max coverage (%)")
ax.set_title("PushT Diffusion Policy 263M: checkpoint sweep (RTX 3080 10GB)")
l1, la1 = ax.get_legend_handles_labels()
l2, la2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, la1 + la2, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(r"E:\Robot_Sim\explore\pusht_sweep_curve.png")
print("chart saved")
