# -*- coding: utf-8 -*-
"""폐루프 결과 차트: mocap 석션 EE vs UR10e(고정 받침대) vs UR10e(리니어 트랙), 각각 oracle / 인식 구동.

입력: explore/twin/closed_loop.json, closed_loop_arm_fixed.json, closed_loop_arm_track.json
출력: assets/twin_closed_loop.png  (왼쪽: 성공률, 가운데: 실패 종류, 오른쪽: 사이클 시간)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = [("mocap 석션\n원래 배치", ROOT / "explore/twin/closed_loop.json"),
       ("mocap 석션\n팔 자리 비움", ROOT / "explore/twin/closed_loop_mocap_armlayout.json"),
       ("UR10e\n고정 받침대", ROOT / "explore/twin/closed_loop_arm_fixed.json"),
       ("UR10e\n트랙 ±0.6 m", ROOT / "explore/twin/closed_loop_arm_track.json"),
       ("mocap 석션\n+ 층 사전", ROOT / "explore/twin/closed_loop_prioronly.json"),
       ("UR10e 트랙\n+ 층 사전", ROOT / "explore/twin/closed_loop_arm_track_prioronly.json")]
OUT = ROOT / "assets" / "twin_closed_loop.png"

for cand in (r"C:\Windows\Fonts\malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
    if Path(cand).exists():
        fm.fontManager.addfont(cand)
        plt.rcParams["font.family"] = fm.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

KO = {"placed": "배치 성공", "misplaced": "놓은 위치 어긋남", "grasp_miss": "흡착 실패", "source_empty": "소스 미검출(층 선택)",
      "rejected_footprint": "컵 아래 결손으로 거부", "ik_unreachable": "도달 불가", "path_collision": "경로 충돌",
      "descent_path_collision": "하강 경로 충돌", "descent_ik_unreachable": "하강 도달 불가", "only_unreachable_left": "남은 박스 도달 불가",
      "place_ik_unreachable": "목적지 도달 불가", "place_path_collision": "목적지 경로 충돌",
      "place_descent_path_collision": "목적지 하강 경로 충돌", "place_descent_ik_unreachable": "목적지 하강 도달 불가"}


def main():
    data = []
    for name, p in SRC:
        if p.exists():
            data.append((name, json.loads(p.read_text(encoding="utf-8"))))
    if not data:
        raise SystemExit("no closed-loop json found")
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.8), gridspec_kw=dict(width_ratios=[1.2, 1.6, 1.0]))
    # (a) 성공률
    ax = axes[0]
    x = np.arange(len(data))
    w = 0.36
    orc = [d["oracle"]["success_rate"] * 100 for _, d in data]
    per = [d["perception"]["success_rate"] * 100 for _, d in data]
    b1 = ax.bar(x - w / 2, orc, w, label="정답 위치(oracle)", color="#4c72b0")
    b2 = ax.bar(x + w / 2, per, w, label="인식 결과로 구동", color="#dd8452")
    for b, v in list(zip(b1, orc)) + list(zip(b2, per)):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in data], fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_ylabel("배치 성공률 %")
    ax.set_title(f"박스 {data[0][1]['perception']['boxes']}개 × 실행기 {len(data)}종", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    # (b) 인식 구동 실패 종류 (실행기별 누적 막대)
    ax = axes[1]
    keys = []
    for _, d in data:
        for k in d["perception"]["outcomes"]:
            if k != "placed" and k not in keys:
                keys.append(k)
    bottom = np.zeros(len(data))
    cmap = plt.get_cmap("tab20")
    for i, k in enumerate(keys):
        vals = np.array([d["perception"]["outcomes"].get(k, 0) for _, d in data], float)
        ax.bar(x, vals, 0.55, bottom=bottom, label=KO.get(k, k), color=cmap(i % 20))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in data], fontsize=8)
    ax.set_ylabel("실패 건수 (인식 구동)")
    ax.set_title("인식 구동 실패 종류 — 어디서 잃는가", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    # (c) 사이클 시간
    ax = axes[2]
    means = [d["oracle"]["cycle_s"]["mean"] if d["oracle"]["cycle_s"] else 0 for _, d in data]
    p95 = [d["oracle"]["cycle_s"]["p95"] if d["oracle"]["cycle_s"] else 0 for _, d in data]
    bars = ax.bar(x, means, 0.5, color="#55a868", yerr=[np.zeros(len(data)), np.array(p95) - np.array(means)], capsize=4)
    for b, v, q in zip(bars, means, p95):
        ax.text(b.get_x() + b.get_width() / 2, q + 0.3, f"{v:.1f} s\n(p95 {q:.1f})", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in data], fontsize=8)
    ax.set_ylim(0, max(p95) * 1.35 if p95 else 1)
    ax.set_ylabel("픽 1회 사이클 (s, oracle)")
    ax.set_title("사이클 시간: mocap 속도 가정 vs 관절 속도 한계", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("saved", OUT)


if __name__ == "__main__":
    main()
