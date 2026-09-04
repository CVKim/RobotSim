# -*- coding: utf-8 -*-
"""셀 트윈 절대 정답(GT) 기반 검출기 평가 — pseudo-GT 순환 참조를 끊는다.

실측 30프레임 평가의 근본 한계: 정답이 검출기 자신의 출력(RGB 육안 대조)이라
위치·치수 오차가 정의상 0 이고 회귀를 잡을 수 없었다.
셀 트윈에서는 박스의 실제 위치·치수·자세를 mjData 에서 직접 읽으므로
검출 리콜뿐 아니라 **중심 오차(mm)·치수 오차(mm)·각도 오차(deg)** 를 절대 기준으로 측정한다.

레이아웃: 층당 4x3, 층 수·잔여 개수를 무작위로 바꿔 디팔레타이징 진행 상황을 재현.
시드마다 배치 지터(±4mm, ±1.2deg)와 노이즈 인스턴스가 달라진다.

산출: explore/twin/detect_eval.json + assets/twin_detect_accuracy.png
실행:  .venv\\Scripts\\python.exe tools/twin_detect_eval.py --scenes 24
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "tools"))

from binpick_topface import detect_boxes_v2  # noqa: E402
from cell_scene import N_COL, N_ROW, build_xml  # noqa: E402
from cell_twin import TwinRenderer, ground_truth, match, settle  # noqa: E402

OUT = ROOT / "explore" / "twin"


def make_layout(rng, n_top=None, n_layers=None):
    """실제 디팔레타이징 상태 재현: 아래층은 가득(4x3), 최상층만 n_top 개 남은 형태.

    실측 시퀀스가 층당 12 -> 3 개까지 줄고 층이 비면 다음 층으로 내려가는 것과 같은 구성.
    n_top 을 1 까지 낮추면 '가득 찬 아래층 위에 1개만 남은' 극단 케이스를 만들 수 있다.
    """
    per = N_COL * N_ROW
    n_layers = int(rng.integers(1, 3)) if n_layers is None else n_layers
    n_top = int(rng.integers(2, per + 1)) if n_top is None else n_top
    cells = [(c, r) for r in range(N_ROW) for c in range(N_COL)]
    out = []
    for layer in range(n_layers - 1):
        out += [(c, r, layer) for (c, r) in cells]
    idx = sorted(rng.permutation(per)[:n_top])
    out += [(cells[i][0], cells[i][1], n_layers - 1) for i in idx]
    return out, n_top, n_layers


def boot_ci(x, n=4000, seed=0):
    """부트스트랩 95% 신뢰구간 (평균)."""
    if len(x) == 0:
        return (None, None)
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(x), size=(n, len(x)))
    means = np.asarray(x)[idx].mean(axis=1)
    return (round(float(np.percentile(means, 2.5)), 2), round(float(np.percentile(means, 97.5)), 2))


def run(n_scenes, noise, seed0=0):
    import mujoco
    rows, per_scene = [], []
    for s in range(n_scenes):
        rng = np.random.default_rng(1000 + seed0 + s)
        # 잔여 개수를 1..12 로 고르게 훑어 '아래층 가득 + 상층 소수' 극단 케이스까지 포함
        layout, n_top, n_layers = make_layout(rng, n_top=1 + (s % 12), n_layers=1 + (s // 12) % 2)
        xml, _ = build_xml(layout, seed=1000 + seed0 + s)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        settle(m, d, 1200)
        r = TwinRenderer(m)
        try:
            f = r.frame(d, noise=noise, rng=np.random.default_rng(2000 + seed0 + s))
            gt = ground_truth(m, d)
            _, _, det = detect_boxes_v2(f)
            pairs, missed, fp = match(gt, det)
            for g, b, dist in pairs:
                rows.append({
                    "scene": s, "center_err_mm": float(dist),
                    "L_err_mm": float(b["dims_mm"][0] - g["dims_mm"][0]),
                    "W_err_mm": float(b["dims_mm"][1] - g["dims_mm"][1]),
                    "ang_err_deg": float(abs((b["ang_deg"] - g["ang_deg"] + 90) % 180 - 90)),
                    "depth_err_mm": float(b["depth_mm"] - g["top_d_mm"]),
                    "source": b.get("source", "detected"),
                    "confidence": float(b.get("confidence", 0.0)),
                })
            per_scene.append({"scene": s, "n_top": n_top, "n_layers": n_layers,
                              "n_gt": len(gt), "n_det": len(det),
                              "matched": len(pairs), "missed": len(missed), "false_pos": len(fp),
                              "valid_frac": round(float((f["D"] < 16000).mean()), 3)})
            print(f"  scene {s:2d}: top={n_top:2d}/L{n_layers} GT {len(gt):2d} det {len(det):2d} "
                  f"matched {len(pairs):2d} miss {len(missed)} FP {len(fp)} "
                  f"valid {100*per_scene[-1]['valid_frac']:.0f}%")
        finally:
            r.close()
    return rows, per_scene


def summarize(rows, per_scene, noise):
    n_gt = sum(p["n_gt"] for p in per_scene)
    n_ok = sum(p["matched"] for p in per_scene)
    n_fp = sum(p["false_pos"] for p in per_scene)
    res = {"noise": noise or "clean", "scenes": len(per_scene), "gt_boxes": n_gt,
           "matched": n_ok, "false_pos": n_fp,
           "recall": round(n_ok / max(n_gt, 1), 4),
           "precision": round(n_ok / max(n_ok + n_fp, 1), 4)}
    for k, unit in (("center_err_mm", "mm"), ("L_err_mm", "mm"), ("W_err_mm", "mm"),
                    ("ang_err_deg", "deg"), ("depth_err_mm", "mm")):
        v = np.array([r[k] for r in rows], float)
        if len(v) == 0:
            continue
        res[k] = {"mean": round(float(v.mean()), 2), "std": round(float(v.std()), 2),
                  "p50": round(float(np.percentile(v, 50)), 2),
                  "p95": round(float(np.percentile(v, 95)), 2),
                  "max": round(float(np.abs(v).max()), 2),
                  "ci95_mean": boot_ci(v)}
    # 잔여 개수별 분해 — 트윈이 드러낸 핵심 실패 모드:
    # 가득 찬 아래층 위에 박스가 몇 개 안 남으면 층 선택기가 아래층으로 점프해 FP 가 폭증한다.
    by_top = {}
    for p in per_scene:
        b = by_top.setdefault(p["n_top"], {"scenes": 0, "gt": 0, "matched": 0, "fp": 0})
        b["scenes"] += 1
        b["gt"] += p["n_gt"]
        b["matched"] += p["matched"]
        b["fp"] += p["false_pos"]
    for k, b in by_top.items():
        b["recall"] = round(b["matched"] / max(b["gt"], 1), 3)
        b["precision"] = round(b["matched"] / max(b["matched"] + b["fp"], 1), 3)
    res["by_boxes_remaining"] = {str(k): by_top[k] for k in sorted(by_top)}
    few = [p for p in per_scene if p["n_top"] <= 3]
    many = [p for p in per_scene if p["n_top"] >= 4]
    for tag, grp in (("remaining_le3", few), ("remaining_ge4", many)):
        g = sum(p["n_gt"] for p in grp)
        mt = sum(p["matched"] for p in grp)
        fp = sum(p["false_pos"] for p in grp)
        res[tag] = {"scenes": len(grp), "gt": g, "matched": mt, "false_pos": fp,
                    "recall": round(mt / max(g, 1), 3),
                    "precision": round(mt / max(mt + fp, 1), 3)}
    inf = [r for r in rows if r["source"] == "inferred"]
    if inf:
        res["inferred_center_err_mm"] = {
            "n": len(inf),
            "mean": round(float(np.mean([r["center_err_mm"] for r in inf])), 2),
            "p95": round(float(np.percentile([r["center_err_mm"] for r in inf], 95)), 2)}
    return res


def chart(all_res, rows_by_noise, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=130)
    colors = {"clean": "#2563eb", "tof": "#d97706"}
    ax = axes[0]
    for k, rows in rows_by_noise.items():
        v = [r["center_err_mm"] for r in rows]
        if v:
            ax.hist(v, bins=28, alpha=0.6, label=f"{k} (n={len(v)})", color=colors.get(k, "#666"))
    ax.set_xlabel("pick center error vs true box pose (mm)")
    ax.set_ylabel("boxes")
    ax.set_title("Absolute center accuracy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    labels, means, errs = [], [], []
    for k, rows in rows_by_noise.items():
        for f, lab in (("L_err_mm", "L"), ("W_err_mm", "W")):
            v = np.array([r[f] for r in rows], float)
            if len(v) == 0:
                continue
            labels.append(f"{lab}\n{k}")
            means.append(v.mean())
            errs.append(1.96 * v.std() / max(np.sqrt(len(v)), 1))
    ax.bar(range(len(means)), means, yerr=errs, capsize=4,
           color=["#2563eb", "#2563eb", "#d97706", "#d97706"][:len(means)])
    ax.axhline(0, color="#111", lw=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("dimension error (mm)")
    ax.set_title("Dimension bias (95% CI)")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    xs = [r["recall"] * 100 for r in all_res]
    ps = [r["precision"] * 100 for r in all_res]
    names = [r["noise"] for r in all_res]
    ax.bar([i - 0.18 for i in range(len(xs))], xs, 0.36, label="recall", color="#2563eb")
    ax.bar([i + 0.18 for i in range(len(ps))], ps, 0.36, label="precision", color="#d97706")
    for i, (a, b) in enumerate(zip(xs, ps)):
        ax.text(i - 0.18, a + 1, f"{a:.0f}", ha="center", fontsize=9)
        ax.text(i + 0.18, b + 1, f"{b:.0f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 112)
    ax.set_ylabel("%")
    ax.set_title("Detection vs true box count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Geometric detector v2 vs TRUE ground truth (MuJoCo cell twin) — no pseudo-GT circularity")
    fig.tight_layout()
    fig.savefig(path)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=24)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    all_res, rows_by_noise = [], {}
    for noise in (None, "tof"):
        key = noise or "clean"
        print(f"--- noise={key}")
        rows, per_scene = run(args.scenes, noise)
        res = summarize(rows, per_scene, noise)
        res["per_scene"] = per_scene
        all_res.append(res)
        rows_by_noise[key] = rows
        print(f"  => recall {res['recall']:.3f} precision {res['precision']:.3f} "
              f"center {res.get('center_err_mm', {}).get('mean')}mm "
              f"(p95 {res.get('center_err_mm', {}).get('p95')})")
    (OUT / "detect_eval.json").write_text(
        json.dumps({"results": all_res, "rows": rows_by_noise}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    chart(all_res, rows_by_noise, ROOT / "assets" / "twin_detect_accuracy.png")
    print("saved", OUT / "detect_eval.json")


if __name__ == "__main__":
    main()
