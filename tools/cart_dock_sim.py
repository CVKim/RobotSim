# -*- coding: utf-8 -*-
"""대차 도킹 시뮬 평가 — AGV(운동학)가 인식한 고리 포즈로 결합 위치까지 가는 폐루프를 여러 시작 자세에서 돌린다.

    python tools/cart_dock_sim.py [--starts 30] [--noise 2.5]
    -> results/cart_dock.json, assets/cart_dock.png

한 회: RelPose(고리 u, 림 v, 대차 요) 시작 → 프레임마다 합성 대차 장면 렌더(synthetic_cart) → analyze_cart → 고리 평면 좌표·요 →
DockController → (v, ω) → 운동학 step → … 결합 판정(|u| < 10 mm, |v − 결합거리| < 10 mm, |요| < 2°) 또는 60스텝(30 s).
oracle(정답 측정)을 나란히 돌려 인식 때문에 잃는 것을 분리한다. 회사 데이터 없이 돈다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robotsim_perception.cart import analyze_cart  # noqa: E402
from robotsim_perception.dock import DockController, RelPose, measurement_from_result, simulate  # noqa: E402
from robotsim_perception.synthetic_cart import make_cart_frame  # noqa: E402

OUT_JSON = ROOT / "results" / "cart_dock.json"
OUT_PNG = ROOT / "assets" / "cart_dock.png"


def random_starts(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(RelPose(float(rng.uniform(-150, 150)), float(rng.uniform(-500, -290)), float(rng.uniform(-8, 8))))
    return out


def run(n_starts: int, noise_mm: float, dt: float, max_steps: int):
    ctrl = DockController()
    starts = random_starts(n_starts)
    rows = {"perception": [], "oracle": []}
    seed = [1000]

    def perception(rp):
        seed[0] += 1
        frame, _ = make_cart_frame(noise_mm=noise_mm, seed=seed[0], **rp.render_kwargs())
        t0 = time.perf_counter()
        res = analyze_cart(frame)
        perception.lat.append((time.perf_counter() - t0) * 1e3)
        return measurement_from_result(res)
    perception.lat = []

    def oracle(rp):
        u, v = rp.hook_plane()
        return u, v, rp.yaw_deg

    for i, rp0 in enumerate(starts):
        for mode, meas in (("perception", perception), ("oracle", oracle)):
            r = simulate(rp0, ctrl, meas, dt_s=dt, max_steps=max_steps)
            rows[mode].append({"start": rp0.to_dict(), **{k: r[k] for k in ("success", "steps", "final", "errors", "truth_docked", "misses")},
                               "path": r["path"]})
        p, o = rows["perception"][-1], rows["oracle"][-1]
        print(f"  start {i:2d} u={rp0.hook_u_mm:6.0f} v={rp0.rim_v_mm:6.0f} yaw={rp0.yaw_deg:5.1f} | "
              f"perception {'ok ' if p['success'] else 'FAIL'} steps {p['steps']:2d} err u {p['errors']['e_u']:5.1f} v {p['errors']['e_v']:5.1f} "
              f"yaw {p['errors']['e_yaw']:4.1f} misses {p['misses']} | oracle steps {o['steps']:2d}", flush=True)

    def summ(rs):
        ok = [r for r in rs if r["success"]]
        eu = [abs(r["errors"]["e_u"]) for r in ok]; ev = [abs(r["errors"]["e_v"]) for r in ok]; ey = [abs(r["errors"]["e_yaw"]) for r in ok]
        return {"n": len(rs), "success": len(ok), "success_rate": round(len(ok) / max(len(rs), 1), 3),
                "truth_docked": sum(1 for r in ok if r["truth_docked"]),
                "steps_mean": round(float(np.mean([r["steps"] for r in ok])), 1) if ok else None,
                "time_s_mean": round(float(np.mean([r["steps"] for r in ok])) * dt, 1) if ok else None,
                "final_err_mm_mean": {"u": round(float(np.mean(eu)), 1), "v": round(float(np.mean(ev)), 1)} if ok else None,
                "final_err_mm_p95": {"u": round(float(np.percentile(eu, 95)), 1), "v": round(float(np.percentile(ev, 95)), 1)} if ok else None,
                "final_yaw_err_deg_mean": round(float(np.mean(ey)), 2) if ok else None,
                "misses_total": sum(r["misses"] for r in rs)}
    out = {"what": "AGV(차동 구동 운동학) 도킹 폐루프: 합성 대차 장면 -> analyze_cart -> 고리 평면 좌표·요 -> pure pursuit -> 결합 위치",
           "controller": DockController().__dict__, "dt_s": dt, "max_steps": max_steps, "noise_mm": noise_mm,
           "start_ranges": {"hook_u_mm": [-150, 150], "rim_v_mm": [-500, -290], "yaw_deg": [-8, 8]},
           "detection_range_note": "합성 대차의 고리는 림 거리 500 mm 까지 검출된다(600 mm 이상은 NO_HOOK) — 시작 범위를 그 안으로 둔다",
           "perception": summ(rows["perception"]), "oracle": summ(rows["oracle"]),
           "perception_latency_ms_mean": round(float(np.mean(perception.lat)), 1),
           "runs": {"perception": [{k: v for k, v in r.items() if k != "path"} for r in rows["perception"]],
                    "oracle": [{k: v for k, v in r.items() if k != "path"} for r in rows["oracle"]]}}
    return out, rows


def chart(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    for cand in (r"C:\Windows\Fonts\malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        if Path(cand).exists():
            fm.fontManager.addfont(cand)
            plt.rcParams["font.family"] = fm.FontProperties(fname=cand).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    for r in rows["perception"]:
        us = [p["hook_u_mm"] for p in r["path"]]; vs = [p["rim_v_mm"] for p in r["path"]]
        ax.plot(us, vs, "-", color="#dd8452" if r["success"] else "#c44e52", alpha=0.6, lw=1)
        ax.plot(us[0], vs[0], "o", color="#4c72b0", ms=3)
    ax.plot([0], [DockController().dock_v_mm], "k*", ms=12, label="결합 위치")
    ax.set_xlabel("고리 u (mm, AGV 기준 좌우)"); ax.set_ylabel("림 v (mm, 음수 = 앞)")
    ax.set_title(f"인식 구동 궤적 {out['perception']['n']}회 (파란 점 = 시작)", fontsize=10); ax.legend(fontsize=8); ax.set_aspect("equal")
    ax = axes[1]
    for mode, col in (("perception", "#dd8452"), ("oracle", "#4c72b0")):
        ok = [r for r in rows[mode] if r["success"]]
        ax.scatter([r["errors"]["e_u"] for r in ok], [r["errors"]["e_v"] for r in ok], s=18, color=col, alpha=0.8,
                   label=f"{'인식 구동' if mode == 'perception' else '정답 측정(oracle)'} n={len(ok)}")
    t = DockController()
    ax.add_patch(plt.Rectangle((-t.tol_u_mm, -t.tol_v_mm), 2 * t.tol_u_mm, 2 * t.tol_v_mm, fill=False, ls="--", color="gray"))
    ax.set_xlabel("최종 측방 오차 u (mm, 정답 기준)"); ax.set_ylabel("최종 거리 오차 v (mm)")
    ax.set_title("결합 판정 시점의 실제 오차 (점선 = 판정 허용치)", fontsize=10); ax.legend(fontsize=8); ax.set_xlim(-30, 30); ax.set_ylim(-30, 30)
    ax = axes[2]
    sp = [r["steps"] for r in rows["perception"] if r["success"]]; so = [r["steps"] for r in rows["oracle"] if r["success"]]
    ax.hist([np.array(so) * out["dt_s"], np.array(sp) * out["dt_s"]], bins=12, color=["#4c72b0", "#dd8452"], label=["oracle", "인식 구동"])
    ax.set_xlabel("도킹 시간 (s)"); ax.set_ylabel("회"); ax.set_title(f"도킹 시간: 인식 평균 {out['perception']['time_s_mean']} s", fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=150); print("saved", OUT_PNG)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=30)
    ap.add_argument("--noise", type=float, default=2.5)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=60)
    args = ap.parse_args()
    out, rows = run(args.starts, args.noise, args.dt, args.max_steps)
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("perception", "oracle", "perception_latency_ms_mean")}, indent=1, ensure_ascii=False))
    print("saved", OUT_JSON)
    chart(out, rows)


if __name__ == "__main__":
    main()
