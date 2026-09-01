# -*- coding: utf-8 -*-
"""ToF 시간적 노이즈 특성 연구 (T7 예열).

같은 고정 카메라로 찍힌 N개 세션에서 '장면이 변하지 않은(정적) 픽셀'만 골라
깊이 반복 측정치의 산포를 구하면, 새 촬영 없이 노이즈 모델을 얻는다:
  - sigma(distance): 거리 의존 노이즈 (BlenderProc add_kinect_azure_noise 튜닝용)
  - sigma(intensity): ToF 물리(shot noise ~ 1/I)에 따른 강도 의존성
정적 판정: 유효 세션 >= min_valid 이고, 세션 값의 80% 이상이 중앙값 +-30mm 이내.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mim_loader import load_mim

try:
    from local_paths import BINPICK_DIR
except ImportError:
    BINPICK_DIR = os.environ["BINPICK_DIR"]

OUT = Path(r"E:\Robot_Sim\explore\noise")
OUT.mkdir(parents=True, exist_ok=True)


def load_stacks(root):
    Ds, Is = [], []
    for s in sorted(d for d in Path(root).iterdir() if d.is_dir()):
        dp, ip = s / "4_tof_D.mim", s / "5_tof_I.mim"
        if dp.exists():
            Ds.append(load_mim(dp))
            Is.append(load_mim(ip))
    return np.stack(Ds), np.stack(Is)  # (N,480,640)


def main():
    D, I = load_stacks(BINPICK_DIR)
    n = D.shape[0]
    valid = (D > 0) & (D < 16000)
    med = np.where(valid, D, np.nan)
    med = np.nanmedian(med, axis=0)

    within = valid & (np.abs(D - med[None]) < 30)
    n_within = within.sum(axis=0)
    static = (valid.sum(axis=0) >= 25) & (n_within >= 0.8 * n)

    # 정적 픽셀의 강건 표준편차 (MAD 기반)
    vals = np.where(within, D, np.nan)
    mad = np.nanmedian(np.abs(vals - med[None]), axis=0)
    sigma = 1.4826 * mad
    med_int = np.nanmedian(np.where(within, I, np.nan), axis=0)

    s_d, s_sig, s_int = med[static], sigma[static], med_int[static]
    print(f"static pixels: {static.sum()} / {static.size} ({static.mean()*100:.0f}%)")

    # --- sigma vs distance ---
    bins = np.arange(2000, 4500, 100)
    idx = np.digitize(s_d, bins)
    bin_c, bin_med, bin_q1, bin_q3 = [], [], [], []
    for b in range(1, len(bins)):
        sel = idx == b
        if sel.sum() < 200:
            continue
        bin_c.append(bins[b - 1] + 50)
        q1, q2, q3 = np.percentile(s_sig[sel], [25, 50, 75])
        bin_med.append(q2), bin_q1.append(q1), bin_q3.append(q3)
    bin_c, bin_med = np.array(bin_c), np.array(bin_med)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    ax.fill_between(bin_c / 1000, bin_q1, bin_q3, alpha=0.25, label="IQR")
    ax.plot(bin_c / 1000, bin_med, "o-", label="median sigma")
    ax.set_xlabel("distance (m)"), ax.set_ylabel("temporal noise sigma (mm)")
    ax.set_title(f"ToF noise vs distance: NON-monotonic (n={n} sessions) -> intensity dominates")
    ax.grid(alpha=0.3), ax.legend()
    fig.tight_layout(), fig.savefig(OUT / "noise_vs_distance.png")

    # --- sigma vs intensity ---
    good = np.isfinite(s_int) & (s_int > 1)
    li = np.log10(s_int[good])
    ibins = np.linspace(li.min(), np.percentile(li, 99), 20)
    iidx = np.digitize(li, ibins)
    ic, imed = [], []
    for b in range(1, len(ibins)):
        sel = iidx == b
        if sel.sum() < 200:
            continue
        ic.append(10 ** (ibins[b - 1] + (ibins[1] - ibins[0]) / 2))
        imed.append(np.median(s_sig[good][sel]))
    ic_a, imed_a = np.array(ic), np.array(imed)
    pc = np.polyfit(np.log10(ic_a), np.log10(imed_a), 1)  # log sigma = b*log I + log a
    fit_sig = 10 ** np.polyval(pc, np.log10(ic_a))
    fig2, ax2 = plt.subplots(figsize=(7, 4.5), dpi=130)
    ax2.loglog(ic, imed, "o-", label="measured")
    ax2.loglog(ic_a, fit_sig, "--", label=f"power law: sigma = {10**pc[1]:.1f} * I^{pc[0]:.2f}")
    ax2.legend()
    ax2.set_xlabel("ToF intensity (a.u.)"), ax2.set_ylabel("sigma (mm)")
    ax2.set_title("ToF depth noise vs return intensity (shot-noise regime)")
    ax2.grid(alpha=0.3, which="both")
    fig2.tight_layout(), fig2.savefig(OUT / "noise_vs_intensity.png")

    # --- 공간 노이즈 맵 ---
    smap = np.full(sigma.shape, np.nan)
    smap[static] = sigma[static]
    fig3, ax3 = plt.subplots(figsize=(7, 5), dpi=130)
    im = ax3.imshow(smap, vmin=0, vmax=np.nanpercentile(smap, 98), cmap="inferno")
    plt.colorbar(im, ax=ax3, label="sigma (mm)")
    ax3.set_title("spatial map of temporal noise (static px only)")
    fig3.tight_layout(), fig3.savefig(OUT / "sigma_map.png")

    stats = {
        "n_sessions": int(n),
        "static_px": int(static.sum()),
        "sigma_intensity_power_law": {"a_mm": round(float(10**pc[1]), 3), "exponent": round(float(pc[0]), 3)},
        "finding": "noise is intensity-dominated, not distance-dominated; use sigma(I) for DR",
        "sigma_by_distance": [
            {"d_mm": int(c), "sigma_med": round(m, 2), "q1": round(a, 2), "q3": round(b, 2)}
            for c, m, a, b in zip(bin_c, bin_med, bin_q1, bin_q3)
        ],
        "sigma_by_intensity": [{"I": round(a, 1), "sigma_med": round(b, 2)} for a, b in zip(ic, imed)],
    }
    (OUT / "noise_stats.json").write_text(json.dumps(stats, indent=1, default=float), encoding="utf-8")
    print("power law: sigma(mm) =", round(float(10**pc[1]), 3), "* I ^", round(float(pc[0]), 3))
    print("saved:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
