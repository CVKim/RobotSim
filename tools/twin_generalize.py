# -*- coding: utf-8 -*-
"""일반화 실험 — 단일 SKU·탑다운 가정이 어디서 깨지는지 트윈에서 잰다.

    .venv\\Scripts\\python.exe tools/twin_generalize.py              # results/twin_generalization.json + assets/twin_generalization.png
    .venv\\Scripts\\python.exe tools/twin_generalize.py --seeds 3 --tilts 0 10 20

레포의 첫 줄 한계가 "박스 종류 하나, 카메라 한 대" 다. 실측 데이터를 더 받을 수는 없지만, 트윈에서는 두 가정을 직접 흔들어
**어디까지 버티는지** 잴 수 있다. 정답을 알기 때문에 리콜·정밀도·중심 오차를 mm 로 낼 수 있다.

  실험 A (혼합 SKU): 팔레트에 서로 다른 크기의 박스 4종을 섞는다. 처음 가설은 'SKU 사전(293×219 ±15%)이 신뢰도를 깎아
                     규격 밖 박스가 픽 계획에서 빠진다' 였는데, 사전을 넓혀도 결과가 같았다 — 병목은 층 선택이었다.
                     그래서 후보 깊이마다 검출해 합치는 변형까지 넣어 비교한다.
  실험 B (카메라 기울기): 카메라를 팔레트 중심 둘레로 0~25도 기울인다. '층 = 같은 깊이' 가정이 깨지는 각도를 본다.

두 실험 모두 회사 데이터가 필요 없다(트윈 렌더). 결과는 공개 가능.
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
sys.path.insert(0, str(ROOT))

import robotsim_perception.geometry as geom  # noqa: E402
from binpick_topface import detect_boxes_v2  # noqa: E402
from robotsim_perception.geometry import layer_roi_mask, top_layer_candidates, valid_mask  # noqa: E402
from cell_scene import N_COL, N_ROW, build_xml  # noqa: E402
from cell_twin import TwinRenderer, camera_transform, ground_truth, match, settle  # noqa: E402

OUT_JSON = ROOT / "results" / "twin_generalization.json"
OUT_PNG = ROOT / "assets" / "twin_generalization.png"

# 혼합 SKU 카탈로그 (m). 전부 격자 피치(0.30 × 0.225) 안에 들어가되 크기·높이가 다르다.
CATALOG = {
    "A 기준": (0.293, 0.219, 0.283),
    "B 중간": (0.245, 0.185, 0.240),
    "C 소형": (0.200, 0.150, 0.200),
    "D 납작": (0.270, 0.205, 0.160),
}
MIN_CONF = 0.55          # 픽 계획에 들어가는 신뢰도 문턱 (runtime 기본값과 같다)


def make_scene(seed: int, n_cells: int, mixed: bool, tilt_deg: float = 0.0, layers: int = 2):
    """소스 팔레트에 2층으로 쌓은 씬. mixed 면 칸(열·행)마다 카탈로그에서 골라 그 칸의 두 층에 같은 SKU 를 쓴다.

    한 칸 안에서 층마다 크기를 바꾸면 실제로도 무너진다(혼합 적재는 보통 칸 단위로 같은 품목을 쌓는다).
    층을 2층으로 두는 것은 이 레포의 기존 트윈 실험과 같은 조건을 쓰기 위해서다 — 1층만 놓으면 데크가 화면을
    지배해 층 선택이 흔들리고, 그것은 이미 아는 문제라 이 실험의 관심사가 아니다."""
    import mujoco
    rng = np.random.default_rng(seed)
    cells = [(c, r) for r in range(N_ROW) for c in range(N_COL)]
    idx = sorted(rng.permutation(len(cells))[:n_cells])
    names = list(CATALOG)
    per_cell = {cells[i]: (names[int(rng.integers(0, len(names)))] if mixed else "A 기준") for i in idx}
    layout, sku, pick = [], [], []
    for layer in range(layers):
        for i in idx:
            c, r = cells[i]
            layout.append((c, r, layer))
            sku.append(CATALOG[per_cell[(c, r)]])
            pick.append(per_cell[(c, r)])
    xml, _ = build_xml(layout, seed=seed, sku=sku, cam_tilt_deg=tilt_deg)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    settle(m, d, 1500)
    return m, d, pick


def uncovered(gt, xy_tol_mm: float = 150.0, dz_mm: float = 50.0):
    """위에 다른 박스가 얹히지 않은 박스 = 지금 집을 수 있는 후보.

    높이가 제각각인 혼합 적재에서는 '최상층 ±45 mm' 라는 정의가 성립하지 않는다. 대신 기하로 판정한다:
    XY 로 겹치면서 더 얕은(카메라에 가까운) 상면이 있으면 덮인 것이다."""
    out = []
    for b in gt:
        c = np.asarray(b["center_mm"], float)
        covered = any(o is not b
                      and float(np.linalg.norm(np.asarray(o["center_mm"], float) - c)) < xy_tol_mm
                      and b["top_d_mm"] - o["top_d_mm"] > dz_mm
                      for o in gt)
        if not covered:
            out.append(b)
    return out


def detect_multi_layer(f, layer_roi_mm=620.0, k=4, dedupe_mm=90.0):
    """높이가 섞인 적재를 위한 변형: 후보 깊이마다 한 번씩 검출해 합친다.

    지금 검출기는 층 하나를 고르고 그 ±40 mm 만 본다. 높이가 다른 품목이 섞이면 상면이 100 mm 넘게 퍼져서
    한 번에 한 부류만 잡힌다. 후보 피크마다 시간 사전(prior_top_mm)을 그 깊이로 주고 돌린 뒤 중심 거리로 중복을 지운다.
    검출기 자체는 그대로 두고 호출 방식만 바꾼 것이라, 이 실험에서 '고칠 수 있는 문제' 임을 보이는 데 쓴다."""
    roi = layer_roi_mask(f, layer_roi_mm)
    cands = top_layer_candidates(f["D"], valid_mask(f), k=k, roi_mask=roi)
    out = []
    for c in cands:
        _, _, det = detect_boxes_v2(f, layer_roi_mm=layer_roi_mm, prior_top_mm=float(c))
        for b in det:
            if not b.get("center_mm"):
                continue
            cm = np.asarray(b["center_mm"][:2], float)
            hit = next((o for o in out if float(np.linalg.norm(np.asarray(o["center_mm"][:2], float) - cm)) < dedupe_mm), None)
            if hit is None:
                out.append(b)
            elif b.get("confidence", 0) > hit.get("confidence", 0):
                out[out.index(hit)] = b
    return out


def evaluate(m, d, rng, noise=None, tol_mm=90.0, layer_roi_mm=620.0, multi_layer=False):
    """한 장면 한 프레임: 검출 결과를 정답과 맞춰 리콜·정밀도·중심 오차·신뢰도를 낸다.

    정답은 '위에 아무것도 얹히지 않은 박스'(uncovered)로 본다. 혼합 SKU 는 높이가 제각각이라
    '최상층 ±45 mm' 라는 기존 정의가 성립하지 않기 때문이고, 그 점 자체가 이 실험이 드러내려는 것이다."""
    rend = TwinRenderer(m)
    try:
        f = rend.frame(d, noise=noise, rng=rng)
    finally:
        rend.close()
    T = camera_transform(m, d)
    gt = uncovered(ground_truth(m, d, top_only=False, T_base_cam=T))
    # 층 히스토그램을 팔레트 영역으로 한정한다(데크가 화면의 대부분이라 그냥 두면 층 선택이 데크로 간다).
    # 이 실험의 관심은 SKU·카메라 각도이지 이미 알려진 층 선택 문제가 아니므로, 알려진 최선 설정으로 고정한다.
    if multi_layer:
        det = detect_multi_layer(f, layer_roi_mm=layer_roi_mm)
    else:
        _, _, det = detect_boxes_v2(f, layer_roi_mm=layer_roi_mm)   # (선택 층, 디버그, 박스 목록)
    det = [b for b in det if b.get("center_mm")]
    pairs, missed, fp = match(gt, det, tol_mm=tol_mm)
    err = [float(np.linalg.norm(np.asarray(b["center_mm"][:2], float) - np.asarray(g["center_mm"], float)))
           for g, b, _ in pairs]
    dim_err = [float(abs(b.get("dims_mm", (0, 0))[0] - g["dims_mm"][0])) for g, b, _ in pairs if b.get("dims_mm")]
    conf = [float(b.get("confidence", 0.0)) for _, b, _ in pairs]
    pickable = [c >= MIN_CONF for c in conf]
    return {"n_gt": len(gt), "n_det": len(det), "matched": len(pairs), "missed": len(missed), "false_pos": len(fp),
            "recall": len(pairs) / max(len(gt), 1), "precision": len(pairs) / max(len(det), 1),
            "center_err_mm_mean": float(np.mean(err)) if err else None,
            "center_err_mm_p95": float(np.percentile(err, 95)) if err else None,
            "dim_err_mm_mean": float(np.mean(dim_err)) if dim_err else None,
            "conf_mean": float(np.mean(conf)) if conf else None,
            "pickable_rate": float(np.mean(pickable)) if pickable else 0.0}


VARIANTS = [
    ("단일 SKU", dict(mixed=False, wide=False, multi=False)),
    ("혼합 SKU", dict(mixed=True, wide=False, multi=False)),
    ("혼합 + 사전 넓힘", dict(mixed=True, wide=True, multi=False)),
    ("혼합 + 다층 병합", dict(mixed=True, wide=False, multi=True)),
]


def run_mixed(seeds, n_cells, noise, wide_tol=0.45):
    """실험 A: 단일 SKU 대비 혼합 SKU, 그리고 두 가지 대응(사전 넓히기 · 후보 깊이마다 검출해 합치기)."""
    rows = []
    base_fn = geom._dims_score
    for label, cfg in VARIANTS:
        # 절제: `_dims_score` 의 기본 인자는 정의 시점에 묶이므로 상수만 바꿔서는 안 먹는다 — 함수를 감싼다.
        if cfg["wide"]:
            geom._dims_score = lambda L, W, prior=geom.SKU_PRIOR_MM, tol=wide_tol: base_fn(L, W, prior, tol)
        try:
            for s in seeds:
                m, d, pick = make_scene(s, n_cells, cfg["mixed"])
                r = evaluate(m, d, np.random.default_rng(500 + s), noise=noise, multi_layer=cfg["multi"])
                rows.append(dict(r, experiment="mixed_sku", variant=label, seed=s,
                                 sku_mix={k: pick.count(k) for k in CATALOG if pick.count(k)}))
        finally:
            geom._dims_score = base_fn
    return rows


def run_tilt(seeds, n_boxes, tilts, noise):
    """실험 B: 카메라를 기울이며 같은 장면을 본다."""
    rows = []
    for t in tilts:
        for s in seeds:
            m, d, _ = make_scene(s, n_boxes, mixed=False, tilt_deg=t)
            r = evaluate(m, d, np.random.default_rng(700 + s), noise=noise)
            rows.append(dict(r, experiment="camera_tilt", tilt_deg=float(t), seed=s))
    return rows


def chart(rows, path):
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
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))

    mixed = [r for r in rows if r["experiment"] == "mixed_sku"]
    groups = [(lab.replace(" + ", chr(10) + "+ "), [r for r in mixed if r["variant"] == lab])
              for lab, _ in VARIANTS]
    x = np.arange(len(groups))
    for k, (key, lab, color) in enumerate((("recall", "리콜", "#1d4ed8"),
                                           ("precision", "정밀도", "#059669"))):
        v = [100 * float(np.mean([r[key] for r in g])) if g else 0.0 for _, g in groups]
        e = [100 * float(np.std([r[key] for r in g])) if g else 0.0 for _, g in groups]
        ax[0].bar(x + (k - 0.5) * 0.36, v, 0.34, yerr=e, capsize=3, color=color, label=lab)
        for xi, vi in zip(x + (k - 0.5) * 0.36, v):
            ax[0].text(xi, vi + 1.5, f"{vi:.0f}", ha="center", fontsize=8)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([n for n, _ in groups], fontsize=9)
    ax[0].set_ylabel("%")
    ax[0].set_ylim(0, 115)
    ax[0].set_title("혼합 SKU 는 리콜이 무너진다 (사전을 넓혀도 그대로)")
    ax[0].legend(fontsize=8, loc="lower left")

    tilt = [r for r in rows if r["experiment"] == "camera_tilt"]
    ts = sorted({r["tilt_deg"] for r in tilt})
    for key, lab, color, a in (("recall", "리콜", "#1d4ed8", ax[1]),
                               ("precision", "정밀도", "#059669", ax[1])):
        v = [100 * float(np.mean([r[key] for r in tilt if r["tilt_deg"] == t])) for t in ts]
        a.plot(ts, v, marker="o", color=color, label=lab)
    ax[1].set_xlabel("카메라 기울기 (deg)")
    ax[1].set_ylabel("%")
    ax[1].set_ylim(0, 105)
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    ax[1].set_title("탑다운 가정은 몇 도까지 버티나")

    v = [float(np.mean([r["center_err_mm_mean"] for r in tilt
                        if r["tilt_deg"] == t and r["center_err_mm_mean"] is not None]) or np.nan) for t in ts]
    ax[2].plot(ts, v, marker="o", color="#b91c1c")
    ax[2].set_xlabel("카메라 기울기 (deg)")
    ax[2].set_ylabel("중심 오차 (mm)")
    ax[2].grid(alpha=0.3)
    ax[2].set_title("찾은 박스의 위치 오차")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cells", type=int, default=12, help="팔레트 칸 수 (2층이므로 박스는 이 두 배)")
    ap.add_argument("--tilts", type=float, nargs="*", default=[0, 5, 10, 15, 20, 25])
    ap.add_argument("--noise", default="tof", choices=["none", "tof"])
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    noise = None if args.noise == "none" else args.noise
    seeds = list(range(args.seeds))

    rows = run_mixed(seeds, args.cells, noise)
    for lab, _ in VARIANTS:
        sub = [r for r in rows if r.get("variant") == lab]
        ce = [r["center_err_mm_mean"] for r in sub if r["center_err_mm_mean"] is not None]
        print(f"  {lab:16s}: 리콜 {100*np.mean([r['recall'] for r in sub]):5.1f}% · "
              f"정밀도 {100*np.mean([r['precision'] for r in sub]):5.1f}% · "
              f"픽 가능 {100*np.mean([r['pickable_rate'] for r in sub]):5.1f}% · "
              f"중심 오차 {np.mean(ce) if ce else float('nan'):.1f} mm", flush=True)

    rows += run_tilt(seeds, args.cells, args.tilts, noise)
    for t in args.tilts:
        sub = [r for r in rows if r["experiment"] == "camera_tilt" and r["tilt_deg"] == float(t)]
        ce = [r["center_err_mm_mean"] for r in sub if r["center_err_mm_mean"] is not None]
        print(f"  기울기 {t:4.1f}°: 리콜 {100*np.mean([r['recall'] for r in sub]):5.1f}% · "
              f"정밀도 {100*np.mean([r['precision'] for r in sub]):5.1f}% · "
              f"중심 오차 {np.mean(ce) if ce else float('nan'):.1f} mm", flush=True)

    summary = {
        "what": "단일 SKU·탑다운 가정을 셀 트윈에서 흔들어 본 기록 (실측 데이터 없이, 정답 있는 환경)",
        "catalog_mm": {k: [round(v * 1000, 1) for v in dims] for k, dims in CATALOG.items()},
        "min_confidence": MIN_CONF, "noise": args.noise, "boxes": args.cells, "seeds": args.seeds,
        "runs": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    chart(rows, OUT_PNG)
    print("saved", args.out, "및", OUT_PNG)


if __name__ == "__main__":
    main()
