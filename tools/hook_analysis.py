# -*- coding: utf-8 -*-
"""대차 견인 고리(hook) ToF 정밀 분석.

파이프라인 (세션당):
  1. X/Y/D 채널 -> 유효 포인트클라우드 (mm, 카메라 좌표계: X 우+, Y 하+, D 전방+)
  2. 지배 수평 플레이트 평면 RANSAC (numpy 직접 구현, 3점 샘플 500회, inlier 8mm,
     법선이 수직축(Y)과 35도 이내인 후보만 인정 -> 수평면 강제)
  3. 평면 위(카메라 -Y 방향) 12~100mm 높이 포인트 -> 픽셀 인접 connected components
     -> 크기/위치/치수로 가장 그럴듯한 클러스터 = 고리 후보
  4. 고리 중심(카메라 좌표 + 평면 기준 상대좌표), 바운딩박스 치수, 포인트 수
  5. 4세션 교차 반복성 + 시각화/JSON 저장

실행:  /d/anaconda/python E:/Robot_Sim/tools/hook_analysis.py
출력:  E:/Robot_Sim/explore/hook/  (PNG + hook_results.json)

주의: 콘솔 출력은 ASCII만 사용 (cp949 콘솔 대응).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mim_loader import load_session, valid_mask  # noqa: E402

try:
    from local_paths import DAECHA_DIR
except ImportError:
    import os
    DAECHA_DIR = os.environ["DAECHA_DIR"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import cv2  # noqa: E402

OUT_DIR = Path(r"E:\Robot_Sim\explore\hook")

# ---- 파라미터 ----------------------------------------------------------------
MAX_RANGE_MM = 1200.0      # 플레이트/고리 탐색 거리 상한 (배경 제외)
RANSAC_ITERS = 500
RANSAC_THRESH = 8.0        # 평면 inlier 임계 (mm)
NORMAL_MAX_TILT_DEG = 35.0 # 평면 법선이 Y축(수직)에서 벗어날 수 있는 최대 각
HOOK_H_MIN = 12.0          # 평면 위 높이 범위 (mm)
HOOK_H_MAX = 100.0
MIN_CLUSTER_PX = 50
MIN_EXTENT_MM = 8.0        # 클러스터 in-plane 최소 폭
MAX_EXTENT_MM = 250.0      # 최대 폭 (이보다 크면 벽/구조물)
RNG_SEED = 42


# ---- 평면 RANSAC (numpy) ------------------------------------------------------
def fit_plane_svd(pts):
    """pts (N,3) -> (unit normal, d)  where  n . p = d  (최소자승, SVD)."""
    c = pts.mean(axis=0)
    q = pts - c
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    n = vt[-1]
    n = n / np.linalg.norm(n)
    return n, float(n @ c)


def ransac_horizontal_plane(pts, iters=RANSAC_ITERS, thresh=RANSAC_THRESH,
                            max_tilt_deg=NORMAL_MAX_TILT_DEG, seed=RNG_SEED):
    """지배 수평 평면 RANSAC. 법선이 카메라 Y축과 max_tilt_deg 이내인 후보만.

    반환: (n, d, inlier_mask)  --  n은 '위쪽'(카메라 -Y 방향) 지향, n.p = d
    """
    rng = np.random.default_rng(seed)
    n_pts = len(pts)
    # 스코어링은 서브샘플로 (속도), 최종 inlier는 전체에서
    score_idx = rng.choice(n_pts, size=min(n_pts, 60000), replace=False)
    score_pts = pts[score_idx]
    cos_max = np.cos(np.deg2rad(max_tilt_deg))

    best = (None, None, -1)
    for _ in range(iters):
        i3 = rng.choice(n_pts, size=3, replace=False)
        p0, p1, p2 = pts[i3]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn
        if abs(n[1]) < cos_max:      # 수평면 강제: 법선이 수직축(Y)에 가까워야
            continue
        d = float(n @ p0)
        cnt = int(np.count_nonzero(np.abs(score_pts @ n - d) < thresh))
        if cnt > best[2]:
            best = (n, d, cnt)

    if best[0] is None:
        raise RuntimeError("RANSAC failed: no horizontal plane candidate")

    n, d = best[0], best[1]
    # 전체 포인트로 2회 정련 (inlier -> SVD 재피팅)
    for _ in range(2):
        inl = np.abs(pts @ n - d) < thresh
        n, d = fit_plane_svd(pts[inl])
    # '위쪽' 지향: 카메라 Y는 아래+ 이므로 위 = -Y => n[1] < 0
    if n[1] > 0:
        n, d = -n, -d
    inl = np.abs(pts @ n - d) < thresh
    return n, d, inl


def plane_frame(n, d):
    """평면 좌표계: origin=카메라원점의 평면 투영, u=카메라X의 평면투영, v=n x u."""
    origin = d * n
    x_axis = np.array([1.0, 0.0, 0.0])
    u = x_axis - (x_axis @ n) * n
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    return origin, u, v


# ---- 세션 분석 ----------------------------------------------------------------
def analyze_session(sess_dir):
    sess = load_session(sess_dir)
    H, W = sess["D"].shape
    vm = valid_mask(sess) & (sess["D"] < MAX_RANGE_MM)
    P = np.stack([sess["X"], sess["Y"], sess["D"]], axis=-1).astype(np.float64)
    pts = P[vm]

    n, d, inl_sub = ransac_horizontal_plane(pts)
    origin, u_ax, v_ax = plane_frame(n, d)

    # 전 픽셀 signed height (평면 위 = +)
    height = np.full((H, W), np.nan)
    height[vm] = P[vm] @ n - d

    plane_inlier_px = vm & (np.abs(height) < RANSAC_THRESH)
    hook_px = vm & (height >= HOOK_H_MIN) & (height <= HOOK_H_MAX)

    # connected components (8-이웃, 픽셀 인접 기반)
    n_lab, labels, stats, centroids = cv2.connectedComponentsWithStats(
        hook_px.astype(np.uint8), connectivity=8)

    cx0, cy0 = W / 2.0, H / 2.0
    candidates = []
    for lab in range(1, n_lab):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < MIN_CLUSTER_PX:
            continue
        cmask = labels == lab
        cpts = P[cmask]
        rel = cpts - origin
        uu, vv, hh = rel @ u_ax, rel @ v_ax, cpts @ n - d
        ext_u = float(uu.max() - uu.min())
        ext_v = float(vv.max() - vv.min())
        if max(ext_u, ext_v) > MAX_EXTENT_MM or max(ext_u, ext_v) < MIN_EXTENT_MM:
            continue
        # 위치 가중 (RGB 기준 고리는 화면 중앙 부근)
        dc = float(np.hypot(centroids[lab][0] - cx0, centroids[lab][1] - cy0))
        score = area * np.exp(-dc / 250.0)
        candidates.append(dict(
            label=lab, area_px=area, score=float(score),
            center_cam_mm=[float(c) for c in cpts.mean(axis=0)],
            center_plane_mm=[float(uu.mean()), float(vv.mean()), float(hh.mean())],
            bbox_plane_mm=dict(u=ext_u, v=ext_v,
                               h=[float(hh.min()), float(hh.max())]),
            centroid_px=[float(centroids[lab][0]), float(centroids[lab][1])],
        ))

    candidates.sort(key=lambda c: -c["score"])
    hook = candidates[0] if candidates else None
    hook_mask = (labels == hook["label"]) if hook else np.zeros((H, W), bool)

    result = dict(
        session=sess_dir.name,
        n_valid=int(vm.sum()),
        plane=dict(normal=[float(x) for x in n], d_mm=float(d),
                   n_inliers=int(plane_inlier_px.sum()),
                   tilt_from_vertical_deg=float(np.rad2deg(np.arccos(abs(n[1]))))),
        hook_detected=hook is not None,
        hook=hook,
        n_candidates=len(candidates),
        other_candidates=[dict(area_px=c["area_px"],
                               center_cam_mm=c["center_cam_mm"])
                          for c in candidates[1:4]],
    )
    viz = dict(D=sess["D"], vm=vm, height=height,
               plane_px=plane_inlier_px, hook_mask=hook_mask)
    return result, viz


# ---- 시각화 -------------------------------------------------------------------
def save_overlay(viz, out_path):
    D, vm = viz["D"], viz["vm"]
    dvis = np.zeros_like(D)
    if vm.any():
        lo, hi = np.percentile(D[vm], [1, 99])
        dvis[vm] = np.clip((D[vm] - lo) / max(hi - lo, 1e-6), 0, 1)
    img = (dvis * 180).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img[viz["plane_px"]] = (160, 160, 160)          # 평면 inlier: 회색
    img[viz["hook_mask"]] = (0, 0, 255)             # 고리 클러스터: 빨강
    cv2.imwrite(str(out_path), img)


def save_heightmap(viz, out_path, title):
    h = viz["height"].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(np.clip(h, -20, 120), cmap="turbo", vmin=-20, vmax=120)
    fig.colorbar(im, ax=ax, label="height above plate plane (mm)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def save_scatter(results, out_path):
    ok = [r for r in results if r["hook_detected"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, key, lbl in [(axes[0], "center_cam_mm", "camera frame X/Y (mm)"),
                         (axes[1], "center_plane_mm", "plate-plane frame u/v (mm)")]:
        xs = [r["hook"][key][0] for r in ok]
        ys = [r["hook"][key][1] for r in ok]
        ax.scatter(xs, ys, c="crimson", s=60, zorder=3)
        for r, x, y in zip(ok, xs, ys):
            ax.annotate(r["session"][-6:], (x, y), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_title("hook centers - " + lbl)
        ax.set_xlabel("X (mm)" if "cam" in key else "u (mm)")
        ax.set_ylabel("Y (mm)" if "cam" in key else "v (mm)")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---- 반복성 -------------------------------------------------------------------
def repeatability(results):
    ok = [r for r in results if r["hook_detected"]]
    if len(ok) < 2:
        return None
    out = {}
    for key in ["center_cam_mm", "center_plane_mm"]:
        arr = np.array([r["hook"][key] for r in ok])
        mean = arr.mean(axis=0)
        dev = arr - mean
        out[key] = dict(
            mean=[float(x) for x in mean],
            std=[float(x) for x in arr.std(axis=0)],
            max_abs_dev=[float(x) for x in np.abs(dev).max(axis=0)],
            rms_3d=float(np.sqrt((dev ** 2).sum(axis=1).mean())),
            max_pairwise_3d=float(max(
                np.linalg.norm(a - b) for i, a in enumerate(arr)
                for b in arr[i + 1:])),
        )
    return out


# ---- main ---------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions = sorted(p for p in Path(DAECHA_DIR).iterdir()
                      if (p / "4_tof_D.mim").exists())
    print("ToF sessions:", len(sessions))

    results = []
    for sp in sessions:
        res, viz = analyze_session(sp)
        results.append(res)
        save_overlay(viz, OUT_DIR / f"{sp.name}_overlay.png")
        save_heightmap(viz, OUT_DIR / f"{sp.name}_heightmap.png",
                       f"{sp.name} height above plate")
        st = "OK " if res["hook_detected"] else "FAIL"
        print(f"[{st}] {sp.name}  valid={res['n_valid']}  "
              f"plane_inl={res['plane']['n_inliers']}  "
              f"tilt={res['plane']['tilt_from_vertical_deg']:.1f}deg  "
              f"cand={res['n_candidates']}")
        if res["hook_detected"]:
            h = res["hook"]
            c = h["center_cam_mm"]
            bb = h["bbox_plane_mm"]
            print(f"       hook center cam=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f})mm "
                  f"bbox u={bb['u']:.1f} v={bb['v']:.1f} "
                  f"h={bb['h'][0]:.1f}..{bb['h'][1]:.1f}mm  px={h['area_px']}")

    rep = repeatability(results)
    save_scatter(results, OUT_DIR / "hook_centers_scatter.png")

    if rep:
        for key in ["center_cam_mm", "center_plane_mm"]:
            r = rep[key]
            print(f"repeatability [{key}]: std=({r['std'][0]:.2f},"
                  f"{r['std'][1]:.2f},{r['std'][2]:.2f})mm  "
                  f"rms3d={r['rms_3d']:.2f}mm  maxpair={r['max_pairwise_3d']:.2f}mm")

    payload = dict(
        meta=dict(
            units="mm", date="2026-09-01",
            camera_frame="X right+, Y down+ (image), D forward+",
            plane_frame=("origin=camera origin projected to plate plane, "
                         "u=camera X projected, v=n x u, h=height above plane "
                         "(n oriented up, i.e. -Y side)"),
            params=dict(ransac_iters=RANSAC_ITERS, inlier_thresh_mm=RANSAC_THRESH,
                        normal_max_tilt_deg=NORMAL_MAX_TILT_DEG,
                        hook_height_mm=[HOOK_H_MIN, HOOK_H_MAX],
                        max_range_mm=MAX_RANGE_MM,
                        min_cluster_px=MIN_CLUSTER_PX),
        ),
        sessions=results,
        repeatability=rep,
    )
    out_json = OUT_DIR / "hook_results.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print("saved:", out_json)


if __name__ == "__main__":
    main()
