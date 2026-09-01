# -*- coding: utf-8 -*-
"""대차 견인 고리(hook) ToF 정밀 분석.

장면/좌표 (탐색 결과 반영):
  * 카메라 좌표: X 우+, Y 하+(이미지), D 전방+ (mm). 어안 광각 ToF.
  * 지배 평면 = 대차 선반 플레이트. 카메라가 크게 피치되어 있어 플레이트 법선은
    카메라 Y축에서 약 50도 기울어짐 (법선 게이트를 65도로 완화).
  * 플레이트 포인트클라우드는 광각 왜곡으로 수 cm 휘어 있음 → RANSAC 평면 후
    평면좌표(u,v) 2차 다항면으로 강건 정련(robust quadratic refinement).
  * '고리가 평면 위로 솟은' 방향 = 카메라가 있는 쪽(플레이트의 보이는 면 쪽).
    반대쪽(+먼쪽)은 대차 몸체/배경/슬롯 투과가 놓임. 따라서 높이 h는
    법선을 카메라 쪽으로 지향시켜 계산한다 (h>0 = 플레이트에서 카메라 쪽으로 돌출).

파이프라인 (세션당):
  1. 유효 포인트클라우드 (D<1200mm)
  2. 지배 플레이트 평면 RANSAC (numpy, 3점 샘플 500회, inlier 8mm,
     법선-Y축 각 <= 65도) + 카메라쪽 지향 + 2차면 강건 정련
  3. h in [12,100]mm 밴드 → 모폴로지 오프닝(3x3 x2) → 픽셀 인접 connected
     components → 크기(mm)/플레이트 인접/화면중앙 근접 점수로 고리 후보 선정
  4. 중심(카메라/평면 좌표), 평면좌표 바운딩박스 치수, 포인트 수
  5. 4세션 교차 반복성 + 시각화 + JSON

실행:  /d/anaconda/python E:/Robot_Sim/tools/hook_analysis.py
출력:  E:/Robot_Sim/explore/hook/   (콘솔 출력은 ASCII만)
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
MAX_RANGE_MM = 1200.0       # 배경 제외 거리 상한
RANSAC_ITERS = 500
RANSAC_THRESH = 8.0         # 평면 inlier 임계 (mm)
NORMAL_MAX_TILT_DEG = 65.0  # 법선-카메라Y축 최대 각 (카메라 피치 커서 65도)
QUAD_REFINE_BAND = 30.0     # 2차면 정련에 쓰는 초기 밴드 (mm)
HOOK_H_MIN = 12.0           # 플레이트 위(카메라쪽) 높이 밴드 (mm)
HOOK_H_MAX = 100.0
PLATE_INLIER_MM = 8.0
MIN_CLUSTER_PX = 50
MIN_EXTENT_MM = 15.0        # 클러스터 in-plane 최소 폭
MAX_EXTENT_MM = 250.0       # 최대 폭
MORPH_ITERS = 2             # 얇은 경계 잡음 제거용 오프닝 반복
SPLIT_ITERS = (4, 6, 8)     # 과대 클러스터(레일+고리 병합) 분리용 추가 오프닝
CENTER_SIGMA_PX = 250.0     # 화면중앙 근접 가중 스케일
SIZE_PRIOR_MM = 45.0        # 고리 폭 사전값 (스펙: 30~60mm) - 소프트 가우시안
SIZE_SIGMA_MM = 35.0
RNG_SEED = 42


# ---- 평면 피팅 ----------------------------------------------------------------
def fit_plane_svd(pts):
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1] / np.linalg.norm(vt[-1])
    return n, float(n @ c)


def ransac_plate_plane(pts, iters=RANSAC_ITERS, thresh=RANSAC_THRESH,
                       max_tilt_deg=NORMAL_MAX_TILT_DEG, seed=RNG_SEED):
    """지배 플레이트 평면 RANSAC.

    반환 (n, d): n.p = d, n은 '카메라 쪽'(고리 돌출 방향) 지향 → 카메라 원점의
    signed height = -d > 0.
    """
    rng = np.random.default_rng(seed)
    sc = pts[rng.choice(len(pts), size=min(len(pts), 60000), replace=False)]
    cos_max = np.cos(np.deg2rad(max_tilt_deg))
    best = (None, None, -1)
    for _ in range(iters):
        p0, p1, p2 = pts[rng.choice(len(pts), size=3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn
        if abs(n[1]) < cos_max:      # 수직 벽(법선-Y각 > 65deg) 배제
            continue
        d = float(n @ p0)
        cnt = int(np.count_nonzero(np.abs(sc @ n - d) < thresh))
        if cnt > best[2]:
            best = (n, d, cnt)
    if best[0] is None:
        raise RuntimeError("RANSAC failed: no plate plane candidate")
    n, d = best[0], best[1]
    for _ in range(2):               # 전체 포인트 SVD 정련
        inl = np.abs(pts @ n - d) < thresh
        n, d = fit_plane_svd(pts[inl])
    if d > 0:                        # 카메라(원점) 쪽 지향: -d = 카메라 높이 > 0
        n, d = -n, -d
    return n, d


def plane_frame(n, d):
    """평면 좌표계: origin=카메라원점의 평면 투영, u=카메라X 투영, v=n x u."""
    origin = d * n
    u = np.array([1.0, 0.0, 0.0]) - n[0] * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return origin, u, v


def quad_refine_height(uu, vv, hh, band=QUAD_REFINE_BAND):
    """플레이트 휨 보정: h ~ quad(u,v) 강건 피팅 후 잔차 높이 반환."""
    sel = np.abs(hh) < band
    coef = np.zeros(6)
    Af = np.stack([np.ones(len(uu)), uu, vv, uu**2, uu * vv, vv**2], axis=1)
    hq = hh
    for _ in range(3):
        A = Af[sel]
        coef, *_ = np.linalg.lstsq(A, hh[sel], rcond=None)
        hq = hh - Af @ coef
        mad = 1.4826 * np.median(np.abs(hq[sel] - np.median(hq[sel])))
        sel = np.abs(hq) < max(3 * mad, 6.0)
    return hq, coef, float(mad)


# ---- 세션 분석 ----------------------------------------------------------------
def analyze_session(sess_dir):
    sess = load_session(sess_dir)
    H, W = sess["D"].shape
    vm = valid_mask(sess) & (sess["D"] < MAX_RANGE_MM)
    P = np.stack([sess["X"], sess["Y"], sess["D"]], axis=-1).astype(np.float64)
    pts = P[vm]

    n, d = ransac_plate_plane(pts)
    origin, u_ax, v_ax = plane_frame(n, d)
    rel = pts - origin
    uu, vv = rel @ u_ax, rel @ v_ax
    hh = pts @ n - d                       # 카메라쪽 = 양수
    hq, quad_coef, plate_mad = quad_refine_height(uu, vv, hh)

    height = np.full((H, W), np.nan)
    height[vm] = hq
    plate_px = vm & (np.abs(height) < PLATE_INLIER_MM)
    band_px = vm & (height >= HOOK_H_MIN) & (height <= HOOK_H_MAX)

    # 얇은 경계(플라잉 픽셀) 제거 후 픽셀 인접 클러스터링
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(band_px.astype(np.uint8), cv2.MORPH_OPEN,
                              kernel, iterations=MORPH_ITERS).astype(bool)

    # 고리 후보는 플레이트에서 솟아야 함 → 플레이트 inlier에 인접(15px 팽창 교차)
    plate_near = cv2.dilate(plate_px.astype(np.uint8), kernel,
                            iterations=15).astype(bool)
    cx0, cy0 = W / 2.0, H / 2.0

    def measure(cmask, split_iters):
        """클러스터 마스크 → 후보 dict (필터 통과 못하면 None, 과대면 'big')."""
        area = int(cmask.sum())
        if area < MIN_CLUSTER_PX:
            return None
        if not (cmask & plate_near).any():
            return None
        cpts = P[cmask]
        crel = cpts - origin
        cu, cv_, ch = crel @ u_ax, crel @ v_ax, height[cmask]
        ext_u = float(np.percentile(cu, 98) - np.percentile(cu, 2))
        ext_v = float(np.percentile(cv_, 98) - np.percentile(cv_, 2))
        if max(ext_u, ext_v) > MAX_EXTENT_MM:
            return "big"
        if max(ext_u, ext_v) < MIN_EXTENT_MM:
            return None
        ys, xs = np.nonzero(cmask)
        dc = float(np.hypot(xs.mean() - cx0, ys.mean() - cy0))
        size_w = float(np.exp(-0.5 * ((max(ext_u, ext_v) - SIZE_PRIOR_MM)
                                      / SIZE_SIGMA_MM) ** 2))
        score = area * float(np.exp(-dc / CENTER_SIGMA_PX)) * size_w
        return dict(
            area_px=area, score=round(score, 1),
            center_cam_mm=[round(float(x), 1) for x in cpts.mean(axis=0)],
            center_plane_mm=[round(float(cu.mean()), 1),
                             round(float(cv_.mean()), 1),
                             round(float(np.median(ch)), 1)],
            bbox_plane_mm=dict(u=round(ext_u, 1), v=round(ext_v, 1),
                               h=[round(float(np.percentile(ch, 2)), 1),
                                  round(float(np.percentile(ch, 98)), 1)]),
            centroid_px=[round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
            dist_to_img_center_px=round(dc, 1),
            split_open_iters=split_iters,
        )

    candidates, cand_masks = [], []
    n_lab, labels = cv2.connectedComponents(opened.astype(np.uint8), connectivity=8)
    for lab in range(1, n_lab):
        cmask = labels == lab
        got = measure(cmask, MORPH_ITERS)
        if isinstance(got, dict):
            candidates.append(got)
            cand_masks.append(cmask)
        elif got == "big":
            # 과대 클러스터(레일 등과 병합된 고리): 더 강한 오프닝으로 분리 시도
            for it in SPLIT_ITERS:
                sub = cv2.morphologyEx(cmask.astype(np.uint8), cv2.MORPH_OPEN,
                                       kernel, iterations=it)
                ns, slab = cv2.connectedComponents(sub, connectivity=8)
                found = False
                for sl in range(1, ns):
                    smask = slab == sl
                    if smask.sum() < MIN_CLUSTER_PX:
                        continue
                    # 침식분 복원: 원 클러스터 내부로 제한한 재팽창
                    rec = cv2.dilate(smask.astype(np.uint8), kernel,
                                     iterations=it - MORPH_ITERS).astype(bool)
                    rec &= cmask
                    got2, used = measure(rec, it), rec
                    if got2 == "big":     # 복원이 브리지 재연결 → 침식 코어 사용
                        got2, used = measure(smask, it), smask
                    if isinstance(got2, dict):
                        candidates.append(got2)
                        cand_masks.append(used)
                        found = True
                if found:
                    break

    order = np.argsort([-c["score"] for c in candidates])
    candidates = [candidates[i] for i in order]
    cand_masks = [cand_masks[i] for i in order]
    hook = candidates[0] if candidates else None
    hook_mask = cand_masks[0] if candidates else np.zeros((H, W), bool)

    result = dict(
        session=sess_dir.name,
        n_valid=int(vm.sum()),
        plane=dict(
            normal_toward_camera=[round(float(x), 4) for x in n],
            d_mm=round(float(d), 1),
            camera_height_above_plate_mm=round(float(-d), 1),
            tilt_from_camera_y_axis_deg=round(
                float(np.rad2deg(np.arccos(abs(n[1])))), 1),
            quad_coef=[float(f"{c:.3e}") for c in quad_coef],
            plate_residual_mad_mm=round(plate_mad, 2),
            n_plate_inliers=int(plate_px.sum()),
        ),
        hook_detected=hook is not None,
        hook=hook,
        n_candidates=len(candidates),
        other_candidates=candidates[1:5],
    )
    viz = dict(D=sess["D"], I=sess.get("I"), vm=vm, height=height,
               plate_px=plate_px, hook_mask=hook_mask, band=opened.astype(bool))
    return result, viz


# ---- 시각화 -------------------------------------------------------------------
def save_overlay(viz, out_path):
    D, vm = viz["D"], viz["vm"]
    base = viz["I"] if viz["I"] is not None else D
    b = base.astype(np.float64).copy()
    b[~np.isfinite(b)] = 0
    lo, hi = np.percentile(b[vm], [1, 99]) if vm.any() else (0, 1)
    g = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.cvtColor((g * 200).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    img[viz["plate_px"]] = (0.55 * img[viz["plate_px"]] +
                            0.45 * np.array([150, 150, 150])).astype(np.uint8)
    img[viz["band"]] = (0, 140, 255)                # 그 외 12-100mm 밴드: 주황
    img[viz["hook_mask"]] = (0, 0, 255)             # 고리 클러스터: 빨강
    cv2.imwrite(str(out_path), img)


def save_heightmap(viz, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(np.clip(viz["height"], -120, 120), cmap="turbo",
                   vmin=-120, vmax=120)
    fig.colorbar(im, ax=ax, label="height toward camera side of plate (mm)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def save_scatter(results, out_path):
    ok = [r for r in results if r["hook_detected"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    specs = [("center_cam_mm", "camera frame X/Y (mm)", "X (mm)", "Y (mm)"),
             ("center_plane_mm", "plate-plane frame u/v (mm)", "u (mm)", "v (mm)")]
    for ax, (key, title, xl, yl) in zip(axes, specs):
        xs = [r["hook"][key][0] for r in ok]
        ys = [r["hook"][key][1] for r in ok]
        ax.scatter(xs, ys, c="crimson", s=70, zorder=3)
        for r, x, y in zip(ok, xs, ys):
            ax.annotate(r["session"][-6:], (x, y), fontsize=8,
                        xytext=(5, 5), textcoords="offset points")
        ax.set_title("hook centers - " + title)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
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
    out = {"n_sessions_used": len(ok)}
    for key in ["center_cam_mm", "center_plane_mm"]:
        arr = np.array([r["hook"][key] for r in ok])
        dev = arr - arr.mean(axis=0)
        out[key] = dict(
            mean=[round(float(x), 1) for x in arr.mean(axis=0)],
            std=[round(float(x), 2) for x in arr.std(axis=0)],
            max_abs_dev=[round(float(x), 2) for x in np.abs(dev).max(axis=0)],
            rms_3d=round(float(np.sqrt((dev ** 2).sum(axis=1).mean())), 2),
            max_pairwise_3d=round(float(max(
                np.linalg.norm(a - b) for i, a in enumerate(arr)
                for b in arr[i + 1:])), 2),
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
                       f"{sp.name} height (camera side +) vs plate")
        st = "OK " if res["hook_detected"] else "FAIL"
        pl = res["plane"]
        print(f"[{st}] {sp.name}  valid={res['n_valid']}  "
              f"plate_inl={pl['n_plate_inliers']}  "
              f"cam_h={pl['camera_height_above_plate_mm']:.0f}mm  "
              f"mad={pl['plate_residual_mad_mm']:.1f}mm  "
              f"cand={res['n_candidates']}")
        if res["hook_detected"]:
            h = res["hook"]
            c, p, bb = h["center_cam_mm"], h["center_plane_mm"], h["bbox_plane_mm"]
            print(f"       hook cam=({c[0]:.0f},{c[1]:.0f},{c[2]:.0f})mm "
                  f"plane=(u{p[0]:.0f},v{p[1]:.0f},h{p[2]:.0f})mm  "
                  f"bbox u={bb['u']:.0f} v={bb['v']:.0f} "
                  f"h={bb['h'][0]:.0f}..{bb['h'][1]:.0f}mm  px={h['area_px']}")

    rep = repeatability(results)
    save_scatter(results, OUT_DIR / "hook_centers_scatter.png")
    if rep:
        for key in ["center_cam_mm", "center_plane_mm"]:
            r = rep[key]
            print(f"repeatability [{key}]: std=({r['std'][0]:.1f},"
                  f"{r['std'][1]:.1f},{r['std'][2]:.1f})mm  "
                  f"rms3d={r['rms_3d']:.1f}mm  maxpair={r['max_pairwise_3d']:.1f}mm")

    payload = dict(
        meta=dict(
            units="mm", date="2026-09-01",
            camera_frame="X right+, Y down+ (image), D forward+",
            height_convention=(
                "plane normal oriented toward the CAMERA side; h>0 means the "
                "point protrudes from the plate toward the camera - this is "
                "the side where the hook physically rises (the far side holds "
                "the cart body / background / slot see-throughs). The camera "
                "sits at h=+camera_height_above_plate_mm."),
            plane_frame=(
                "origin = camera origin projected onto the plate plane, "
                "u = camera X axis projected, v = n x u, h = quad-refined "
                "height (camera side positive)"),
            method_notes=(
                "RANSAC plane (500 iters, 8mm, normal within 65 deg of camera "
                "Y axis because the camera is pitched ~50 deg) followed by a "
                "robust quadratic surface refinement in plane coords to absorb "
                "the wide-angle warp (raw plate planarity ~30mm, refined MAD "
                "~3mm). Hook = largest center-weighted connected component in "
                "the 12-100mm band after 3x3 opening x2, required to touch the "
                "dilated plate-inlier footprint. Oversized merged components "
                "(hook touching a rail) are split by progressively stronger "
                "opening (see split_open_iters). Score = area_px * "
                "exp(-dist_to_image_center/250px) * gaussian size prior "
                "(45mm +/- 35mm on the max in-plane extent, from the 30-60mm "
                "expected hook width)."),
            params=dict(ransac_iters=RANSAC_ITERS,
                        inlier_thresh_mm=RANSAC_THRESH,
                        normal_max_tilt_deg=NORMAL_MAX_TILT_DEG,
                        hook_height_band_mm=[HOOK_H_MIN, HOOK_H_MAX],
                        max_range_mm=MAX_RANGE_MM,
                        min_cluster_px=MIN_CLUSTER_PX,
                        extent_mm=[MIN_EXTENT_MM, MAX_EXTENT_MM],
                        morph_open_iters=MORPH_ITERS),
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
