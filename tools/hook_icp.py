# -*- coding: utf-8 -*-
"""후크 반복성 v2: 세션 간 ICP 정합으로 대차 실이동을 제거한 센서 반복성 측정.

기존(hook_analysis): 후크 중심의 세션 간 편차 u/v 12~24mm — 대차 실이동 포함 추정.
본 분석: 대차 구조물 포인트클라우드를 기준 세션에 ICP 정렬 → 정렬 변환을 후크 중심에
적용 → 잔여 편차 = (센서 노이즈 + 검출 오차 + 비강체성).
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import sys
sys.path.insert(0, str(Path(__file__).parent))
from mim_loader import load_session, valid_mask  # noqa: E402

try:
    from local_paths import DAECHA_DIR
except ImportError:
    import os
    DAECHA_DIR = os.environ["DAECHA_DIR"]

OUT = Path(r"E:\Robot_Sim\explore\hook")


def cloud(sess, d_max=1200.0, voxel=8.0):
    v = valid_mask(sess) & (sess["D"] < d_max)
    pts = np.stack([sess["X"][v], sess["Y"][v], sess["D"][v]], axis=-1)
    # 복셀 다운샘플
    key = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[idx].astype(np.float64)


def icp(src, dst, iters=25, dist_thresh=40.0):
    """rigid ICP: src → dst. 반환 (R, t, rmse)."""
    tree = cKDTree(dst)
    R, t = np.eye(3), np.zeros(3)
    cur = src.copy()
    for _ in range(iters):
        d, j = tree.query(cur, k=1)
        m = d < dist_thresh
        if m.sum() < 100:
            break
        A, B = cur[m], dst[j[m]]
        ca, cb = A.mean(0), B.mean(0)
        H = (A - ca).T @ (B - cb)
        U, _, Vt = np.linalg.svd(H)
        Ri = Vt.T @ U.T
        if np.linalg.det(Ri) < 0:
            Vt[2] *= -1
            Ri = Vt.T @ U.T
        ti = cb - Ri @ ca
        cur = cur @ Ri.T + ti
        R, t = Ri @ R, Ri @ t + ti
    d, _ = tree.query(cur, k=1)
    return R, t, float(np.sqrt(np.mean(np.minimum(d, dist_thresh) ** 2)))


def main():
    hooks = json.loads((OUT / "hook_results.json").read_text(encoding="utf-8"))
    sess_names = [s["session"] for s in hooks["sessions"]] if "sessions" in hooks else None
    if sess_names is None:  # 포맷 유연 대응
        sess_names = [k for k in hooks if k.startswith("126")]

    clouds, centers = {}, {}
    for name in sess_names:
        sess = load_session(Path(DAECHA_DIR) / name)
        clouds[name] = cloud(sess)
        rec = (hooks["sessions"] if "sessions" in hooks else hooks)
        if isinstance(rec, list):
            c = next(r for r in rec if r["session"] == name)["hook"]["center_cam_mm"]
        else:
            c = rec[name]["hook"]["center_cam_mm"]
        centers[name] = np.asarray(c, np.float64)

    ref = sess_names[0]
    aligned = {ref: centers[ref]}
    report = {"ref": ref, "icp": {}}
    for name in sess_names[1:]:
        R, t, rmse = icp(clouds[name], clouds[ref])
        aligned[name] = R @ centers[name] + t
        motion = float(np.linalg.norm((R @ centers[name] + t) - centers[name]))
        report["icp"][name] = {"rmse_mm": round(rmse, 2),
                               "applied_motion_mm": round(motion, 1)}

    raw = np.stack([centers[n] for n in sess_names])
    ali = np.stack([aligned[n] for n in sess_names])
    report["hook_spread_raw_std_mm"] = [round(v, 2) for v in raw.std(0)]
    report["hook_spread_aligned_std_mm"] = [round(v, 2) for v in ali.std(0)]
    report["raw_rms_mm"] = round(float(np.sqrt((raw.std(0) ** 2).sum())), 2)
    report["aligned_rms_mm"] = round(float(np.sqrt((ali.std(0) ** 2).sum())), 2)

    (OUT / "hook_icp.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
