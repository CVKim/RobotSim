# -*- coding: utf-8 -*-
"""인식→정책 통합 데모 v2: 실제 ToF 적재 상태 → 학습된 PPO의 '다음 배치 위치' 제안.

v1 → v2 개선 (docs/22_통합데모_개선.md):
  1. 좌표 왜곡 제거: 팔레트 영역을 그리드에 늘려 맞추던 방식 → 실측 25mm/셀 등척 매핑
  2. 빈 슬롯 모드: 제안을 '바닥층(지지높이 0)' 배치로 제한해 온탑 제안과의 혼동 제거
  3. 역투영 안정화: 최근접 픽셀 스냅 → 검출 박스 (mm↔px) 대응점 최소자승 어파인 변환
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from binpick_topface import detect_boxes, render_overlay  # noqa: E402
from mim_loader import load_session  # noqa: E402
from palletize_env import BOX_H_MU, CELL_MM, GRID, PalletizeEnv  # noqa: E402

try:
    from local_paths import BINPICK_DIR
except ImportError:
    import os
    BINPICK_DIR = os.environ["BINPICK_DIR"]

OUT = Path(r"E:\Robot_Sim\explore\integration")
OUT.mkdir(parents=True, exist_ok=True)

REF_SESSION = "126013011364085"  # 1층 만재(12박스) — 팔레트 기준 레이아웃


def detect_with_centers(sess):
    top_d, mask, boxes = detect_boxes(sess)
    X, Y = sess["X"], sess["Y"]
    out = []
    for b in boxes:
        pts = cv2.boxPoints(b["rect_px"]).astype(np.int32)
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        out.append({"mm": (float(X[cy, cx]), float(Y[cy, cx])),
                    "px": (cx, cy), "box": b})
    return top_d, mask, boxes, out


def build_state(target_session_dir):
    """실제 장면 → PalletizeEnv 상태 (등척 25mm/셀, 원점 = 기준 레이아웃 좌하단)."""
    ref = load_session(Path(BINPICK_DIR) / REF_SESSION)
    _, _, _, ref_c = detect_with_centers(ref)
    xs = [c["mm"][0] for c in ref_c]
    ys = [c["mm"][1] for c in ref_c]
    x0 = min(xs) - 293 / 2 - CELL_MM   # 좌하단 원점 (박스 반치수 + 1셀 여유)
    y0 = min(ys) - 219 / 2 - CELL_MM

    sess = load_session(target_session_dir)
    top_d, mask, boxes, cur = detect_with_centers(sess)

    env = PalletizeEnv(max_boxes=120)
    env.reset(seed=0)
    env.heightmap[:] = 0.0

    # 등척 매핑: 셀 = 실제 25mm (왜곡 없음)
    for c in cur:
        gx = int((c["mm"][0] - x0) / CELL_MM)
        gy = int((c["mm"][1] - y0) / CELL_MM)
        L, W = c["box"]["dims_mm"]
        nx = max(int(round(L / CELL_MM / 2)), 1)
        ny = max(int(round(W / CELL_MM / 2)), 1)
        env.heightmap[max(gx - nx, 0):min(gx + nx, GRID),
                      max(gy - ny, 0):min(gy + ny, GRID)] = BOX_H_MU
    env.box = np.array([293.0, 219.0, BOX_H_MU])
    return env, (x0, y0), (sess, top_d, mask, boxes, cur)


def fit_affine_mm_to_px(cur):
    """검출 박스들의 (mm ↔ px) 대응점으로 2x3 어파인 최소자승 피팅."""
    src = np.array([c["mm"] for c in cur], np.float64)
    dst = np.array([c["px"] for c in cur], np.float64)
    A = np.hstack([src, np.ones((len(src), 1))])
    M, *_ = np.linalg.lstsq(A, dst, rcond=None)  # (3,2)
    return M


def suggest_and_render(session_name, model, floor_only=True):
    env, (x0, y0), (sess, top_d, mask_img, boxes, cur) = build_state(
        Path(BINPICK_DIR) / session_name)

    hm = env.heightmap.astype(np.float32).ravel() / 1500.0
    obs = np.concatenate([hm, np.asarray(env.box, np.float32) / 1500.0])

    amask = env.action_mask()                     # (44,44,2)
    if floor_only:                                # 개선 2: 빈 슬롯(바닥층) 배치만 제안
        for rot in (0, 1):
            hmax, _ = env._placement_maps(rot)
            amask[:, :, rot] &= (hmax == 0)
    flat = amask.ravel()
    if not flat.any():
        return session_name, None, len(cur)
    act, _ = model.predict(obs, action_masks=flat, deterministic=True)
    gx, gy, rot = np.unravel_index(int(act), (GRID, GRID, 2))

    bx = 293.0 if rot == 0 else 219.0
    by = 219.0 if rot == 0 else 293.0
    px0, py0 = x0 + gx * CELL_MM, y0 + gy * CELL_MM
    corners_mm = np.array([(px0, py0), (px0 + bx, py0),
                           (px0 + bx, py0 + by), (px0, py0 + by)])

    # 개선 3: 어파인 역투영 (검출 박스 대응점 기반 — 무효 픽셀과 무관)
    M = fit_affine_mm_to_px(cur if len(cur) >= 3 else cur * 3)
    pix = (np.hstack([corners_mm, np.ones((4, 1))]) @ M).astype(np.int32)

    out_path = OUT / f"{session_name}_suggest_v2.png"
    render_overlay(sess, top_d, mask_img, boxes, out_path)
    img = cv2.imread(str(out_path))
    overlay = img.copy()
    cv2.fillPoly(overlay, [pix], (60, 200, 255))
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    cv2.polylines(img, [pix], True, (0, 215, 255), 2)
    cx, cy = pix.mean(axis=0).astype(int)
    cv2.putText(img, "PLACE", (cx - 34, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4)
    cv2.putText(img, "PLACE", (cx - 34, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (60, 220, 255), 2)
    cv2.putText(img, "GREEN = detected boxes (pick targets) | AMBER = where PPO would PLACE the next box",
                (10, 448), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 255), 1)
    cv2.putText(img, f"palletizing policy suggestion: grid=({gx},{gy}) rot={90*int(rot)}deg (empty floor slots only)",
                (10, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.imwrite(str(out_path), img)
    return session_name, (int(gx), int(gy), int(rot)), len(cur)


if __name__ == "__main__":
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(r"E:\Robot_Sim\runs\palletize_ppo\final", device="cpu")
    for name in ["126013011372695", "126013011374496", "126013011380406"]:
        s, a, n = suggest_and_render(name, model)
        print(f"{s}: boxes={n} -> suggestion grid={a}")
