# -*- coding: utf-8 -*-
"""인식→정책 통합 데모: 실제 ToF 장면의 적재 상태를 학습된 PPO에 넣어
'다음 박스 배치 위치'를 제안받고, 실제 depth 이미지 위에 오버레이한다.

파이프라인:
  1. 실제 세션 → 박스 검출(binpick_topface) + 기준 레이아웃 슬롯 매칭
  2. 기준 레이아웃 bbox = 팔레트 영역으로 보고 PalletizeEnv 그리드에 상태 이식
     (빈 슬롯 = 바닥높이, 점유 슬롯 = +283mm)
  3. 학습된 MaskablePPO로 다음 박스(실측 평균 치수) 배치 액션 추론
  4. 제안 위치를 X/Y 좌표맵 역탐색으로 이미지 픽셀에 매핑해 시각화
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


def box_centers_mm(sess):
    _, _, boxes = detect_boxes(sess)
    X, Y = sess["X"], sess["Y"]
    out = []
    for b in boxes:
        pts = cv2.boxPoints(b["rect_px"]).astype(np.int32)
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        out.append((float(X[cy, cx]), float(Y[cy, cx]), b))
    return out


def build_state(target_session_dir):
    """실제 장면 → (PalletizeEnv 상태 이식, 팔레트 원점/스케일, 세션 데이터)."""
    ref = load_session(Path(BINPICK_DIR) / REF_SESSION)
    ref_centers = box_centers_mm(ref)
    xs = [c[0] for c in ref_centers]
    ys = [c[1] for c in ref_centers]
    # 팔레트 영역: 기준 레이아웃 bbox + 박스 반치수 여유
    x0, y0 = min(xs) - 150, min(ys) - 115
    x1, y1 = max(xs) + 150, max(ys) + 115

    sess = load_session(target_session_dir)
    cur_centers = box_centers_mm(sess)

    env = PalletizeEnv(max_boxes=120)
    env.reset(seed=0)
    env.heightmap[:] = 0.0
    scale_x = (x1 - x0) / (GRID * CELL_MM)
    scale_y = (y1 - y0) / (GRID * CELL_MM)

    def to_grid(x, y):
        gx = int((x - x0) / (x1 - x0) * GRID)
        gy = int((y - y0) / (y1 - y0) * GRID)
        return np.clip(gx, 0, GRID - 1), np.clip(gy, 0, GRID - 1)

    # 점유 슬롯: 박스 상면 footprint를 그리드에 +283mm로 마킹
    nx = max(int(293 / (scale_x * CELL_MM) / 2), 1)
    ny = max(int(219 / (scale_y * CELL_MM) / 2), 1)
    for x, y, b in cur_centers:
        gx, gy = to_grid(x, y)
        env.heightmap[max(gx - nx, 0):gx + nx, max(gy - ny, 0):gy + ny] = BOX_H_MU
    env.box = np.array([293.0, 219.0, BOX_H_MU])
    return env, (x0, y0, x1, y1), sess, cur_centers


def suggest_and_render(session_name, model):
    sdir = Path(BINPICK_DIR) / session_name
    env, (x0, y0, x1, y1), sess, cur = build_state(sdir)

    hm = env.heightmap.astype(np.float32).ravel() / 1500.0
    obs = np.concatenate([hm, np.asarray(env.box, np.float32) / 1500.0])
    mask = env.action_mask().ravel()
    act, _ = model.predict(obs, action_masks=mask, deterministic=True)
    gx, gy, rot = np.unravel_index(int(act), (GRID, GRID, 2))

    # 그리드 → 팔레트 mm → 카메라 XY mm
    bx = 293.0 if rot == 0 else 219.0
    by = 219.0 if rot == 0 else 293.0
    px = x0 + (gx * CELL_MM) / (GRID * CELL_MM) * (x1 - x0)
    py = y0 + (gy * CELL_MM) / (GRID * CELL_MM) * (y1 - y0)
    corners_mm = [(px, py), (px + bx, py), (px + bx, py + by), (px, py + by)]

    # 카메라 XY → 픽셀: X/Y 맵 최근접 역탐색 (4px 격자 코스 서치)
    X, Y = sess["X"], sess["Y"]
    step = 4
    Xs, Ys = X[::step, ::step], Y[::step, ::step]
    pix = []
    for (mx, my) in corners_mm:
        d2 = (Xs - mx) ** 2 + (Ys - my) ** 2
        iy, ix = np.unravel_index(np.argmin(d2), d2.shape)
        pix.append((ix * step, iy * step))
    pix = np.array(pix, np.int32)

    top_d, mask_img, boxes = detect_boxes(sess)
    out_path = OUT / f"{session_name}_suggest.png"
    render_overlay(sess, top_d, mask_img, boxes, out_path)
    img = cv2.imread(str(out_path))
    overlay = img.copy()
    cv2.fillPoly(overlay, [pix], (0, 220, 60))
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    cv2.polylines(img, [pix], True, (0, 255, 80), 2)
    cx, cy = pix.mean(axis=0).astype(int)
    cv2.putText(img, "NEXT", (cx - 28, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"PPO suggests: grid=({gx},{gy}) rot={90*int(rot)}deg",
                (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.imwrite(str(out_path), img)
    return session_name, (int(gx), int(gy), int(rot)), len(cur)


if __name__ == "__main__":
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(r"E:\Robot_Sim\runs\palletize_ppo\final", device="cpu")
    # 픽킹이 진행돼 빈 슬롯이 있는 세션들로 데모
    for name in ["126013011372695", "126013011374496", "126013011380406"]:
        s, a, n = suggest_and_render(name, model)
        print(f"{s}: boxes_present={n} -> PPO next placement grid={a}")
