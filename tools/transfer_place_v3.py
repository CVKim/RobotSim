# -*- coding: utf-8 -*-
"""이적재 PLACE 제안 v3: 목적지(왼쪽) 스택 영역에 다음 박스 배치 위치를 제안.

공정 이해 교정(사용자 확인): 이 데이터는 소스 팔레트(중앙) → 왼쪽 목적지 스택으로
옮겨 쌓는 '이적재' 공정. v2까지는 소스 팔레트의 빈 슬롯을 제안했으나(오해),
v3는 실제 작업 방향대로 목적지 영역의 실측 높이맵 위에 배치를 제안한다.

방법:
  1. 목적지 베이스맵: 스택 반출 후(빈 상태) 세션들의 ROI 픽셀별 중앙값 깊이
  2. 현재 높이맵 = 베이스 - 현재 깊이 → 25mm 그리드로 양자화 (목적지 밖 셀은 벽=1500mm)
  3. 학습된 PPO로 배치 추론 (스택 위 적재 허용 = floor_only 아님)
  4. ROI 픽셀 (X,Y)↔픽셀 어파인으로 제안 영역 역투영
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from binpick_topface import detect_boxes, render_overlay  # noqa: E402
from mim_loader import load_session, valid_mask  # noqa: E402
from palletize_env import BOX_H_MU, CELL_MM, GRID, PalletizeEnv  # noqa: E402

try:
    from local_paths import BINPICK_DIR
except ImportError:
    import os
    BINPICK_DIR = os.environ["BINPICK_DIR"]

OUT = Path(r"E:\Robot_Sim\explore\integration")
ROI = (55, 185, 130, 340)  # 목적지 스택 영역 (x0,x1,y0,y1) — 사용자 표시 기반
EMPTY_SESSIONS = ["126013011432055", "126013011443804", "126013011460258"]


def dest_base_map():
    """빈 상태 세션들로 목적지 픽셀별 베이스 깊이."""
    stack = []
    for n in EMPTY_SESSIONS:
        sess = load_session(Path(BINPICK_DIR) / n)
        D = np.where(valid_mask(sess), sess["D"], np.nan)
        stack.append(D)
    return np.nanmedian(np.stack(stack), axis=0)


def suggest(session_name, model):
    x0, x1, y0, y1 = ROI
    base = dest_base_map()
    sess = load_session(Path(BINPICK_DIR) / session_name)
    D, X, Y = sess["D"], sess["X"], sess["Y"]
    v = valid_mask(sess)

    # 현재 목적지 높이 (mm, 픽셀 단위)
    h_px = np.where(v, base - D, np.nan)
    h_px = np.clip(h_px, 0, 1490)

    # 목적지 영역의 mm 좌표 범위
    rv = v[y0:y1, x0:x1] & np.isfinite(h_px[y0:y1, x0:x1])
    Xr, Yr = X[y0:y1, x0:x1][rv], Y[y0:y1, x0:x1][rv]
    Hr = h_px[y0:y1, x0:x1][rv]
    mx0, my0 = Xr.min(), Yr.min()

    env = PalletizeEnv(max_boxes=120)
    env.reset(seed=0)
    env.heightmap[:] = 1500.0  # 기본: 벽 (목적지 밖 배치 금지)
    gx = ((Xr - mx0) / CELL_MM).astype(int)
    gy = ((Yr - my0) / CELL_MM).astype(int)
    ok = (gx >= 0) & (gx < GRID) & (gy >= 0) & (gy < GRID)
    # 셀별 최대 높이로 양자화
    hm = np.zeros((GRID, GRID)); cnt = np.zeros((GRID, GRID))
    np.maximum.at(hm, (gx[ok], gy[ok]), Hr[ok])
    np.add.at(cnt, (gx[ok], gy[ok]), 1)
    # 층 단위(283mm) 스냅: 표면 노이즈 제거 + 학습 분포(평탄 층)와 일치
    hm = np.round(hm / BOX_H_MU) * BOX_H_MU
    # 목적지 내부 bbox 안의 구멍(무효 픽셀 셀)을 이웃 최대값으로 채움 — 벽 핀홀 방지
    gx0, gx1 = gx[ok].min(), gx[ok].max() + 1
    gy0, gy1 = gy[ok].min(), gy[ok].max() + 1
    interior = np.zeros((GRID, GRID), bool); interior[gx0:gx1, gy0:gy1] = True
    filled = (cnt > 3)
    for _ in range(6):
        hole = interior & ~filled
        if not hole.any():
            break
        dil = cv2.dilate(np.where(filled, hm, 0).astype(np.float32),
                         np.ones((3, 3), np.uint8))
        grow = hole & (cv2.dilate(filled.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        hm[grow] = dil[grow]; filled |= grow
    env.heightmap[interior] = hm[interior]
    env.box = np.array([293.0, 219.0, BOX_H_MU])

    flat = env.action_mask().ravel()
    if not flat.any():
        return session_name, None
    obs = np.concatenate([env.heightmap.astype(np.float32).ravel() / 1500.0,
                          np.asarray(env.box, np.float32) / 1500.0])
    act, _ = model.predict(obs, action_masks=flat, deterministic=True)
    agx, agy, rot = np.unravel_index(int(act), (GRID, GRID, 2))

    bx = 293.0 if rot == 0 else 219.0
    by = 219.0 if rot == 0 else 293.0
    px0, py0 = mx0 + agx * CELL_MM, my0 + agy * CELL_MM
    corners = np.array([(px0, py0), (px0 + bx, py0),
                        (px0 + bx, py0 + by), (px0, py0 + by)])

    # ROI 픽셀 샘플로 mm→px 어파인
    ys_px, xs_px = np.nonzero(rv)
    sel = np.random.default_rng(0).choice(len(xs_px), size=min(400, len(xs_px)), replace=False)
    src = np.stack([Xr[sel], Yr[sel]], axis=-1)
    dst = np.stack([xs_px[sel] + x0, ys_px[sel] + y0], axis=-1).astype(np.float64)
    A = np.hstack([src, np.ones((len(src), 1))])
    M, *_ = np.linalg.lstsq(A, dst, rcond=None)
    pix = (np.hstack([corners, np.ones((4, 1))]) @ M).astype(np.int32)

    # 렌더
    top_d, mask_img, boxes = detect_boxes(sess)
    out_path = OUT / f"{session_name}_transfer_v3.png"
    render_overlay(sess, top_d, mask_img, boxes, out_path)
    img = cv2.imread(str(out_path))
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 180, 60), 1)
    cv2.putText(img, "DEST STACK", (x0 + 2, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 80), 1)
    ov = img.copy(); cv2.fillPoly(ov, [pix], (60, 200, 255))
    img = cv2.addWeighted(ov, 0.4, img, 0.6, 0)
    cv2.polylines(img, [pix], True, (0, 215, 255), 2)
    cx, cy = pix.mean(axis=0).astype(int)
    cv2.putText(img, "PLACE", (cx - 30, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(img, "PLACE", (cx - 30, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 255), 2)
    cur_h = float(np.nanpercentile(h_px[y0:y1, x0:x1], 95))
    cv2.putText(img, f"transfer flow: GREEN source -> AMBER place on DEST (cur stack {cur_h:.0f}mm)",
                (10, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.imwrite(str(out_path), img)
    return session_name, (int(agx), int(agy), int(rot), round(cur_h))


if __name__ == "__main__":
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(r"E:\Robot_Sim\runs\palletize_ppo\final", device="cpu")
    for name in ["126013011372695", "126013011385674"]:
        print(suggest(name, model))
