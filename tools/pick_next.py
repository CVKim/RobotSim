# -*- coding: utf-8 -*-
"""PICK NEXT 제안: 목적지(왼쪽 스택)까지 로봇 동선이 최소인 픽 대상 선택 + 실제와 비교.

배경(사용자 제안): 이적재 공정에서 사이클타임의 핵심은 '어디에 놓을까'보다
'다음에 어떤 박스를 집어야 동선이 최소인가'. 세션 차분으로 복원한 실제 픽 순서로
그리디 최근접 예측을 평가한 결과: 정확 일치 50%, top-2 이내 81%, 실제 픽의
거리순위 평균 1.69 (n=16 전이) — 실제 시스템이 최소 동선 원칙을 따름을 데이터로 확인.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from binpick_topface import detect_boxes, render_overlay  # noqa: E402
from mim_loader import load_session  # noqa: E402

try:
    from local_paths import BINPICK_DIR
except ImportError:
    import os
    BINPICK_DIR = os.environ["BINPICK_DIR"]

OUT = Path(r"E:\Robot_Sim\explore\integration")
DEST_PX = (120, 235)  # 목적지 스택 ROI 중심 (픽셀)


def render_pick_next(session_name):
    pp = json.load(open(r"E:\Robot_Sim\explore\pickpoints\pickpoints.json", encoding="utf-8"))
    ev = json.load(open(r"E:\Robot_Sim\explore\pickpoints\pick_order_eval.json", encoding="utf-8"))
    dest_xy = ev["dest_xy"]

    sess_info = pp["sessions"][session_name]
    slot = pp["slots"][session_name]
    ref_picks = pp["sessions"][slot["ref"]]["picks"]

    present = [i for i, o in enumerate(slot["occupied"]) if o]
    dist = {i: float(np.hypot(ref_picks[i]["center_mm"][0] - dest_xy[0],
                              ref_picks[i]["center_mm"][1] - dest_xy[1])) for i in present}
    # 열 스캔 규칙 (실제 픽 순서 마이닝: 그리디 50% -> 열 스캔 81%):
    # 목적지 최근접 박스가 속한 열(|dx|<80mm)에서 가장 먼 것부터 픽
    P = min(present, key=lambda i: dist[i])
    col = [i for i in present
           if abs(ref_picks[i]["center_mm"][0] - ref_picks[P]["center_mm"][0]) < 80]
    pred = max(col, key=lambda i: dist[i])
    actual = next((t["actual"] for t in ev["transitions"] if t["from"] == session_name[-6:]), None)

    sess = load_session(Path(BINPICK_DIR) / session_name)
    top_d, mask_img, boxes = detect_boxes(sess)
    out_path = OUT / f"{session_name}_picknext.png"
    render_overlay(sess, top_d, mask_img, boxes, out_path)
    img = cv2.imread(str(out_path))

    def slot_px(i):
        # 현재 세션 박스 중 슬롯 i 기준중심에 가장 가까운 박스의 픽셀 중심
        rc = ref_picks[i]["center_mm"][:2]
        best = min(sess_info["picks"],
                   key=lambda p: np.hypot(p["center_mm"][0] - rc[0], p["center_mm"][1] - rc[1]))
        return tuple(best["center_px"])

    px = slot_px(pred)
    cv2.arrowedLine(img, px, DEST_PX, (60, 220, 255), 2, tipLength=0.06)
    cv2.circle(img, px, 26, (60, 220, 255), 3)
    cv2.putText(img, "PICK", (px[0] - 24, px[1] - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
    cv2.putText(img, "PICK", (px[0] - 24, px[1] - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 220, 255), 2)
    mid = ((px[0] + DEST_PX[0]) // 2, (px[1] + DEST_PX[1]) // 2 - 8)
    cv2.putText(img, f"{dist[pred]:.0f}mm", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 255), 1)

    if actual is not None and actual != pred:
        apx = slot_px(actual)
        cv2.circle(img, apx, 26, (255, 80, 255), 2)
        cv2.putText(img, "ACTUAL", (apx[0] - 34, apx[1] + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 255), 1)
    elif actual == pred:
        cv2.putText(img, "= ACTUAL", (px[0] - 34, px[1] + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 120), 1)

    cv2.putText(img, "min-motion pick suggestion -> DEST | column-scan rule: 81% exact match with real pick order (greedy 50%)",
                (10, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    cv2.imwrite(str(out_path), img)
    return pred, actual, round(dist[pred])


if __name__ == "__main__":
    for name in ["126013011365017", "126013011372695", "126013011384643"]:
        print(name, render_pick_next(name))
