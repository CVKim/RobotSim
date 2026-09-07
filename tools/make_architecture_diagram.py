# -*- coding: utf-8 -*-
"""한 장 요약 다이어그램 생성기.

한 개의 레이아웃 스펙(GROUPS / ARROWS)에서 두 산출물을 뽑는다:
  assets/architecture.svg                GitHub README 에서 바로 렌더 (외부 의존 없음)
  docs/diagrams/architecture.excalidraw  excalidraw.com 에서 열어 손으로 편집

둘을 같은 스펙에서 만들어야 문서와 그림이 어긋나지 않는다. 수치가 바뀌면 이 파일의 텍스트만 고치고 다시 실행.
    python tools/make_architecture_diagram.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
W, H = 1180, 600
FONT = '"Segoe UI", "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", Arial, sans-serif'

TITLE = "Robot_Sim — 실측 ToF → 인식 패키지 → 로봇 좌표 → 셀 트윈 폐루프 → ROS2"

# style: normal | new | todo
GROUPS = [
    dict(id="data", title="실데이터 (비공개, 로컬 전용)", x=30, y=60, w=250, h=165, style="normal", lines=[
        "ToF .mim  X/Y/D/I · 640×480 · mm",
        "30프레임 · 단일 SKU 293×219×283",
        "RGB 5MP — 육안 대조에만 사용",
        "노이즈 실측 σ(mm) = 180.3·I^−0.805",
    ]),
    dict(id="learn", title="학습 · 시뮬레이션", x=30, y=255, w=250, h=200, style="normal", lines=[
        "sim2real  0.00 → 0.43 → 0.99 (mAP50)",
        "YOLO11n-seg 0.99 · 무효 15%↑ 붕괴",
        "RL MaskablePPO 64.7±0.3 (+14.3%)",
        "   mask 제거 43.2 < 휴리스틱 56.6",
        "BC/DART 100% · SmolVLA VRAM 4.7GB",
    ]),
    dict(id="pkg", title="인식 패키지  robotsim_perception  (numpy / opencv)", x=350, y=60, w=440, h=340,
         style="normal", cards=[
        dict(title="geometry.py — 검출", x=362, y=95, w=205, h=135, lines=[
            "층 깊이 히스토그램",
            "I-채널 에지로 이음새 분리",
            "X/Y mm 맵 minAreaRect 치수",
            "v2 격자 보완·층 선택 → 167/167",
        ]),
        dict(title="pose.py — 로봇 좌표", x=577, y=95, w=205, h=135, lines=[
            "평면 피팅: 중심·법선·기울기",
            "T_base_cam (4×4) → base_link",
            "6-DoF 픽 포즈: 접근·yaw·pre/post",
            "석션 풋프린트 판정",
        ]),
        dict(title="runtime.py — 판정", x=362, y=250, w=205, h=135, lines=[
            "OK / RETAKE / LOW_CONFIDENCE",
            "LAYER_EMPTY / NO_SURFACE",
            "임계 JSON 외부화",
            "드리프트 감시 (정적 배경)",
        ]),
        dict(title="planner.py — 픽 순서", x=577, y=250, w=205, h=135, lines=[
            "열 스캔 규칙 (실제 81% 일치)",
            "JSON 스키마 1.0 · CLI",
            "지연 v1 52 ms / v2 340 ms",
            "pytest 51 (46개 합성만으로)",
        ]),
    ]),
    dict(id="ros", title="ROS2 Humble (WSL2)  노드 6개: 인식 · 대차 · 제어 · 트윈 브리지 · 도킹 — 신규", x=350, y=430, w=440, h=140, style="new", lines=[
        "perception_node: /tof/points · pick_poses · TF base_link → tof_optical (정적)",
        "pick_executor: pick_poses → /robot/target_poses 3점 · 완료 보고 대기",
        "cart_node: hook_pose · 동적 TF → dock_node+agv_sim 도킹  ·  twin_bridge: 트윈 (TCP)",
        "status JSON · rviz2 · launch_testing 통합 4 · ros2 bag · colcon test 38",
    ]),
    dict(id="twin", title="MuJoCo 셀 디지털 트윈", x=860, y=60, w=300, h=160, style="normal", lines=[
        "실측 역산 지오메트리 (2970 vs 2973 mm)",
        "절대 정답: 중심오차 8.5 / 9.8 mm",
        "폐루프: 인식 43% → 층 사전 78% (정답 100%)",
        "UR10e IK·충돌: 167포즈 도달 100% · ROS 구동 12/12",
    ]),
    dict(id="verify", title="검증", x=860, y=250, w=300, h=150, style="normal", lines=[
        "30세션 전체 파리티  v1 152 / v2 167",
        "교란 격자 3시드 · 부트스트랩 95% CI",
        "찾은 결함 5건 수정 · 기각 시도 2건 기록",
        "트윈 자체 결함 2건 실측 대조로 판정",
    ]),
    dict(id="todo", title="미구현 — 하드웨어 · 데이터 필요", x=860, y=430, w=300, h=140, style="todo", lines=[
        "핸드아이 외참 실측값 · 경로 계획(장애물 회피)",
        "줄자 GT · 혼합 SKU · 다른 조명",
        "트윈 박스 이음새 대비 (실측 반사율 편차)",
        "실로봇 연결 (여기까지는 시뮬·재생)",
    ]),
]

# (x1,y1) -> (x2,y2), label, dashed
ARROWS = [
    dict(a=(280, 140), b=(350, 140), label="X/Y/D/I mm", dashed=False),
    dict(a=(155, 225), b=(155, 255), label="σ(I) · 치수 시드", dashed=False),
    dict(a=(790, 115), b=(860, 115), label="6-DoF 포즈", dashed=False),
    dict(a=(860, 170), b=(790, 170), label="렌더 + 정답", dashed=False),
    dict(a=(790, 325), b=(860, 325), label="파리티·강건성", dashed=False),
    dict(a=(570, 400), b=(570, 430), label="Box · PickPose · Decision", dashed=False),
    dict(a=(280, 350), b=(350, 350), label="seg 대안", dashed=True),
    dict(a=(790, 500), b=(860, 500), label="트윈 ↔ ROS (TCP)", dashed=False),
]

STYLE = {
    "normal": dict(fill="#f7f8fa", stroke="#9aa3ad", title="#1f2937", dash=""),
    "new":    dict(fill="#eef5ff", stroke="#2563eb", title="#1d4ed8", dash=""),
    "todo":   dict(fill="#fff5f5", stroke="#d93025", title="#b91c1c", dash="6,4"),
}


# ----------------------------------------------------------------------------- SVG
def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg() -> str:
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family=\'{FONT}\'>',
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
         '<path d="M0,0 L10,5 L0,10 z" fill="#4b5563"/></marker></defs>',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="30" y="36" font-size="17" font-weight="700" fill="#111827">{_esc(TITLE)}</text>']

    def card(x, y, w, h, title, lines, fill="#ffffff", stroke="#cbd5e1", tcol="#111827", dash=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="1.2"{d}/>')
        o.append(f'<text x="{x + 10}" y="{y + 19}" font-size="12.5" font-weight="700" fill="{tcol}">{_esc(title)}</text>')
        for i, ln in enumerate(lines):
            o.append(f'<text x="{x + 10}" y="{y + 40 + 17 * i}" font-size="11.5" fill="#374151">{_esc(ln)}</text>')

    for g in GROUPS:
        st = STYLE[g["style"]]
        d = f' stroke-dasharray="{st["dash"]}"' if st["dash"] else ""
        o.append(f'<rect x="{g["x"]}" y="{g["y"]}" width="{g["w"]}" height="{g["h"]}" rx="9" '
                 f'fill="{st["fill"]}" stroke="{st["stroke"]}" stroke-width="1.5"{d}/>')
        o.append(f'<text x="{g["x"] + 12}" y="{g["y"] + 21}" font-size="13" font-weight="700" '
                 f'fill="{st["title"]}">{_esc(g["title"])}</text>')
        for i, ln in enumerate(g.get("lines", [])):
            o.append(f'<text x="{g["x"] + 12}" y="{g["y"] + 46 + 19 * i}" font-size="12" fill="#1f2937">{_esc(ln)}</text>')
        for c in g.get("cards", []):
            card(c["x"], c["y"], c["w"], c["h"], c["title"], c["lines"])

    for ar in ARROWS:
        (x1, y1), (x2, y2) = ar["a"], ar["b"]
        d = ' stroke-dasharray="5,4"' if ar["dashed"] else ""
        o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#4b5563" stroke-width="1.6"{d} '
                 f'marker-end="url(#ah)"/>')
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        vertical = abs(x2 - x1) < abs(y2 - y1)
        lx, ly, anchor = (mx + 8, my + 4, "start") if vertical else (mx, my - 7, "middle")
        o.append(f'<text x="{lx}" y="{ly}" font-size="10" fill="#374151" text-anchor="{anchor}" '
                 f'paint-order="stroke" stroke="#ffffff" stroke-width="4">{_esc(ar["label"])}</text>')
    o.append("</svg>")
    return "\n".join(o)


# ----------------------------------------------------------------------------- Excalidraw
_rng = random.Random(7)


def _base(t, x, y, w, h, **kw):
    e = dict(type=t, version=1, versionNonce=_rng.randint(1, 2**31 - 1), isDeleted=False,
             id=f"{t}-{_rng.randint(1, 10**9)}", fillStyle="solid", strokeWidth=1, strokeStyle="solid",
             roughness=1, opacity=100, angle=0, x=x, y=y, strokeColor="#1e1e1e", backgroundColor="transparent",
             width=w, height=h, seed=_rng.randint(1, 2**31 - 1), groupIds=[], frameId=None,
             roundness={"type": 3} if t == "rectangle" else None, boundElements=[], updated=1, link=None,
             locked=False)
    e.update(kw)
    return e


def _text(x, y, text, size=14, color="#1e1e1e", bold=False, group=None):
    lines = text.split("\n")
    width = max(sum(1.0 if ord(ch) > 0x2E80 else 0.58 for ch in ln) for ln in lines) * size
    e = _base("text", x, y, round(width, 1), round(size * 1.25 * len(lines), 1), text=text, fontSize=size,
              fontFamily=1, textAlign="left", verticalAlign="top", baseline=size, containerId=None,
              originalText=text, lineHeight=1.25, strokeColor=color, roundness=None)
    if bold:
        e["strokeWidth"] = 2
    if group:
        e["groupIds"] = [group]
    return e


def excalidraw() -> dict:
    els = [_text(30, 18, TITLE, size=20, bold=True)]
    for g in GROUPS:
        st = STYLE[g["style"]]
        grp = f"g-{g['id']}"
        els.append(_base("rectangle", g["x"], g["y"], g["w"], g["h"], strokeColor=st["stroke"],
                         backgroundColor=st["fill"], strokeStyle="dashed" if st["dash"] else "solid",
                         groupIds=[grp]))
        els.append(_text(g["x"] + 12, g["y"] + 8, g["title"], size=15, color=st["title"], bold=True, group=grp))
        if g.get("lines"):
            els.append(_text(g["x"] + 12, g["y"] + 36, "\n".join(g["lines"]), size=13, group=grp))
        for c in g.get("cards", []):
            els.append(_base("rectangle", c["x"], c["y"], c["w"], c["h"], strokeColor="#94a3b8",
                             backgroundColor="#ffffff", groupIds=[grp]))
            els.append(_text(c["x"] + 10, c["y"] + 7, c["title"], size=14, bold=True, group=grp))
            els.append(_text(c["x"] + 10, c["y"] + 32, "\n".join(c["lines"]), size=12.5, group=grp))
    for ar in ARROWS:
        (x1, y1), (x2, y2) = ar["a"], ar["b"]
        e = _base("arrow", x1, y1, abs(x2 - x1), abs(y2 - y1), points=[[0, 0], [x2 - x1, y2 - y1]],
                  lastCommittedPoint=None, startBinding=None, endBinding=None, startArrowhead=None,
                  endArrowhead="arrow", strokeStyle="dashed" if ar["dashed"] else "solid",
                  roundness={"type": 2}, strokeColor="#4b5563")
        els.append(e)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        vertical = abs(x2 - x1) < abs(y2 - y1)
        els.append(_text(mx + 8 if vertical else mx - 40, my - 6 if vertical else my - 22, ar["label"],
                         size=11, color="#374151"))
    return {"type": "excalidraw", "version": 2, "source": "tools/make_architecture_diagram.py",
            "elements": els, "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None}, "files": {}}


def main():
    out_svg = ROOT / "assets" / "architecture.svg"
    out_exc = ROOT / "docs" / "diagrams" / "architecture.excalidraw"
    out_svg.write_text(svg(), encoding="utf-8")
    out_exc.write_text(json.dumps(excalidraw(), ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", out_svg, out_exc)


if __name__ == "__main__":
    main()
