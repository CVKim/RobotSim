# -*- coding: utf-8 -*-
"""로봇 관련 핵심만 추린 2장 요약 PDF (A4 가로). 로컬 산출물 — docs/research/ (gitignore).

    python tools/make_summary_pdf.py  ->  docs/research/Robot_Sim_요약_2p.pdf (+ 페이지 PNG 미리보기)

1장: 한 줄 소개 + 워크플로 그림(데이터 → 인식 → 로봇 좌표 → 트윈 폐루프 → ROS2, 대차 갈래) + 핵심 수치 타일 6개 + 한계 4줄.
2장: 결과 그림 6장. 캡션은 화면에 보이는 것부터 두 줄 안에.
글은 되도록 줄이고 그림과 수치로 보여 준다. 남기는 문장은 기호 나열 대신 문장으로 쓴다.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "assets"
OUT = ROOT / "docs" / "research" / "Robot_Sim_요약_2p.pdf"
PAGE_W, PAGE_H = 11.69, 8.27          # A4 가로 (inch)

for cand in (r"C:\Windows\Fonts\malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
    if Path(cand).exists():
        fm.fontManager.addfont(cand)
        plt.rcParams["font.family"] = fm.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, ACC, RED, RULE = "#111827", "#4b5563", "#1d4ed8", "#b91c1c", "#d1d5db"


def _clean(s: str) -> str:
    # Malgun Gothic 에 없는 글리프 치환 (U+2212 MINUS 등)
    return s.replace("−", "-").replace("≥", ">=").replace("≤", "<=")


def _wrap(s: str, width_frac: float, size: float) -> str:
    """페이지 폭 비율 width_frac 안에 들어가게 줄바꿈. 한글 혼용 기준 보수적 폭 추정."""
    width_in = width_frac * PAGE_W
    chars = max(int(width_in * 72 / (size * 0.78)), 20)
    return "\n".join(textwrap.fill(line, chars) for line in _clean(s).split("\n"))


def text(fig, x, y, s, size=9.5, color=INK, weight="normal", ha="left", va="top", wrap=None):
    if wrap:
        s = _wrap(s, wrap, size)
    else:
        s = _clean(s)
    fig.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, linespacing=1.45)


def rule(fig, y, x0=0.04, x1=0.96):
    fig.add_artist(plt.Line2D([x0, x1], [y, y], transform=fig.transFigure, color=RULE, linewidth=0.6))


def image(fig, path, box, title=None, caption=None, caption_size=8.0):
    """box = (x0, y_top, w, h) 페이지 비율. 이미지 비율을 유지해 박스 안 왼쪽 위에 붙이고,
    제목은 이미지 바로 위, 캡션은 이미지 바로 아래에 (박스가 아니라 실제 이미지 기준)."""
    x0, y_top, w, h = box
    if title:
        text(fig, x0, y_top + 0.004, title, 10.5, INK, weight="bold", va="bottom")
    if not Path(path).exists():
        fig.text(x0 + w / 2, y_top - h / 2, f"(missing) {Path(path).name}", ha="center", color=RED)
        return
    im = imread(path)
    ih, iw = im.shape[:2]
    box_w_in, box_h_in = w * PAGE_W, h * PAGE_H
    scale = min(box_w_in / iw, box_h_in / ih)
    w_fit, h_fit = iw * scale / PAGE_W, ih * scale / PAGE_H
    ax = fig.add_axes((x0, y_top - h_fit, w_fit, h_fit))
    ax.axis("off")
    ax.imshow(im)
    if caption:
        text(fig, x0, y_top - h_fit - 0.008, caption, caption_size, MUTED, wrap=w)


# ----------------------------------------------------------------------------- 도형 (워크플로용)
def box(fig, x, y, w, h, fill="#f7f8fa", edge="#9aa3ad", lw=1.0, dash=None):
    """페이지 비율 좌표의 둥근 상자."""
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                       transform=fig.transFigure, facecolor=fill, edgecolor=edge, linewidth=lw,
                       linestyle=(0, (4, 3)) if dash else "solid", zorder=1)
    fig.add_artist(p)
    return p


def arrow(fig, x1, y1, x2, y2, color="#4b5563", label=None, lw=1.2):
    a = FancyArrowPatch((x1, y1), (x2, y2), transform=fig.transFigure, arrowstyle="-|>",
                        mutation_scale=11, color=color, linewidth=lw, zorder=2)
    fig.add_artist(a)
    if label:
        fig.text((x1 + x2) / 2, max(y1, y2) + 0.012, _clean(label), fontsize=7.0, color=MUTED, ha="center")


def stage(fig, x, y, w, h, title, lines, kpi=None, fill="#f7f8fa", edge="#9aa3ad", tcol=INK):
    """워크플로 한 칸: 제목 + 짧은 줄 + 강조 수치."""
    box(fig, x, y, w, h, fill, edge)
    text(fig, x + 0.010, y + h - 0.016, title, 9.6, tcol, weight="bold")
    for i, ln in enumerate(lines):
        text(fig, x + 0.010, y + h - 0.045 - 0.026 * i, ln, 8.0, MUTED)
    if kpi:
        text(fig, x + 0.010, y + 0.012, kpi, 9.0, ACC, weight="bold", va="bottom")


def tile(fig, x, y, w, h, value, label, note):
    """핵심 수치 타일: 큰 숫자 + 무엇인지 + 뜻 한 줄."""
    box(fig, x, y, w, h, "#ffffff", "#cbd5e1")
    fig.text(x + w / 2, y + h - 0.030, _clean(value), fontsize=17, color=ACC, fontweight="bold",
             ha="center", va="center")
    fig.text(x + w / 2, y + h - 0.062, _clean(label), fontsize=8.4, color=INK, ha="center", va="center",
             fontweight="bold")
    fig.text(x + w / 2, y + 0.012, _clean(note), fontsize=7.4, color=MUTED, ha="center", va="bottom")


def page1(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    text(fig, 0.04, 0.965, "Robot_Sim — 공장 ToF 데이터로 만든 로봇 인식 소프트웨어", 15, weight="bold")
    text(fig, 0.04, 0.932,
         "공장에서 찍은 거리(ToF) 카메라 데이터로 두 가지를 만들었다. 팔레트 위 박스를 찾아 로봇이 집을 위치·자세를 내는 것, "
         "대차의 견인 고리 위치를 대차 기준 좌표로 내는 것. 같은 셀을 MuJoCo 로 재현해 인식 결과만으로 집어 옮기는 것까지 닫았고 "
         "ROS2 노드로 감쌌다. 실제 로봇은 없고 원본 데이터는 비공개다.", 9.0, MUTED, wrap=0.92)

    # ---------------------------------------------------------------- 워크플로
    text(fig, 0.04, 0.868, "워크플로 — 데이터에서 로봇 명령까지", 11.5, weight="bold")
    y, h = 0.66, 0.185
    xs = [0.040, 0.234, 0.428, 0.622, 0.816]
    w = 0.164
    stages = [
        ("① 실측 ToF", ["30장면 · 단일 SKU", "X/Y/거리/강도 640×480", "노이즈 실측으로 모델링"], "비공개 · 로컬 전용"),
        ("② 인식 (numpy/opencv)", ["층 깊이 + 강도 에지", "격자 보완으로 빈 칸 채움", "판정: OK/재촬영/저신뢰"], "167 / 167 박스"),
        ("③ 로봇 좌표", ["평면 피팅 → 중심·법선", "카메라→로봇 4×4 변환", "6-DoF 픽 포즈(요 포함)"], "중심 오차 8.5 mm"),
        ("④ 셀 트윈 폐루프", ["MuJoCo 로 셀 재현", "인식 결과만으로 집기", "UR10e IK·충돌·속도"], "인식 구동 78%"),
        ("⑤ ROS2 (Humble)", ["노드 6개 · rviz2", "픽 실행 액션 · TF", "launch_testing 통합 5"], "12박스 12/12"),
    ]
    for i, (x, (t, lines, kpi)) in enumerate(zip(xs, stages)):
        last = i == len(stages) - 1
        stage(fig, x, y, w, h, t, lines, kpi,
              fill="#eef5ff" if last else "#f7f8fa", edge="#2563eb" if last else "#9aa3ad",
              tcol=ACC if last else INK)
        if i:
            arrow(fig, xs[i - 1] + w, y + h / 2, x, y + h / 2)

    # 대차 갈래 (아래 줄)
    y2, h2 = 0.505, 0.115
    stages2 = [
        ("대차 장면 (실측 4회 · 합성)", ["카메라가 데크를 40° 비스듬히 봄"], None),
        ("고리 좌표", ["데크 평면·림·레일에서 대차 좌표계 추정"], "반복 RMS 3.85 mm"),
        ("AGV 도킹 (시뮬)", ["고리 포즈로 결합 위치까지 주행 · stop-and-go"], "30회 중 27회 결합"),
    ]
    xs2 = [0.040, 0.330, 0.620]
    w2 = 0.258
    for i, (x, (t, lines, kpi)) in enumerate(zip(xs2, stages2)):
        stage(fig, x, y2, w2, h2, t, lines, kpi)
        if i:
            arrow(fig, xs2[i - 1] + w2, y2 + h2 / 2, x, y2 + h2 / 2)
    # 좌표 변환이 맞는지 스스로 검사한 값 (같은 고리를 두 경로로 구해 비교)
    box(fig, 0.878, y2, 0.082, h2, "#ffffff", "#cbd5e1")
    fig.text(0.919, y2 + h2 - 0.032, "0.003 mm", fontsize=11.5, color=ACC, fontweight="bold", ha="center", va="center")
    fig.text(0.919, y2 + h2 - 0.062, _clean("좌표 변환 검사"), fontsize=7.8, color=INK, ha="center", va="center", fontweight="bold")
    fig.text(0.919, y2 + 0.014, _clean("TF 값 vs 인식 값"), fontsize=7.0, color=MUTED, ha="center", va="bottom")

    # ---------------------------------------------------------------- 핵심 수치 타일
    text(fig, 0.04, 0.455, "핵심 수치", 11.5, weight="bold")
    ty, th = 0.285, 0.135
    tw, gap = 0.1455, 0.0164
    tiles = [
        ("167 / 167", "박스 검출 (실측 30장면)", "RGB 사진으로 센 수와 같다"),
        ("8.5 mm", "픽 위치 오차 (트윈)", "시뮬레이터의 진짜 위치와 비교"),
        ("43 → 78 %", "인식만으로 집어 옮기기", "직전 층을 사전으로 쓴 뒤"),
        ("87 → 100 %", "UR10e 도달 (167포즈)", "리니어 트랙 0.6 m 를 더하면"),
        ("27 / 30", "AGV 도킹 결합 (시뮬)", "정답 기준 26 · 평균 9.1초"),
        ("84 + 40", "자동 테스트 (pytest+ROS)", "회사 데이터 없이 78건이 돈다"),
    ]
    for i, (v, lab, note) in enumerate(tiles):
        tile(fig, 0.04 + i * (tw + gap), ty, tw, th, v, lab, note)

    # ---------------------------------------------------------------- 한계
    text(fig, 0.04, 0.245, "한계 (읽고 수치를 쓸 것)", 11.5, weight="bold")
    lim = [
        "박스 종류 하나, 카메라 한 대, 연속 30장면에서만 확인했다.",
        "실측 정답은 검출 결과를 사람이 눈으로 확인한 것이다. 줄자로 잰 정답은 없다.",
        "팔·AGV 는 기구학 시뮬이다. 장애물 회피 경로 계획과 흡착·바퀴 물리는 없다.",
        "실제 로봇과 센서 드라이버는 연결하지 않았다. 카메라·로봇 자리를 트윈이 대신한다.",
        "학습 모델은 무효 픽셀이 15%를 넘으면 무너진다. 기본 경로는 학습 없는 기하 검출이다.",
    ]
    for i, ln in enumerate(lim):
        text(fig, 0.045, 0.212 - 0.031 * i, "· " + ln, 8.6, MUTED)

    # 결과 화면 미리보기 두 장 (자세한 그림은 2장)
    text(fig, 0.545, 0.245, "결과 화면 (자세한 그림은 2장)", 11.5, weight="bold")
    image(fig, A / "detector_v2_recovered.png", (0.545, 0.212, 0.185, 0.150),
          caption="ToF 거리 이미지 위의 검출 결과. 왼쪽이 첫 버전, 오른쪽이 개선 버전.", caption_size=7.2)
    image(fig, A / "ros2_twin_cycle_rviz.png", (0.755, 0.212, 0.205, 0.150),
          caption="ROS2 노드가 트윈의 UR10e 를 구동하는 rviz 화면.", caption_size=7.2)

    text(fig, 0.04, 0.030,
         "github.com/CVKim/RobotSim  ·  수치 원본은 results/*.json  ·  원본 데이터는 비공개(집계 수치와 depth 시각화만 공개)",
         7.8, MUTED)
    pdf.savefig(fig)
    plt.close(fig)


def page2(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    text(fig, 0.04, 0.965, "결과 그림 — 화면에 무엇이 보이는지", 15, weight="bold")
    # 2열 3행 — 가로로 긴 차트가 커야 읽힌다 (3열이면 폭에 눌려 축 글씨가 안 보였다)
    W = 0.435
    X = (0.045, 0.525)
    R = (0.900, 0.610, 0.320)
    H = 0.195
    image(fig, A / "detector_v2_recovered.png", (X[0], R[0], W, H),
          "1. 박스 검출, 첫 버전과 개선 버전",
          "ToF 거리 이미지 위에 찾은 박스 윗면. 왼쪽 열이 첫 버전, 오른쪽 열이 개선 버전이다. 위 줄은 붙은 박스를 합친 경우, "
          "아래 줄은 남은 2개에서 층을 잘못 고른 경우다. 30장면 전체로는 152 → 167개(전부).")
    image(fig, A / "twin_detect_accuracy.png", (X[1], R[0], W, H),
          "2. 시뮬레이터에서 잰 위치 오차",
          "시뮬레이터는 진짜 위치를 아니까 직접 비교할 수 있다. 왼쪽이 중심 오차 분포(평균 8.5 mm), 가운데가 크기 편차"
          "(8~15 mm 작게 잰다), 오른쪽이 개수 리콜·정밀도다. 이 비교에서 검출기 결함 2건을 고쳤다.")
    image(fig, A / "twin_closed_loop.png", (X[0], R[1], W, H),
          "3. 인식 결과만으로 집어 옮기기 (실행기 6종)",
          "왼쪽이 성공률(파랑 = 정답 위치, 주황 = 인식 결과), 가운데가 실패 사유, 오른쪽이 사이클 시간이다. "
          "오른쪽 두 쌍은 직전 층을 사전으로 쓴 결과로 43 → 78%(mocap), 65 → 75%(트랙 팔).")
    image(fig, A / "ros2_twin_cycle_rviz.png", (X[1], R[1], W, H),
          "4. ROS2 노드가 시뮬레이터의 팔을 구동하는 화면",
          "파란 팔이 시뮬레이터의 UR10e 다(관절각을 받아 rviz 가 그린다). 초록·주황 판이 남은 박스, 하늘색 선이 팔 끝 경로, "
          "노란 세 점이 이번 명령이다. 팔이 완료를 보고하면 인식 노드가 다음 프레임을 찍는다. 12박스를 전부 놓았다.")
    image(fig, A / "ros2_rviz_cart_synthetic.png", (X[0], R[2], W, H),
          "5. ROS2 대차 노드를 rviz로 본 화면",
          "파란 판이 데크, 자홍 선이 앞 테두리, 노란 선이 레일, 초록 기둥이 고리, 위쪽 축이 카메라다. 이 셋으로 대차 좌표계를 "
          "매 프레임 새로 잡는다. 카메라 높이가 바뀌어도 고리는 같은 자리(u 36 mm)에 잡힌다.")
    image(fig, A / "hook_v2_overlay.png", (X[1], R[2], W, H),
          "6. 실제 대차에서 고리를 찾은 화면",
          "실제 대차의 ToF 반사 강도 이미지. 빨강이 고리 벽, 주황이 레일, 노랑이 고리 꼭대기, 십자가 추정한 고리 위치다. "
          "네 번 촬영에서 이 위치의 흩어짐(RMS)은 3.85 mm.")
    text(fig, 0.04, 0.03, "자세한 내용: docs/30_결과_상세.md (수치), docs/41_셀_트윈.md (시뮬레이터), docs/42_ROS2_핸즈온.md (ROS2)", 7.8, MUTED)
    pdf.savefig(fig)
    plt.close(fig)


class _PdfAndPng:
    """PdfPages 래퍼 — 페이지마다 PNG 미리보기도 저장 (poppler 없이 검토·공유용)."""

    def __init__(self, pdf, stem):
        self.pdf, self.stem, self.n = pdf, stem, 0

    def savefig(self, fig):
        self.n += 1
        fig.savefig(self.stem.with_name(f"{self.stem.name}_p{self.n}.png"), dpi=160)
        self.pdf.savefig(fig)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        both = _PdfAndPng(pdf, OUT.with_suffix(""))
        page1(both)
        page2(both)
    print("saved", OUT, "+ page PNGs")


if __name__ == "__main__":
    main()
