# -*- coding: utf-8 -*-
"""로봇 관련 핵심만 추린 2장 요약 PDF (A4 가로). 로컬 산출물 — docs/research/ (gitignore).

    python tools/make_summary_pdf.py  ->  docs/research/Robot_Sim_요약_2p.pdf (+ 페이지 PNG 미리보기)

1장: 무엇을 만들었나(세 문장) + 다이어그램 + 검증 방법·한계 + 핵심 수치(숫자와 그 뜻)
2장: 결과 그림 6장. 그림마다 "화면에 뭐가 보이는지"와 "숫자가 뜻하는 것"을 풀어 쓴다.
글은 기호를 이어 붙이지 말고 문장으로 쓴다 (읽는 사람이 이 프로젝트를 모른다고 가정).
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


def page1(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    text(fig, 0.04, 0.965, "Robot_Sim — 공장 ToF 데이터로 만든 로봇 인식 소프트웨어", 15, weight="bold")
    text(fig, 0.04, 0.93,
         "공장에서 찍은 ToF(거리) 카메라 데이터 두 종류로 인식 소프트웨어를 만들었다. 하나는 팔레트 위 박스를 찾아 로봇이 집을 위치와 자세를 "
         "계산하는 것, 다른 하나는 대차의 견인 고리 위치를 대차 기준 좌표로 계산하는 것이다. 같은 셀을 시뮬레이터(MuJoCo)에 재현해 인식 결과만으로 "
         "박스를 집어 옮기는 것까지 확인했고, ROS2 노드로 감싸 rviz에서 볼 수 있게 했다. 실제 로봇은 없고, 원본 데이터는 공개하지 않는다.",
         9.0, MUTED, wrap=0.92)

    # ---- 다이어그램 (왼쪽) + 검증 방법 / 한계 (오른쪽)
    image(fig, A / "architecture.png", (0.04, 0.845, 0.92, 0.385))
    rx = 0.625
    text(fig, rx, 0.845, "어떻게 검증했나", 11, weight="bold")
    text(fig, rx, 0.815,
         "1. 실측 30장면의 검출 결과를 RGB 사진과 사람이 대조해 박스 수를 확인했다.\n"
         "2. 시뮬레이터에서는 박스의 진짜 위치를 알 수 있어, 검출 위치 오차를 mm 단위로 직접 쟀다.\n"
         "3. 무효 픽셀, 깊이 노이즈, 큰 결손을 일부러 넣어 검출이 얼마나 버티는지 봤다.\n"
         "4. 자동 테스트 75건(합성 데이터만으로 69건)과 ROS2 테스트 38건(노드를 실제로 띄우는 통합 테스트 4건 포함)이 돈다.",
         8.2, wrap=0.34)
    text(fig, rx, 0.64, "한계", 11, weight="bold", color=RED)
    text(fig, rx, 0.61,
         "1. 박스 종류 하나, 카메라 한 대, 연속 30장면에서만 확인했다.\n"
         "2. 실측 데이터의 정답은 검출기 출력을 사람이 눈으로 확인한 것이다. 줄자로 잰 정답은 없다.\n"
         "3. 시뮬레이터의 팔은 역기구학·충돌 검사·관절 속도까지만 본다. 장애물을 돌아가는 경로 계획과 흡착 물리는 없다.\n"
         "4. 박스가 1~3개만 남으면 층을 잘못 고르던 문제는 직전 프레임의 층을 사전으로 써서 대부분 해결했다. 남은 손실은 시뮬레이터 박스의 이음새 대비 부족.",
         8.2, wrap=0.34)

    # ---- 핵심 수치: 숫자와 그 뜻 (전체 폭)
    rule(fig, 0.455)
    text(fig, 0.04, 0.44, "핵심 수치와 그 뜻", 11, weight="bold")
    cols = (0.04, 0.215, 0.43)
    text(fig, cols[0], 0.41, "항목", 8.2, MUTED, weight="bold")
    text(fig, cols[1], 0.41, "결과", 8.2, MUTED, weight="bold")
    text(fig, cols[2], 0.41, "이 숫자가 뜻하는 것", 8.2, MUTED, weight="bold")
    rows = [
        ("박스 검출 (실측 30장면)",
         "실제 박스 167개 중 167개 검출, 빈 장면 오검출 0",
         "RGB 사진으로 사람이 센 박스 수와 같다. 첫 버전은 152개였는데, 붙어 있는 박스 둘이 하나로 합쳐지는 문제를 "
         "격자 규칙으로 빈 칸을 채워서 고쳤다. 한 장면 처리에 0.34초."),
        ("위치 정확도 (시뮬레이터 48장면)",
         "박스 중심 오차 평균 8.5 mm, 노이즈 포함 9.8 mm",
         "시뮬레이터 안에서는 박스의 진짜 위치를 아니까 직접 비교한 값이다. 실측 데이터에는 이런 정답이 없다. "
         "박스 크기는 실제보다 8~15 mm 작게 잰다."),
        ("인식 결과로 집어 옮기기 (시뮬레이터 60회)",
         "정답 100% vs 인식 43% → 층 사전 78% (mocap), 트랙 팔 65 → 75%",
         "손실 대부분은 박스가 1~3개 남았을 때 층을 잘못 골라 포기한 것. 직전 프레임의 층 깊이를 다음 후보로 쓰면 그 손실이 "
         "거의 사라진다(인식 비용 57 → 22 %p). 팔에 정답을 주면 60개 전부, ROS2 구동은 12개 전부 놓았다."),
        ("팔 도달 범위 (UR10e, 실측 픽 포즈 167개)",
         "고정 받침대 87%, 리니어 트랙을 달면 100%, 무충돌 99~100%",
         "1.3 m 팔 하나로는 두 팔레트(중심 거리 1.19 m)의 반대편 모서리에 못 닿는다. 트랙 0.6 m 를 더하면 전부 닿는다. "
         "관절 속도 한계로 계산한 픽 1회 순수 이동은 3.1초(흡착·해제 대기 0.6초 별도)."),
        ("대차 고리 위치 (실측 4회 촬영)",
         "고리 반복 3.85 mm (면내 3.1 mm) · 도킹 시뮬 30회 중 27회 결합",
         "촬영마다 카메라 높이가 6 cm, 대차 위치가 5 cm까지 달랐는데, 데크 평면과 테두리·레일로 대차 좌표를 잡으면 고리가 같은 자리다. "
         "이 고리 위치로 AGV(시뮬)를 결합 위치까지 몰고 가는 도킹도 닫았다: 평균 9.1초, 최종 측방 오차 3.0 mm."),
        ("팔레타이징 강화학습 (3시드)",
         "학습 정책 64.7개 vs 규칙 56.6개 (14% 더 쌓음)",
         "같은 팔레트에 박스를 몇 개 쌓느냐. 놓을 수 없는 자리를 미리 막는 마스크를 빼면 43.2개로 규칙보다 못하다. "
         "즉 성능의 핵심은 마스크다."),
        ("ROS2 노드 6개 (Humble, WSL2)",
         "빈피킹·대차·제어·트윈 브리지·도킹·AGV 시뮬 노드. 테스트 38건 통과",
         "인식 결과를 로봇 제어기가 받는 표준 형식(포즈, 좌표 변환 TF, 마커)으로 내보낸다. 제어 노드는 픽 포즈를 세 점 궤적으로 "
         "바꿔 보내고 완료 보고를 기다린 뒤 재촬영을 요청한다. 브리지 노드가 그 명령을 시뮬레이터의 팔에 실행시켜 12박스를 전부 놓았다(실패 0)."),
    ]
    y = 0.385
    for k, v, why in rows:                    # 7행: 행 간격 0.05 로 바닥글 위에서 끝나게 (넘치면 여기부터 줄인다)
        text(fig, cols[0], y, k, 8.0, INK, weight="bold", wrap=0.165)
        text(fig, cols[1], y, v, 8.0, INK, wrap=0.205)
        text(fig, cols[2], y, why, 8.0, MUTED, wrap=0.53)
        y -= 0.05
    text(fig, 0.04, 0.018, "github.com/CVKim/RobotSim  ·  수치 원본은 results/*.json  ·  원본 데이터는 비공개(집계 수치와 depth 시각화만 공개)",
         7.8, MUTED)
    pdf.savefig(fig)
    plt.close(fig)


def page2(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    text(fig, 0.04, 0.965, "결과 그림 — 화면에 무엇이 보이는지", 15, weight="bold")
    W, H = 0.285, 0.30
    X = (0.04, 0.355, 0.67)
    # 1번 그림은 거의 정사각형이라 같은 높이를 주면 캡션이 아래 줄 제목을 덮는다 -> 조금 낮게
    image(fig, A / "detector_v2_recovered.png", (X[0], 0.905, W, 0.255),
          "1. 박스 검출, 첫 버전과 개선 버전",
          "ToF 거리 이미지 위에 찾은 박스 윗면을 표시했다. 왼쪽 열이 첫 버전, 오른쪽 열이 개선 버전. 위 줄: 첫 버전은 붙어 있는 박스 둘을 "
          "하나로 합쳤고, 개선 버전은 격자 규칙으로 빈 칸을 채워 12개를 다 찾았다. 아래 줄: 박스가 2개만 남으면 첫 버전은 바닥을 층으로 "
          "착각했고, 개선 버전은 층을 다시 골라 2개를 찾았다. 30장면 전체로는 152개에서 167개(전부)가 됐다.")
    image(fig, A / "twin_detect_accuracy.png", (X[1], 0.905, W, H),
          "2. 시뮬레이터에서 잰 위치 오차",
          "시뮬레이터 안에서는 박스의 진짜 위치를 아니까 검출 결과와 직접 비교할 수 있다. 왼쪽: 박스 중심 오차 분포(평균 8.5 mm, "
          "노이즈를 넣으면 9.8 mm). 가운데: 박스 크기는 실제보다 8~15 mm 작게 잰다. 오른쪽: 찾은 박스 수는 정답과 같다. "
          "이 비교에서 검출기 결함 2건을 찾아 고쳤다.")
    image(fig, A / "twin_closed_loop.png", (X[2], 0.905, W, H),
          "3. 인식 결과만으로 집어 옮기기 (실행기 6종)",
          "시뮬레이터 셀에서 박스를 집어 옆 팔레트로 옮긴다. 왼쪽: 정답 위치를 줄 때(파랑)와 인식 결과로 할 때(주황)의 성공률. "
          "오른쪽 두 막대는 직전 프레임의 층 깊이를 사전으로 쓴 결과다: 인식 구동 성공률이 43 → 78%(mocap), 65 → 75%(트랙 팔)로 오르고, "
          "가운데 그림의 '소스 미검출'과 '흡착 실패'가 사라진다. 팔은 정답 위치를 주면 트랙 100%, 고정 받침대 88%. 오른쪽: 사이클 시간.")
    H = 0.27
    image(fig, A / "ros2_twin_cycle_rviz.png", (X[0], 0.45, W, H),
          "4. ROS2 노드가 시뮬레이터의 팔을 구동하는 화면",
          "세 번째 픽이 실행되는 중. 초록·주황 판은 인식 노드가 본 남은 박스, 왼쪽 위 박스 위의 노란 세 점이 이번 명령(pre-pick, pick, lift), "
          "하늘색 선이 시뮬레이터 팔 끝이 지나온 경로, 주황 구가 지금 팔 끝 위치다. 팔이 명령을 끝내면 완료를 보고하고, 인식 노드가 다음 프레임을 "
          "찍는다. 12박스를 전부 놓고 팔레트가 비어 멈췄다.")
    image(fig, A / "ros2_rviz_cart_synthetic.png", (X[1], 0.45, W, H),
          "5. ROS2 대차 노드를 rviz로 본 화면",
          "대차 견인 고리용 노드. 카메라가 대차를 비스듬히 보므로 데크 평면, 앞 테두리(림), 레일에서 대차 기준 좌표계를 매 프레임 새로 잡는다. "
          "파란 판이 데크, 자홍 선이 림, 노란 선이 레일, 초록 기둥이 고리, 위쪽의 축이 카메라 위치다. 카메라 높이를 바꿔도 고리는 "
          "대차 좌표에서 같은 자리(u 36 mm)에 잡힌다.")
    image(fig, A / "hook_v2_overlay.png", (X[2], 0.45, W, H),
          "6. 실제 대차에서 고리를 찾은 화면",
          "실제 대차의 ToF 반사 강도 이미지. 데크 위 높이 12~150 mm 범위에서 밝게 반사되는 부분이 고리 벽(빨강)이고, 레일(주황)과 고리 윗면은 "
          "어두워서 자연히 분리된다. 노란 부분이 고리 꼭대기, 십자가 추정한 고리 위치다. 4번 촬영에서 이 위치가 3.85 mm 안에서 반복됐다.")
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
