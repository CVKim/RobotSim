# -*- coding: utf-8 -*-
"""로봇 관련 핵심만 추린 2장 요약 PDF (A4 가로). 로컬 산출물 — docs/research/ (gitignore).

    python tools/make_summary_pdf.py  ->  docs/research/Robot_Sim_요약_2p.pdf (+ 페이지 PNG 미리보기)

1장: 한 장 다이어그램 + 핵심 수치 + 검증 방식 + 한계
2장: 결과 그림 4장 (v1→v2 회수, 트윈 절대 정확도, 폐루프, ROS2 rviz) + 캡션
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
INK, MUTED, ACC, RED = "#111827", "#4b5563", "#1d4ed8", "#b91c1c"


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


def image(fig, path, box, title=None, caption=None):
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
        text(fig, x0, y_top - h_fit - 0.008, caption, 8.2, MUTED, wrap=w)


def page1(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    text(fig, 0.04, 0.965, "Robot_Sim — 실측 ToF 인식 → 로봇 좌표 → 셀 트윈 폐루프 → ROS2", 15, weight="bold")
    text(fig, 0.04, 0.928,
         "실공장 ToF 데이터 두 종(빈피킹 30프레임 · 대차 견인 고리 4세션)으로 박스 검출·mm 치수·6-DoF 픽 포즈(로봇 베이스 좌표)와 "
         "대차 구조물 기준 좌표계를 만들고, MuJoCo 셀 트윈에서 인식 출력만으로 집어 옮기는 폐루프를 닫은 뒤 ROS2 노드 2종/rviz2 로 연결. "
         "절대 정답·교란 격자·전 세션 파리티로 검증. 실로봇 없음, 원본 데이터 비공개(집계·depth 시각화만).",
         9.0, MUTED, wrap=0.92)
    image(fig, A / "architecture.png", (0.04, 0.855, 0.92, 0.41))

    # ---- 핵심 수치 (좌, 두 열)
    text(fig, 0.04, 0.415, "핵심 수치", 11, weight="bold")
    rows = [
        ("박스 검출 (실측 30프레임, RGB 대조 167개)", "v1 152 → v2 167/167 · 오검출 0 · 340 ms"),
        ("절대 정확도 (트윈 48장면)", "중심 8.5 mm(클린) / 9.8 mm(노이즈) · 깊이 0.0 · 잔여>=4 정밀도 0.95"),
        ("인식→제어 폐루프 (트윈 60회)", "oracle 100% vs 인식 46.7% → 인식 비용 53 %p · 11.7 s/픽"),
        ("강건성 (교란 격자, 3시드)", "결손 30개에서 위치일치 리콜 v1 57% → v2 74%"),
        ("대차 고리 · 대차 좌표계 (실측 4세션, 카메라 높이 400~463 mm)", "데크 평면·림·레일에서 좌표계 추정 → 고리 반복성 22.7 → 3.85 mm (데크 ICP 정련 시 3.2)"),
        ("팔레타이징 RL (3시드)", "MaskablePPO 64.7±0.3 vs 휴리스틱 56.6 (+14.3%) · mask 제거 43.2"),
        ("ROS2 (Humble, WSL2) 노드 2종", "빈피킹: PoseArray(base_link)·정적 TF · 대차: 동적 TF tof_optical→cart·도킹 목표 포즈 · rviz2 · colcon test 12"),
    ]
    y = 0.385
    for k, v in rows:
        text(fig, 0.04, y, k, 8.4, MUTED, wrap=0.235)
        text(fig, 0.285, y, v, 8.4, INK, wrap=0.31)
        y -= 0.040

    # ---- 검증 / 한계 (우)
    text(fig, 0.63, 0.415, "검증 방식", 11, weight="bold")
    text(fig, 0.63, 0.385,
         "• pytest 51 (46개는 합성 프레임만으로) · 30세션 전체 파리티 v1/v2 · 대차 4세션 파리티 · colcon test 12\n"
         "• 셀 트윈 절대 정답: mjData 의 실제 박스 포즈와 비교 (pseudo-GT 순환 참조 제거)\n"
         "• 교란 격자(무효픽셀·깊이노이즈·대비·대면적 결손) × 3시드, 부트스트랩 95% CI\n"
         "• 폐루프는 oracle(정답 포즈)을 나란히 돌려 인식 비용을 분리",
         8.2, wrap=0.33)
    text(fig, 0.63, 0.215, "한계", 11, weight="bold", color=RED)
    text(fig, 0.63, 0.185,
         "• 단일 SKU·단일 카메라·연속 30프레임. 정답은 검출기 출력 + RGB 육안 대조\n"
         "• 폐루프는 석션 EE 를 mocap 구동 — 팔 기구학·IK·충돌·석션 물리 없음\n"
         "• 실로봇·센서 드라이버·핸드아이 외참 실측값·줄자 GT 없음 (하드웨어 필요)\n"
         "• 미해결: 잔여 <=3개일 때 층 선택이 아래층으로 점프 (시도 2건 기각·기록)",
         8.2, wrap=0.33)
    text(fig, 0.04, 0.03, "github.com/CVKim/RobotSim · 원본 데이터 비공개(집계·depth 시각화만) · 수치 원본 results/*.json", 7.8, MUTED)
    pdf.savefig(fig)
    plt.close(fig)


def page2(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    text(fig, 0.04, 0.965, "결과 그림", 15, weight="bold")
    W, H = 0.29, 0.34
    X = (0.04, 0.355, 0.67)
    image(fig, A / "detector_v2_recovered.png", (X[0], 0.895, W, H),
          "1. 검출 v1 → v2 (실측 depth)",
          "2층 중앙 2박스 병합 → 격자 보완으로 회수, 잔여 2박스 층 재시도. 30프레임 152 → 167/167.")
    image(fig, A / "twin_detect_accuracy.png", (X[1], 0.895, W, H),
          "2. 트윈 절대 정답 정확도 (48장면)",
          "실제 박스 포즈 대비 중심 8.5/9.8 mm, 치수 L -8.0/-15.4 mm. 검출기 결함 2건을 여기서 발견해 수정.")
    image(fig, A / "twin_closed_loop.png", (X[2], 0.895, W, H),
          "3. 인식→제어 폐루프 (셀 트윈)",
          "정책은 인식 출력만 사용. oracle 100% vs 인식 46.7% — 지배 실패는 잔여 소수 층 선택. 11.7 s/픽.")
    image(fig, A / "ros2_rviz_synthetic.png", (X[0], 0.46, W, H),
          "4. ROS2 perception_node (빈피킹)",
          "초록 직접 검출·주황 격자 보완, 픽 포즈 축(base_link), 파란 화살표 = 다음 픽. 정적 TF, Trigger 서비스.")
    image(fig, A / "ros2_rviz_cart_synthetic.png", (X[1], 0.46, W, H),
          "5. ROS2 cart_node (대차 고리)",
          "고정 프레임 = 대차. 파란 판 데크, 자홍 림, 노란 레일, 초록 고리, 위의 축 = 카메라(tof_optical). 동적 TF.")
    image(fig, A / "hook_v2_overlay.png", (X[2], 0.46, W, H),
          "6. 실측 대차 고리 검출 (ToF 강도 이미지)",
          "높이 밴드 안 '밝은' 성분 = 고리 벽(빨강), 크라운(노랑), 레일(주황) 분리. 4세션 반복성: 구조 프레임 3.85 · 데크 ICP 3.2 mm RMS.")
    text(fig, 0.04, 0.03, "상세: docs/30_결과_상세.md · docs/41_셀_트윈.md · docs/42_ROS2_핸즈온.md", 7.8, MUTED)
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
