# -*- coding: utf-8 -*-
"""로봇 관련 핵심만 추린 2장 요약 PDF (A4 가로). 로컬 산출물 — docs/research/ (gitignore).

    python tools/make_summary_pdf.py  ->  docs/research/Robot_Sim_요약_2p.pdf

1장: 한 장 다이어그램 + 핵심 수치 + 검증 방식 + 한계
2장: 결과 그림 4장 (v1→v2 회수, 트윈 절대 정확도, 폐루프, ROS2 rviz) + 캡션
"""
from __future__ import annotations

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

for cand in (r"C:\Windows\Fonts\malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
    if Path(cand).exists():
        fm.fontManager.addfont(cand)
        plt.rcParams["font.family"] = fm.FontProperties(fname=cand).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, ACC, RED = "#111827", "#4b5563", "#1d4ed8", "#b91c1c"


def text(fig, x, y, s, size=9.5, color=INK, weight="normal", ha="left", va="top", wrap_w=None):
    fig.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, linespacing=1.45)


def image(fig, path, box, title=None, caption=None):
    """box = (x0, y0, w, h) in figure fraction. 이미지 비율 유지, 박스 안에 맞춤."""
    ax = fig.add_axes(box)
    ax.axis("off")
    if Path(path).exists():
        im = imread(path)
        ax.imshow(im)
    else:
        ax.text(0.5, 0.5, f"(missing) {Path(path).name}", ha="center", va="center", color=RED)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK, loc="left", pad=4)
    if caption:
        x0, y0, w, h = box
        fig.text(x0, y0 - 0.006, caption, fontsize=8.3, color=MUTED, va="top", linespacing=1.4)


def page1(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))
    text(fig, 0.04, 0.965, "Robot_Sim — 실측 ToF 인식 → 로봇 좌표 → 셀 트윈 폐루프 → ROS2", 15, weight="bold")
    text(fig, 0.04, 0.925,
         "실공장 ToF 30프레임(단일 SKU)으로 박스 검출·mm 치수·6-DoF 픽 포즈(로봇 베이스 좌표)를 만들고, MuJoCo 셀 트윈에서 "
         "인식 출력만으로 집어 옮기는 폐루프를 닫은 뒤 ROS2 노드/rviz2 로 연결. 절대 정답·교란 격자·전 세션 파리티로 검증.",
         9.2, MUTED)
    image(fig, A / "architecture.png", (0.04, 0.44, 0.92, 0.45))

    # ---- 핵심 수치 (좌)
    text(fig, 0.04, 0.415, "핵심 수치", 11, weight="bold")
    rows = [
        ("박스 검출 (실측 30프레임, RGB 대조 167개)", "v1 152 → v2 167/167 · 오검출 0 · 340 ms"),
        ("절대 정확도 (트윈 48장면)", "중심 8.5 mm(클린) / 9.8 mm(노이즈) · 깊이 0.0 · 잔여≥4 정밀도 0.95"),
        ("인식→제어 폐루프 (트윈 60회)", "oracle 100% vs 인식 46.7% → 인식 비용 53 %p · 11.7 s/픽"),
        ("강건성 (교란 격자, 3시드)", "결손 30개 위치일치 리콜 v1 57% → v2 74%"),
        ("대차 후크 반복성", "데크 ICP 22.7 → 3.2 mm (전체장면 ICP 17.6 mm 오류 원인 규명)"),
        ("팔레타이징 RL (3시드)", "MaskablePPO 64.7±0.3 vs 휴리스틱 56.6 (+14.3%) · mask 제거 43.2"),
        ("ROS2 (Humble, WSL2)", "PointCloud2·MarkerArray·PoseArray(base_link)·status·diagnostics·TF·Trigger, rviz2"),
    ]
    y = 0.385
    for k, v in rows:
        text(fig, 0.04, y, k, 8.6, MUTED)
        text(fig, 0.29, y, v, 8.6, INK)
        y -= 0.036

    # ---- 검증 / 한계 (우)
    text(fig, 0.63, 0.415, "검증 방식", 11, weight="bold")
    text(fig, 0.63, 0.385,
         "• pytest 37 (34개는 합성 프레임만으로) · 30세션 전체 파리티 v1/v2\n"
         "• 셀 트윈 절대 정답: mjData 의 실제 박스 포즈와 비교 (pseudo-GT 순환 참조 제거)\n"
         "• 교란 격자(무효픽셀·깊이노이즈·대비·대면적 결손) × 3시드, 부트스트랩 95% CI\n"
         "• 폐루프는 oracle(정답 포즈)을 나란히 돌려 인식 비용을 분리",
         8.4)
    text(fig, 0.63, 0.235, "한계", 11, weight="bold", color=RED)
    text(fig, 0.63, 0.205,
         "• 단일 SKU·단일 카메라·연속 30프레임. 정답은 검출기 출력 + RGB 육안 대조\n"
         "• 폐루프는 석션 EE 를 mocap 구동 — 팔 기구학·IK·충돌·석션 물리 없음\n"
         "• 실로봇·센서 드라이버·핸드아이 외참 실측값·줄자 GT 없음 (하드웨어 필요)\n"
         "• 미해결: 잔여 ≤3개일 때 층 선택이 아래층으로 점프 (시도 2건 기각·기록)",
         8.4)
    text(fig, 0.04, 0.03, "github.com/CVKim/RobotSim · 원본 데이터 비공개(집계·depth 시각화만) · 수치 원본 results/*.json", 7.8, MUTED)
    pdf.savefig(fig)
    plt.close(fig)


def page2(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))
    text(fig, 0.04, 0.965, "결과 그림", 15, weight="bold")
    W, H = 0.44, 0.36
    image(fig, A / "detector_v2_recovered.png", (0.04, 0.55, W, H),
          "1. 검출 v1 → v2 (실측 depth, 공개 가능 시각화)",
          "위: 2층 중앙 2박스가 v1 에서 병합 → 격자 보완으로 회수. 아래: 잔여 2박스가 바닥 피크에 밀림 → 층 재시도. 30프레임 152 → 167/167.")
    image(fig, A / "twin_detect_accuracy.png", (0.53, 0.55, W, H),
          "2. 트윈 절대 정답 기준 정확도 (48장면)",
          "실제 박스 포즈 대비 중심오차 8.5/9.8 mm, 치수 바이어스 L −8.0/−15.4 mm. 검출기 결함 2건(센티넬 오염·층 선택)을 여기서 발견해 수정.")
    image(fig, A / "twin_closed_loop.png", (0.04, 0.09, W, H),
          "3. 인식→제어 폐루프 (MuJoCo 셀 트윈)",
          "정책은 인식 출력만 사용. oracle 100% vs 인식 구동 46.7% — 지배 실패는 소스 미검출(잔여 소수 층 선택). 사이클 11.7 s/픽.")
    image(fig, A / "ros2_rviz_synthetic.png", (0.53, 0.09, W, H),
          "4. ROS2 perception_node + rviz2 (합성 소스)",
          "포인트클라우드(tof_optical) · 박스 마커·픽 포즈(base_link) · TF · status/diagnostics · Trigger 서비스. WSL2 Humble, 회사 데이터 없이 실행.")
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
