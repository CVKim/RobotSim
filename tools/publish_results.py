# -*- coding: utf-8 -*-
"""집계 결과 JSON 을 공개 레포(results/)로 내보낸다 — 수치 검증 가능하게.

문제: README 의 모든 수치(167/167, mAP50 0.988, 64.9박스, 폐루프 56.7% 등)의 원본이
explore/·runs/ 에만 있고 둘 다 .gitignore 라, 공개 레포에는 PNG 와 마크다운 숫자만 있었다.
집계 JSON 은 회사 원본 데이터가 아니므로(docs/03 공개 정책) 커밋할 수 있다.

익명화: 세션 폴더명은 현장 타임스탬프이므로 frame_01..frame_30 으로 치환한다.
공개 대상은 집계·지표만이며 깊이맵·좌표 원본은 포함하지 않는다.

실행: python tools/publish_results.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"

# (원본 경로, 공개 파일명, 설명)
SOURCES = [
    ("explore/robustness/robustness_v2.json", "detector_robustness.json",
     "기하 검출 v1 vs v2 강건성 (실측 30프레임, 교란 격자)"),
    ("explore/twin/detect_eval.json", "twin_detect_accuracy.json",
     "셀 트윈 절대 정답 기반 검출 정확도 (48장면)"),
    ("explore/twin/closed_loop.json", "twin_closed_loop.json",
     "셀 트윈 인식->제어 폐루프 (oracle vs 인식 구동)"),
    ("explore/sim2real/reeval_v2gt.json", "sim2real_reeval.json",
     "학습 모델 v2 정답 재평가 (홀드아웃 24장)"),
    ("explore/sim2real/gap.json", "sim2real_gap.json", "sim2real 사다리 (합성 전용 -> 실측)"),
    ("explore/sim2real/cotrain.json", "sim2real_cotrain.json", "합성 + 실측 6장 co-training"),
    ("explore/sim2real/tool_compare.json", "sdg_tool_compare.json",
     "SDG 툴 비교 (Isaac Replicator vs BlenderProc)"),
    ("explore/seg/seg_results.json", "seg_vs_geometric.json",
     "학습 세그멘테이션 vs 기하 검출 (열화 그리드 포함)"),
    ("explore/hook/hook_results_v2.json", "hook_repeatability.json",
     "대차 후크 위치 반복성 (정렬 방식 4종)"),
    ("explore/noise/noise_stats.json", "tof_noise_model.json", "ToF 시간 노이즈 실측 모델"),
    ("runs/palletize_ppo/eval.json", "palletize_rl.json", "팔레타이징 MaskablePPO vs DBL 휴리스틱"),
    ("runs/hil_bc/results.json", "imitation_bc.json", "모방학습 BC/DART (가상 Franka)"),
    ("results/palletize_multiseed.json", "palletize_multiseed.json",
     "팔레타이징 RL 다중 시드 + action mask ablation (시드 3개씩)"),
]

SESSION_RE = re.compile(r"\b1260\d{11}\b")


def anonymize(obj, mapping):
    """세션 타임스탬프 -> frame_NN 치환 (키·값·문자열 내부 모두)."""
    if isinstance(obj, dict):
        return {anonymize(k, mapping): anonymize(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [anonymize(v, mapping) for v in obj]
    if isinstance(obj, str):
        def sub(m):
            s = m.group(0)
            if s not in mapping:
                mapping[s] = f"frame_{len(mapping) + 1:02d}"
            return mapping[s]
        return SESSION_RE.sub(sub, obj)
    return obj


def main():
    OUT.mkdir(exist_ok=True)
    mapping, index = {}, []
    for src, name, desc in SOURCES:
        p = ROOT / src
        if not p.exists():
            print(f"  skip (없음): {src}")
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  skip (파싱 실패): {src} — {e}")
            continue
        clean = anonymize(data, mapping)
        (OUT / name).write_text(json.dumps(clean, indent=1, ensure_ascii=False), encoding="utf-8")
        index.append({"file": name, "source": src, "description": desc})
        print(f"  {src} -> results/{name}")

    readme = ["# results — 공개 집계 결과\n",
              "README 의 수치를 검증할 수 있도록 실험 산출 JSON 을 모아 둔 디렉터리.",
              "생성: `python tools/publish_results.py` (원본은 `explore/`·`runs/`, 둘 다 비공개).\n",
              "회사 원본 데이터(깊이맵·좌표·RGB)는 포함하지 않으며, 세션 폴더명(현장 타임스탬프)은",
              f"`frame_01`..`frame_{len(mapping):02d}` 으로 익명화했다 (docs/03 공개 정책).\n",
              "| 파일 | 내용 |", "|---|---|"]
    for it in index:
        readme.append(f"| [{it['file']}]({it['file']}) | {it['description']} |")
    (OUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"\n{len(index)}개 파일, 세션 {len(mapping)}개 익명화 -> {OUT}")


if __name__ == "__main__":
    main()
