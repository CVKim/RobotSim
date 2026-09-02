# -*- coding: utf-8 -*-
"""sim2real v5: 합성(noisy) + 실측 소량(6장) co-training vs 합성 전용.

실측 30장 → train 6 / holdout 24 분리. 두 모델 모두 holdout 24로만 평가 (공정 비교).
산출: explore/sim2real/cotrain.json
"""
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(r"E:\Robot_Sim\explore")
OUT = ROOT / "sim2real"

REAL = ROOT / "real_eval"
HOLD = ROOT / "real_holdout"
CO = ROOT / "sdg_yolo_cotrain"


def build():
    reals = sorted((REAL / "images/val").glob("*.png"))
    train_real, hold_real = reals[::5][:6], [r for r in reals if r not in reals[::5][:6]]

    # holdout 데이터셋 (24장, val 전용)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (HOLD / sub).mkdir(parents=True, exist_ok=True)
    for r in hold_real:
        shutil.copy(r, HOLD / "images/val" / r.name)
        shutil.copy(REAL / "labels/val" / (r.stem + ".txt"), HOLD / "labels/val" / (r.stem + ".txt"))
    (HOLD / "data.yaml").write_text(
        f"path: {HOLD}\ntrain: images/train\nval: images/val\nnames:\n  0: box\n", encoding="utf-8")

    # co-train 데이터셋 = noisy 합성 전체 + 실측 6장(train에 추가)
    if CO.exists():
        shutil.rmtree(CO)
    shutil.copytree(ROOT / "sdg_yolo_noisy", CO)
    for r in train_real:
        shutil.copy(r, CO / "images/train" / ("real_" + r.name))
        shutil.copy(REAL / "labels/val" / (r.stem + ".txt"),
                    CO / "labels/train" / ("real_" + r.stem + ".txt"))
    (CO / "data.yaml").write_text(
        f"path: {CO}\ntrain: images/train\nval: images/val\nnames:\n  0: box\n", encoding="utf-8")
    print(f"built: cotrain(+{len(train_real)} real), holdout({len(hold_real)})")


def main():
    build()
    results = {}

    # 기존 noisy-only 모델을 holdout으로 재평가 (공정 기준선)
    noisy_w = OUT / "train_noisy" / "weights" / "best.pt"
    m0 = YOLO(str(noisy_w))
    r0 = m0.val(data=str(HOLD / "data.yaml"), split="val", device=0, verbose=False)
    results["noisy_only"] = {"holdout_map50": round(float(r0.box.map50), 4)}

    # co-train 학습 + holdout 평가
    m1 = YOLO("yolo11n.pt")
    m1.train(data=str(CO / "data.yaml"), epochs=60, imgsz=640, device=0, batch=16,
             project=str(OUT), name="train_cotrain", verbose=False, plots=False, workers=0)
    r1 = m1.val(data=str(HOLD / "data.yaml"), split="val", device=0, verbose=False)
    results["noisy_plus_6real"] = {"holdout_map50": round(float(r1.box.map50), 4)}

    (OUT / "cotrain.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(results)


if __name__ == "__main__":
    main()
