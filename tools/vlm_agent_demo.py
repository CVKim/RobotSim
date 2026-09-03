# -*- coding: utf-8 -*-
"""VLM 에이전트 미니 데모 (로컬 Qwen2.5-VL-3B, 데이터 외부 유출 없음).

시나리오: 이적재 셀의 depth 장면 이미지를 VLM에 보여주고
  1) 장면 이해: 박스 수·적재 상태·이상 여부 (그라운딩 검증: 검출기 결과와 대조)
  2) 도구 호출: 인식 파이프라인 결과(검출 박스, 최소동선 PICK 제안, 목적지 스택 높이)를
     JSON으로 제공 → VLM이 다음 액션 계획을 JSON으로 산출
  3) 평가: 박스 수 정답률, 제안 채택 여부, JSON 유효성

실행: python tools/vlm_agent_demo.py  (GPU 1장, bf16 약 7GB)
"""
import json
import re
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_DIR = r"E:\Robot_Sim\models\qwen25vl3b"
ROOT = Path(r"E:\Robot_Sim\explore")
OUT = ROOT / "vlm_agent"
OUT.mkdir(parents=True, exist_ok=True)

SESSIONS = ["126013011365017", "126013011372695", "126013011380406"]


def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda")
    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    return model, proc


def ask(model, proc, image_path, prompt, max_new=400):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(msgs)
    inputs = proc(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    out = out[:, inputs.input_ids.shape[1]:]
    return proc.batch_decode(out, skip_special_tokens=True)[0].strip()


def tool_outputs(session):
    """인식 파이프라인 결과를 '도구 호출 결과' JSON으로 패키징."""
    pp = json.loads((ROOT / "pickpoints" / "pickpoints.json").read_text(encoding="utf-8"))
    ev = json.loads((ROOT / "pickpoints" / "pick_order_eval.json").read_text(encoding="utf-8"))
    picks = pp["sessions"][session]["picks"]
    slot = pp["slots"][session]
    ref = pp["sessions"][slot["ref"]]["picks"]
    present = [i for i, o in enumerate(slot["occupied"]) if o]
    dest = ev["dest_xy"]
    dist = {i: float(((ref[i]["center_mm"][0] - dest[0]) ** 2 + (ref[i]["center_mm"][1] - dest[1]) ** 2) ** 0.5)
            for i in present}
    P = min(present, key=lambda i: dist[i])
    col = [i for i in present if abs(ref[i]["center_mm"][0] - ref[P]["center_mm"][0]) < 80]
    pick = max(col, key=lambda i: dist[i])
    return {
        "detector": {"n_boxes": len(picks),
                     "top_face_mm": [p["dims_mm"] for p in picks[:3]],
                     "max_tilt_deg": max(p["tilt_deg"] for p in picks)},
        "pick_planner": {"suggested_slot": pick, "distance_to_dest_mm": round(dist[pick]),
                         "rule": "column-scan (81% match with real order)"},
        "destination_stack": {"height_mm": 1094, "capacity_ok": True},
    }


def main():
    model, proc = load_model()
    log = []
    for s in SESSIONS:
        img = ROOT / "pickpoints" / f"{s}_picks.png"
        tools = tool_outputs(s)

        # 1) 장면 이해 (도구 없이) — 그라운딩 검증용
        q1 = ("This is a depth image of a logistics cell viewed from above. Boxes on the top layer "
              "are highlighted. Answer in JSON only: {\"n_boxes\": int, \"layout\": str, "
              "\"anomaly\": str or null}.")
        a1 = ask(model, proc, img, q1)

        # 2) 도구 결과를 주고 액션 계획 — 에이전트 스텝
        q2 = ("You are a robot cell supervisor. The perception scan is COMPLETE and CONFIRMED; "
              "tool outputs are authoritative:\n" + json.dumps(tools) +
              "\nRules: if detector.n_boxes > 0 and destination_stack.capacity_ok and max_tilt_deg < 5, "
              "the correct action is \"pick\" using pick_planner.suggested_slot. Use \"alert\" only for "
              "a real anomaly, \"wait\" only if no boxes remain. Answer in JSON only: "
              "{\"action\": \"pick\"|\"wait\"|\"alert\", \"target_slot\": int or null, "
              "\"place\": \"destination_stack\", \"reason\": str, \"safety_ok\": bool}.")
        a2 = ask(model, proc, img, q2)

        def parse(t):
            m = re.search(r"\{.*\}", t, re.S)
            try:
                return json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                return None

        j1, j2 = parse(a1), parse(a2)
        rec = {
            "session": s, "tools": tools,
            "scene_understanding": {"raw": a1, "json": j1,
                                    "count_correct": bool(j1 and j1.get("n_boxes") == tools["detector"]["n_boxes"])},
            "action_plan": {"raw": a2, "json": j2,
                            "adopted_tool_suggestion": bool(j2 and j2.get("target_slot") == tools["pick_planner"]["suggested_slot"]),
                            "action_is_pick": bool(j2 and j2.get("action") == "pick"),
                            "valid_json": j2 is not None},
        }
        log.append(rec)
        print(s, "| count_ok:", rec["scene_understanding"]["count_correct"],
              "| adopted:", rec["action_plan"]["adopted_tool_suggestion"],
              "| pick:", rec["action_plan"]["action_is_pick"],
              "| valid:", rec["action_plan"]["valid_json"])

    (OUT / "transcript.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {
        "n": len(log),
        "count_accuracy": sum(r["scene_understanding"]["count_correct"] for r in log) / len(log),
        "tool_adoption": sum(r["action_plan"]["adopted_tool_suggestion"] for r in log) / len(log),
        "action_pick_rate": sum(r["action_plan"]["action_is_pick"] for r in log) / len(log),
        "valid_json_rate": sum(r["action_plan"]["valid_json"] for r in log) / len(log),
        "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
