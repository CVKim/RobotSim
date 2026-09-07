# -*- coding: utf-8 -*-
"""트윈 연결 사이클(ROS2 그래프 + 트윈 서버) 한 번의 기록을 요약해 공개용 JSON 으로 만든다.

    python tools/twin_cycle_summary.py            # explore/twin/twin_server.jsonl + explore/ros2/twin_cycle.log -> results/ros2_twin_cycle.json

트윈 서버 JSONL(프레임·실행 기록)이 원본이고, ROS 로그에서는 인식 노드가 프레임마다 찾은 박스 수와 pick_executor 의 DONE 사유만 뽑는다.
경로·세션 ID 같은 로컬 정보는 넣지 않는다.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRV = ROOT / "explore" / "twin" / "twin_server.jsonl"
LOG = ROOT / "explore" / "ros2" / "twin_cycle.log"
OUT = ROOT / "results" / "ros2_twin_cycle.json"


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    recs = [json.loads(l) for l in SRV.read_text(encoding="utf-8").splitlines() if l.strip()]
    starts = [r for r in recs if r["event"] == "start"]
    if not starts:
        raise SystemExit(f"{SRV}: no 'start' event — server log predates the start record; rerun the demo")
    st = starts[-1]
    ARM_KO = {"track": "UR10e + 리니어 트랙", "fixed": "UR10e 고정 받침대", "none": "mocap 석션 (팔 없음)"}
    frames = [r for r in recs if r["event"] == "frame"]
    execs = [r for r in recs if r["event"] == "execute"]
    log = LOG.read_text(encoding="utf-8", errors="replace") if LOG.exists() else ""
    # 인식 노드 로그: "[k] twin#k (remaining R, placed P): STATUS boxes=B pickable=N valid=V% ..."
    per_frame = []
    for m in re.finditer(r"\[(\d+)\] twin#(\d+) \(remaining (\d+), placed (\d+)\): (\w+) boxes=(\d+) pickable=(\d+) valid=(\d+)%", log):
        per_frame.append({"frame": int(m.group(2)), "remaining_truth": int(m.group(3)), "placed": int(m.group(4)),
                          "status": m.group(5), "boxes_detected": int(m.group(6)), "pickable": int(m.group(7)),
                          "valid_pct": int(m.group(8))})
    done = re.search(r"DONE: ([^\n]*)", log)
    exec_rows = [{"cmd": e["seq"], "result": e["result"], "ok": e["ok"], "sim_cycle_s": e["cycle_s"], "wall_s": e.get("wall_s"),
                  "gap_mm": e.get("gap_mm"), "place_err_mm": e.get("place_err_mm"), "remaining_after": e["remaining"],
                  "placed_after": e["placed"]} for e in execs]
    outcomes = {}
    for e in execs:
        outcomes[e["result"]] = outcomes.get(e["result"], 0) + 1
    cyc = [e["cycle_s"] for e in execs]
    n0 = frames[0]["remaining"] if frames else None
    det = [f for f in per_frame if f["status"] == "OK"]
    summary = {
        "what": "ROS2 그래프(perception_node -> pick_executor -> twin_bridge)가 MuJoCo 셀 트윈을 카메라·로봇으로 써서 소스 팔레트를 비운 기록",
        "arm": f"{ARM_KO.get(st['arm'], st['arm'])} (tools/twin_server.py --arm {st['arm']} --seed {st['seed']} --boxes {st['n_boxes']} --noise {st['noise']})",
        "server": {k: st.get(k) for k in ("arm", "seed", "n_boxes", "noise", "arm_cfg")},
        "boxes_start": n0,
        "frames": len(frames), "commands": len(execs),
        "placed": execs[-1]["placed"] if execs else 0,
        "remaining_end": execs[-1]["remaining"] if execs else n0,
        "outcomes": outcomes,
        "sim_cycle_s": {"mean": round(float(np.mean(cyc)), 2), "min": round(float(min(cyc)), 2), "max": round(float(max(cyc)), 2)} if cyc else None,
        "wall_per_execute_s": round(float(np.mean([e.get("wall_s", 0) for e in execs])), 2) if execs else None,
        "detected_vs_truth": [{"frame": f["frame"], "truth": f["remaining_truth"], "detected": f["boxes_detected"]} for f in per_frame],
        # OK 프레임만의 평균과, 검출 0 으로 끝난 LAYER_EMPTY 프레임까지 넣은 평균을 나눠 적는다 (이름 없는 필터는 오해를 낳는다 — 리뷰 지적)
        "detection_recall_mean_ok_frames": (round(float(np.mean([min(f["boxes_detected"], f["remaining_truth"]) / f["remaining_truth"]
                                                                 for f in det if f["remaining_truth"] > 0])), 3) if det else None),
        "detection_recall_mean_all_frames": (round(float(np.mean([min(f["boxes_detected"], f["remaining_truth"]) / f["remaining_truth"]
                                                                  for f in per_frame if f["remaining_truth"] > 0])), 3) if per_frame else None),
        "frames_not_ok": [{"frame": f["frame"], "status": f["status"], "truth": f["remaining_truth"]} for f in per_frame if f["status"] != "OK"],
        "done_reason": done.group(1).strip() if done else None,
        "executions": exec_rows,
    }
    OUT.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("executions", "detected_vs_truth")}, indent=1, ensure_ascii=False))
    print("saved", OUT)


if __name__ == "__main__":
    main()
