# -*- coding: utf-8 -*-
"""twin_bridge 의 순수 파이썬 부분 — 메시지 <-> 트윈 요청/응답 변환. ROS 없이 pytest 로 검사한다."""
from __future__ import annotations

import math
from typing import Optional, Sequence


def execute_request(poses: Sequence, frame_id: str = "base_link") -> dict:
    """/robot/target_poses 의 3점 [(pos, quat), ...] (pre-pick, pick, lift) -> 트윈 `execute` 요청 필드.

    pick_executor 규약: 3점은 같은 툴 자세, 순서는 pre-pick -> pick -> lift. 자세는 pick 것을 쓴다.
    점이 3개 미만이거나 쿼터니언이 0 이면 ValueError (명령을 트윈에 보내지 않는다)."""
    if len(poses) < 3:
        raise ValueError(f"need 3 poses (pre-pick, pick, lift), got {len(poses)}")
    (pre, _), (pick, quat), (lift, _) = poses[0], poses[1], poses[2]
    q = [float(v) for v in quat]
    if math.sqrt(sum(v * v for v in q)) < 1e-6:
        raise ValueError("zero quaternion in pick pose")
    return {"pre_pick": [float(v) for v in pre], "pick": [float(v) for v in pick], "lift": [float(v) for v in lift],
            "quat": q, "frame_id": frame_id}


RESULT_KEYS = ("ok", "result", "cycle_s", "wall_s", "remaining", "placed", "seq", "gap_mm", "pick_err_mm", "place_err_mm",
               "landed_z", "slot", "held", "error")


def result_message(res: dict, cmd_seq: int, stamp: Optional[Sequence[int]] = None) -> dict:
    """트윈 응답 -> /robot/execution_result JSON (tcp_path 같은 큰 필드는 뺀다)."""
    out = {k: res[k] for k in RESULT_KEYS if k in res}
    out.setdefault("ok", False)
    out.setdefault("result", "unknown")
    out["cmd_seq"] = int(cmd_seq)
    if stamp is not None:
        out["cmd_stamp"] = [int(stamp[0]), int(stamp[1])]
    return out


def path_points(res: dict, max_points: int = 400) -> list:
    """tcp_path [[t,x,y,z],...] -> [(x,y,z),...] (rviz LINE_STRIP 용, 너무 길면 등간격으로 줄인다)."""
    path = res.get("tcp_path") or []
    if len(path) > max_points:
        step = len(path) / float(max_points)
        path = [path[int(i * step)] for i in range(max_points)] + [path[-1]]
    return [(float(p[1]), float(p[2]), float(p[3])) for p in path]
