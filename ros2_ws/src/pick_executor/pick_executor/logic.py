# -*- coding: utf-8 -*-
"""pick_executor 의 판단 로직 — ROS 없이 순수 파이썬/numpy. 그래서 pytest 로 그대로 검사할 수 있다.

노드(pick_executor_node.py)는 메시지를 풀어 이 모듈에 넘기고, 여기서 나온 Action 을 다시 메시지로 만들 뿐이다.
"ROS 노드 = 얇은 껍데기, 판단은 순수 함수" 로 나누는 것이 테스트 가능한 노드를 만드는 기본 습관이다.

동작 (한 사이클)
  1. 인식 노드가 /perception/pick_poses(PoseArray) 와 /perception/status(JSON) 를 한 프레임에 하나씩 낸다.
     두 메시지는 도착 순서가 보장되지 않으므로 둘 다 모이면 판단한다. 같은 프레임인지는 스탬프(key)로 맞춘다 —
     키가 다르면 오래된 쪽을 버린다 (한 번 어긋난 짝이 영원히 밀리는 것을 막는다).
  2. status 가 OK 가 아니면 명령을 내지 않는다. LAYER_EMPTY 가 연속 empty_retries+1 번이면 종료(DONE) —
     한 프레임의 층 선택 실패로 조기 종료하지 않게.
  3. 실행 중(EXECUTING)에 새 프레임이 오면 명령을 내지 않는다(busy). 로봇이 움직이는 동안 목표를 바꾸면 안 된다.
  4. 첫 포즈에 대해 pre-pick -> pick -> lift 3점 궤적을 만든다. 접근 방향은 포즈 쿼터니언의 툴 +Z 축.
  5. 직전에 보낸 pick 위치와 min_dist_m 안이면 같은 장면의 반복으로 보고 보내지 않는다.
  6. 노드가 '실행'(시뮬레이션: execute_time_s 대기) 을 끝내면 on_execute_done() -> 재촬영(Trigger 서비스) 요청.
  8. 명령을 못 낸 프레임(상태 불량·후보 없음·전부 차단)이 max_no_progress(12)회 연속이면 DONE — 트윈 소스는 소스가 비기 전에는
     스스로 소진되지 않으므로 RETAKE 가 계속되면 종착점이 없었다(리뷰 지적).
  7. 로봇(또는 트윈 브리지)이 결과를 보고하면(on_robot_result) 그것으로 실행을 끝낸다. 실패(도달 불가·충돌·흡착 실패)로 보고된
     pick 위치는 막아 두고(blocked), 다음 프레임에서는 그 후보를 건너뛰어 다음 후보를 보낸다 — 안 그러면 인식 노드가 매 프레임
     같은 1번 픽을 내고 실행기는 '같은 장면'으로 걸러 영원히 재촬영만 한다(트윈 연결 첫 실행에서 실제로 그렇게 됐다).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """geometry_msgs Quaternion (x,y,z,w) -> 3x3 회전행렬. 열 = 툴 x,y,z 축을 부모(base) 좌표로 표현.

    노름이 0 이면(기본 생성된 빈 쿼터니언) ValueError — 그대로 단위행렬로 보면 '아래에서 위로 접근'이 되어 위험하다."""
    n = float(np.sqrt(x * x + y * y + z * z + w * w))
    if n < 1e-6:
        raise ValueError("zero quaternion: orientation missing")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class Trajectory:
    pre_pick: tuple            # (x, y, z) m, base 좌표
    pick: tuple
    lift: tuple
    orientation: tuple         # (x, y, z, w) — 세 점 모두 같은 툴 자세
    approach: tuple            # 단위 벡터, 박스로 향하는 방향 (= 툴 +Z)

    def points(self):
        return [self.pre_pick, self.pick, self.lift]


def three_point_trajectory(position: Sequence[float], quat: Sequence[float],
                           clearance_m: float = 0.15, lift_m: float = 0.25) -> Trajectory:
    """pick 포즈 하나 -> pre-pick(접근 시작) / pick / lift(들어올린 뒤) 3점.

    인식 노드의 규약: 툴 +Z = 접근 벡터(박스 쪽). 탑다운이면 (0,0,-1) 이라 pre-pick 은 pick 의 위쪽이다.
    pre  = pick - approach * clearance
    lift = pick - approach * (clearance + lift)
    """
    p = np.asarray(position, float)
    R = quat_to_matrix(*quat)
    approach = R[:, 2]
    approach = approach / (np.linalg.norm(approach) or 1.0)
    pre = p - approach * clearance_m
    lift = p - approach * (clearance_m + lift_m)
    t = lambda v: tuple(float(a) for a in v)  # noqa: E731
    return Trajectory(pre_pick=t(pre), pick=t(p), lift=t(lift), orientation=tuple(float(q) for q in quat),
                      approach=t(approach))


@dataclass
class Action:
    kind: str                  # send | busy | skip_duplicate | skip_status | empty | done | recapture | noop
    reason: str = ""
    trajectory: Optional[Trajectory] = None
    status: Optional[str] = None


@dataclass
class Executor:
    """인식 결과 -> 로봇 명령 판단기 (상태 기계). IDLE -> (send) EXECUTING -> (on_execute_done) IDLE ... -> DONE."""
    min_dist_m: float = 0.05
    clearance_m: float = 0.15
    lift_m: float = 0.25
    stop_on_empty: bool = True
    empty_retries: int = 2            # LAYER_EMPTY 가 연속 이 횟수를 넘어야 DONE
    max_no_progress: int = 12         # 명령을 못 낸 프레임이 연속 이 횟수를 넘으면 DONE (RETAKE/LOW_CONFIDENCE 가 계속될 때의 종착점)
    no_progress: int = 0
    state: str = "IDLE"
    last_sent_pick: Optional[np.ndarray] = None
    consecutive_empty: int = 0
    counters: dict = field(default_factory=lambda: dict(sent=0, busy=0, skipped_duplicate=0, skipped_status=0,
                                                        empty=0, recapture=0, unpaired_dropped=0,
                                                        robot_ok=0, robot_fail=0, robot_error=0))
    outcomes: dict = field(default_factory=dict)  # 로봇이 보고한 결과별 횟수 (placed, grasp_miss, ...)
    blocked: list = field(default_factory=list)   # 로봇이 실패를 보고한 pick 위치 — 다음 프레임에서 그 후보는 건너뛴다
    _pending_poses: Optional[tuple] = None      # (poses, key)
    _pending_status: Optional[tuple] = None     # (status, key)

    # ---- 입력 ------------------------------------------------------------
    def on_poses(self, poses: list, key=None) -> Optional[Action]:
        """poses = [((x,y,z), (qx,qy,qz,qw)), ...] 픽 순서대로. key = 프레임 식별자(스탬프). status 가 모이면 판단."""
        self._pending_poses = (list(poses), key)
        return self._maybe_decide()

    def on_status(self, status: dict, key=None) -> Optional[Action]:
        """status = /perception/status JSON (dict). 'status' 키만 필수."""
        self._pending_status = (dict(status), key)
        return self._maybe_decide()

    def on_execute_done(self) -> Action:
        """노드가 실행(시뮬레이션 대기) 을 끝냈다 -> 다음 프레임을 요청."""
        if self.state == "EXECUTING":
            self.state = "IDLE"
        self.counters["recapture"] += 1
        return Action("recapture", "execution finished, request next frame")

    def on_robot_result(self, result: dict) -> Action:
        """로봇(트윈 브리지)이 실행 결과를 보고했다: {"ok": bool, "result": "placed"|"grasp_miss"|..., ...}.

        실패면 그 pick 위치를 막는다 — 박스가 그 자리에 그대로 남아 인식 노드가 다시 1번으로 낼 것이기 때문이다.
        성공이든 실패든 다음 프레임을 요청한다."""
        ok = bool(result.get("ok", False))
        name = str(result.get("result") or ("ok" if ok else "failed"))
        # 박스 쪽 실패(도달 불가·충돌·흡착 실패·놓기 실패)만 그 자리를 막는다. 전송·브리지 오류(bridge_error, busy, unknown, error:*)는
        # 박스에 대해 아무것도 말해 주지 않으므로 막지 않고 다시 촬영만 한다 (리뷰 지적: 서버 재시작 한 번에 자리가 영구 차단됐다)
        robot_side = name in ("ik_unreachable", "path_collision", "grasp_miss", "misplaced") or name.startswith(("descent_", "place_"))
        if ok:
            self.counters["robot_ok"] += 1
        elif robot_side:
            self.counters["robot_fail"] += 1
        else:
            self.counters["robot_error"] += 1
        self.outcomes[name] = self.outcomes.get(name, 0) + 1
        if not ok and robot_side and self.last_sent_pick is not None:
            self.blocked.append(self.last_sent_pick.copy())
        if not ok and not robot_side:
            self.last_sent_pick = None       # 전송 오류 뒤에는 같은 자리를 다시 보내야 한다 — 중복 규칙(min_dist)이 막지 않게
        if self.state == "EXECUTING":
            self.state = "IDLE"
        self.counters["recapture"] += 1
        return Action("recapture", f"robot reported {name} ({'ok' if ok else 'failed'}), request next frame")

    def _is_blocked(self, p) -> bool:
        return any(float(np.linalg.norm(np.asarray(p, float) - b)) < self.min_dist_m for b in self.blocked)

    def on_capture_failed(self, message: str) -> Action:
        """Trigger 응답 success=False.

        인식 노드는 프레임을 처리했지만 상태가 OK 가 아닐 때도 success=False 에 상태 JSON 을 담아 준다 — 그 경우 프레임은
        이미 토픽으로 나갔으므로 여기서 할 일이 없다(noop; 판단은 on_poses/on_status 가 한다). 진짜 실패는
        '소스 소진'(종료) 과 그 외 전송 오류(재시도) 두 가지다."""
        if "exhausted" in message:
            self.state = "DONE"
            return Action("done", f"source exhausted: {message}")
        try:
            d = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            d = None
        if isinstance(d, dict) and d.get("status") == "SOURCE_ERROR":
            # 인식 노드가 프레임을 못 받았다(트윈 서버 끊김 등). 즉시 다시 부르면 빈 루프가 되므로 잠시 뒤에
            return Action("retry_later", f"source error: {d.get('reason', '')}")
        if isinstance(d, dict) and "status" in d:
            return Action("noop", f"frame processed with status {d['status']}; decision comes from the topics")
        return Action("recapture", f"capture failed ({message}), retry")

    # ---- 판단 ------------------------------------------------------------
    def _maybe_decide(self) -> Optional[Action]:
        if self._pending_poses is None or self._pending_status is None:
            return None
        (poses, kp), (st, ks) = self._pending_poses, self._pending_status
        if kp is not None and ks is not None and kp != ks:
            # 다른 프레임의 짝 -> 오래된 쪽을 버리고 새 쪽을 기다린다
            self.counters["unpaired_dropped"] += 1
            if kp < ks:
                self._pending_poses = None
            else:
                self._pending_status = None
            return None
        self._pending_poses = self._pending_status = None
        return self.decide(poses, st)

    def decide(self, poses: list, status: dict) -> Action:
        a = self._decide(poses, status)
        if a.kind == "send":
            self.no_progress = 0
        elif a.kind in ("skip_status", "empty"):
            self.no_progress += 1
            if self.stop_on_empty and self.no_progress > self.max_no_progress and self.state != "DONE":
                self.state = "DONE"
                return Action("done", f"no command for {self.no_progress} consecutive frames (last: {a.reason})", status=a.status)
        return a

    def _decide(self, poses: list, status: dict) -> Action:
        if self.state == "DONE":
            return Action("done", "already finished", status=status.get("status"))
        s = str(status.get("status", "UNKNOWN"))
        if s == "LAYER_EMPTY":
            self.counters["empty"] += 1
            self.consecutive_empty += 1
            if self.stop_on_empty and self.consecutive_empty > self.empty_retries:
                self.state = "DONE"
                return Action("done", f"layer empty {self.consecutive_empty} times in a row — nothing left to pick", status=s)
            return Action("empty", f"layer empty ({self.consecutive_empty}/{self.empty_retries + 1}), retake", status=s)
        if s != "OK":
            self.consecutive_empty = 0
            self.counters["skipped_status"] += 1
            return Action("skip_status", f"perception status {s}: {status.get('reason', '')}", status=s)
        if not poses:
            self.consecutive_empty = 0
            self.counters["empty"] += 1
            return Action("empty", "status OK but no pick poses", status=s)
        if self.state == "EXECUTING":
            self.counters["busy"] += 1
            return Action("busy", "robot still executing the previous command — frame ignored", status=s)
        cands = [pq for pq in poses if not self._is_blocked(pq[0])]
        if not cands:
            # 남은 후보가 전부 로봇이 실패를 보고한 자리다. LAYER_EMPTY 와 같은 규칙으로 연속 횟수를 세어 끝낸다.
            self.counters["empty"] += 1
            self.consecutive_empty += 1
            if self.stop_on_empty and self.consecutive_empty > self.empty_retries:
                self.state = "DONE"
                return Action("done", f"all {len(poses)} remaining candidates were reported failed by the robot", status=s)
            return Action("empty", f"all {len(poses)} candidates blocked by robot failures "
                                   f"({self.consecutive_empty}/{self.empty_retries + 1}), retake", status=s)
        self.consecutive_empty = 0
        position, quat = cands[0]
        p = np.asarray(position, float)
        if self.last_sent_pick is not None and float(np.linalg.norm(p - self.last_sent_pick)) < self.min_dist_m:
            self.counters["skipped_duplicate"] += 1
            return Action("skip_duplicate",
                          f"pick within {self.min_dist_m * 1e3:.0f} mm of the last command — same scene", status=s)
        try:
            traj = three_point_trajectory(position, quat, self.clearance_m, self.lift_m)
        except ValueError as e:
            self.counters["skipped_status"] += 1
            return Action("skip_status", f"invalid pick pose: {e}", status=s)
        self.last_sent_pick = p
        self.state = "EXECUTING"
        self.counters["sent"] += 1
        return Action("send", f"pick #{self.counters['sent']}", trajectory=traj, status=s)

    def snapshot(self) -> dict:
        return dict(state=self.state, **self.counters, consecutive_empty=self.consecutive_empty, no_progress=self.no_progress,
                    blocked=len(self.blocked), outcomes=dict(self.outcomes),
                    last_pick_m=None if self.last_sent_pick is None else [round(float(v), 4) for v in self.last_sent_pick])
