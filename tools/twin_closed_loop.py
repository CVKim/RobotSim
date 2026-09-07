# -*- coding: utf-8 -*-
"""인식 -> 계획 -> 실행 폐루프 (MuJoCo 셀 트윈).

지금까지 인식-정책 연동은 depth 오버레이에 PICK/PLACE 를 그리는 데서 끝났고,
실제로 무언가를 집어 옮긴 적이 없었다. 이 스크립트가 루프를 닫는다:

    렌더(ToF) -> detect_boxes_v2 -> 픽 순서 계획 -> 6-DoF 포즈(로봇 베이스 좌표)
             -> 석션 EE 이동/흡착/이송/해제(물리) -> 결과 판정 -> 다음 프레임 재촬영

핵심: 정책이 **인식 출력만** 보고 움직인다. 시뮬레이터의 정답 위치를 쓰지 않는다.
따라서 인식 오차(병합·누락·중심 오차)가 실제 픽 실패로 어떻게 이어지는지 측정된다.
정답 위치로 집는 'oracle' 모드를 함께 돌려 인식 때문에 잃는 성능을 분리한다.

두 가지 실행기:
  --arm none  (기본)  석션 EE 를 mocap 으로 직접 구동. 도달범위·특이점·주변 충돌은 평가되지 않는다.
  --arm fixed | track  UR10e(sim/arm.py) 가 관절 속도 한계로 움직인다. IK 가 안 풀리면 '도달 불가',
                       경로에서 팔이 박스·설비에 닿으면 '경로 충돌' 로 그 픽을 건너뛴다. 사이클 시간은 관절 속도로 정해진다.
                       fixed 는 받침대 고정(results/twin_arm_reach.json 의 최적 위치), track 은 x 방향 리니어 트랙 ±0.6 m.

실행:  .venv\\Scripts\\python.exe tools/twin_closed_loop.py --episodes 8 [--arm track]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from binpick_topface import detect_boxes_v2  # noqa: E402
from cell_scene import (BOX, CAM_H, DECK_H, N_COL, N_ROW, build_xml,  # noqa: E402
                        dest_slots, dest_world_xy, grid_xy)
from cell_twin import TwinRenderer, ground_truth, settle  # noqa: E402
from robotsim_perception.pose import (box_to_pick_pose, suction_footprint_ok,  # noqa: E402
                                      topdown_camera_transform)

OUT = ROOT / "explore" / "twin"
CTRL_HZ = 500.0          # timestep 0.002
SUCTION_REACH_MM = 45.0  # 이 거리 안이면 흡착 성립
SOURCE_ROI_MM = 620.0    # 소스 팔레트 반경 (카메라 좌표 mm) — 목적지 스택 제외용
LATERAL_TOL_MM = 90.0    # 흡착 성립 측면 허용 오차 (컵 아래 면에만 붙음)
ARM_DEFAULT_CFG = dict(base_xy=(-0.4, 0.85), pedestal_h=1.0)


def cam_to_world(p_mm):
    """카메라 좌표 mm -> 월드 m (트윈 관례: ToF_X=world X, ToF_Y=-world Y, D=CAM_H-z)."""
    return np.array([p_mm[0] / 1000.0, -p_mm[1] / 1000.0, CAM_H - p_mm[2] / 1000.0])


def arm_cfg_from_results(track: bool) -> dict:
    """tools/twin_arm_reach.py 가 고른 최적 받침대 위치를 쓴다. 없으면 기본값.
    트랙 구성은 레일+캐리지(0.12 m)만큼 받침대를 낮춰 팔 베이스 높이를 고정 구성과 맞춘다."""
    from arm import TRACK_STACK_H
    p = ROOT / "results" / "twin_arm_reach.json"
    cfg = dict(ARM_DEFAULT_CFG)
    if p.exists():
        try:
            best = json.loads(p.read_text(encoding="utf-8"))["best_fixed"]["cfg"]
            cfg = dict(base_xy=tuple(best["base_xy"]), pedestal_h=float(best["pedestal_h"]))
        except Exception:
            pass
    if track:
        cfg["track_range"] = 0.6
        cfg["pedestal_h"] = round(cfg["pedestal_h"] - TRACK_STACK_H, 3)
    return cfg


def shared_layout_cfg() -> dict:
    """세 실행기(mocap · 고정 · 트랙)가 같은 설비 집합을 갖도록 공통 제외 구역 = 트랙 구성의 것."""
    c = arm_cfg_from_results(track=True)
    return dict(base_xy=c["base_xy"], track_range=c["track_range"])


class Cell:
    """mocap 석션 EE 로 구동하는 셀 (팔 기구학 없음)."""

    def __init__(self, layout, seed, arm_cfg=None, arm_layout=False, layout_cfg=None):
        import mujoco
        self.mj = mujoco
        xml, _ = build_xml(layout, seed=seed, arm=arm_cfg, arm_layout=arm_layout, layout_cfg=layout_cfg)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        if arm_cfg:
            # 정착(settle) 전에 팔을 홈 자세로. 기본 qpos=0 은 팔이 수평으로 뻗은 자세라 낮은 받침대에서는
            # 정착 중 목적지 스택을 휩쓴다 (검토에서 발견)
            from arm import Arm
            a = Arm(self.m, self.d)
            a.set_q(a.home)
            a.set_ctrl(a.home)
        settle(self.m, self.d, 1200)
        self.rend = TwinRenderer(self.m)
        self.grip_eq = {}
        self.box_ids = []
        for k in range(self.m.nbody):
            nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, k)
            if nm and nm.startswith("box"):
                self.box_ids.append(int(nm[3:]))
        if arm_cfg is None:
            self.mocap_id = self.m.body_mocapid[
                mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "suction_target")]
            self.suction_bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "suction")
            for k in range(self.m.neq):
                nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, k)
                if nm and nm.startswith("grip"):
                    self.d.eq_active[k] = 0          # weld 는 쓰지 않는다 (아래 주석 참조)
        self.held = None
        self.hold_offset = None
        self.steps = 0          # 물리 스텝 누적 -> 사이클 타임(초) = steps / CTRL_HZ

    def close(self):
        self.rend.close()

    def box_bid(self, i):
        return self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_BODY, f"box{i}")

    # ---- EE 위치 (서브클래스가 바꾼다) -----------------------------------------
    def ee_pos(self):
        return self.d.xpos[self.suction_bid]

    def move_to(self, xyz_w, speed_mps=0.6, settle_steps=150, **_):
        """mocap 을 목표까지 S-커브로 이동(거리에 비례한 시간) 후 안정화.

        박스를 문 채로 급가속하면 흡착 weld 가 버티지 못해 박스가 뒤처진다.
        이동 시간을 거리/속도로 잡고 smoothstep 프로파일을 써서 가속도를 제한한다.
        반환 (ok, reason, err_mm)
        """
        start = self.d.mocap_pos[self.mocap_id].copy()
        goal = np.asarray(xyz_w, float)
        dist = float(np.linalg.norm(goal - start))
        steps = max(int(CTRL_HZ * dist / max(speed_mps, 1e-3)), 60)
        for i in range(steps):
            t = (i + 1) / steps
            a = t * t * (3 - 2 * t)           # smoothstep: 시작/끝 가속도 0
            self.d.mocap_pos[self.mocap_id] = start + (goal - start) * a
            self.mj.mj_step(self.m, self.d)
            self.steps += 1
            self._carry()
        for _ in range(settle_steps):
            self.mj.mj_step(self.m, self.d)
            self.steps += 1
            self._carry()
        return True, "ok", float(np.linalg.norm(self.ee_pos() - goal)) * 1000.0

    def move_linear(self, xyz_w, **kw):
        """mocap 은 원래 직선으로 움직인다 — 팔 실행기와 인터페이스를 맞추기 위한 별칭."""
        return self.move_to(xyz_w, **kw)

    def _carry(self):
        """흡착 중인 박스를 EE 에 강체 부착 (운동학적 이송 모델).

        MuJoCo weld equality 를 런타임에 켜면 기본 relpose 가 항등이라 박스를 EE 프레임으로
        끌어당겨 0.8 m 씩 튕겨 나간다(1차 시도에서 발생). relpose 를 매번 맞춰 쓰는 대신
        이송 구간만 운동학적으로 부착한다. 파지 전 접촉과 해제 후 안착은 물리 그대로다.
        한계: 석션 컵 컴플라이언스·이송 중 박스 흔들림·관성에 의한 파지 실패는 모델링되지 않는다.
        """
        if self.held is None:
            return
        jid = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_JOINT, f"box{self.held}_j")
        if jid < 0:
            return
        adr = self.m.jnt_qposadr[jid]
        self.d.qpos[adr:adr + 3] = self.ee_pos() + self.hold_offset
        self.d.qpos[adr + 3:adr + 7] = self.hold_quat
        vadr = self.m.jnt_dofadr[jid]
        self.d.qvel[vadr:vadr + 6] = 0.0
        self.mj.mj_forward(self.m, self.d)

    def grasp(self):
        """EE 근처에서 가장 가까운 박스를 흡착. 성공 시 box index, 실패 시 None."""
        ee = self.ee_pos()
        best, bd = None, 1e9
        for i in self.box_ids:
            bid = self.box_bid(i)
            if bid < 0:
                continue
            top = self.d.xpos[bid] + np.array([0, 0, 0.1415])
            lat = float(np.hypot(top[0] - ee[0], top[1] - ee[1]))
            vert = abs(float(top[2] - ee[2]))
            # 석션은 컵 바로 아래 면에만 붙는다. '가장 가까운 박스'를 잡으면 옆 박스를
            # 측면으로 어긋나게 물어 그만큼 빗나가게 놓고, 이웃과 겹쳐 폭발한다(1차 시도).
            if lat > LATERAL_TOL_MM / 1000.0 or vert > SUCTION_REACH_MM / 1000.0:
                continue
            dist = float(np.linalg.norm(top - ee))
            if dist < bd:
                bd, best = dist, i
        if best is None:
            return None, bd * 1000.0 if bd < 1e8 else -1.0
        self._attach(best)
        for _ in range(60):
            self.mj.mj_step(self.m, self.d)
            self.steps += 1
            self._carry()
        return best, bd * 1000.0

    def _attach(self, best):
        bid = self.box_bid(best)
        self.hold_offset = (self.d.xpos[bid] - self.ee_pos()).copy()
        jid = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_JOINT, f"box{best}_j")
        adr = self.m.jnt_qposadr[jid]
        self.hold_quat = self.d.qpos[adr + 3:adr + 7].copy()
        self.held = best
        # 운반 중에는 이 박스의 충돌을 끈다. 위치를 매 스텝 강제하면서 다른 물체와 겹치면
        # 접촉 임펄스가 누적돼 해제 순간 박스가 수십 m 튕겨 나간다(1차 시도에서 발생).
        # 해제 시 복원하므로 스택 위 안착은 정상적으로 물리 계산된다.
        gid = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_GEOM, f"box{best}_g")
        self._saved_con = (int(self.m.geom_contype[gid]), int(self.m.geom_conaffinity[gid]))
        self._held_gid = gid
        self.m.geom_contype[gid] = 0
        self.m.geom_conaffinity[gid] = 0

    SOFT_RELEASE_STEPS = 200      # 0.4 s
    SOFT_SOLREF = (0.05, 1.0)     # 기본 접촉 timeconst 0.02 -> 0.05: 겹침을 천천히 푼다

    def release(self):
        """부착 해제 후 물리로 안착시킨다 (스택 위 낙하·정렬은 물리 계산).

        놓는 박스의 접촉을 처음 0.4 s 동안 부드럽게(solref timeconst 0.05) 둔다. 이송 중 박스는 충돌이 꺼져 있어 놓는 순간
        이웃과 몇 mm 겹쳐 있을 수 있고(픽 위치 오차 = 인식 오차), 기본 강성으로는 그 침투가 한 스텝에 임펄스로 풀려 박스가
        수십 cm~수 m 튕긴다(seed 504 oracle: box3 와 -4.7 mm 겹침 -> 26 mm 밀림 -> 다음 박스 -111 mm 겹침 -> 7 m 사출).
        실제 셀에서는 컵 컴플라이언스와 골판지 압축이 그 겹침을 흡수한다 — 그 대용이다. 0.4 s 뒤 원래 강성으로 되돌린다."""
        gid = getattr(self, "_held_gid", None)
        saved_solref = None
        if gid is not None:
            self.m.geom_contype[gid] = self._saved_con[0]
            self.m.geom_conaffinity[gid] = self._saved_con[1]
            saved_solref = self.m.geom_solref[gid].copy()
            self.m.geom_solref[gid] = self.SOFT_SOLREF
            self._held_gid = None
        self.held = None
        self.hold_offset = None
        self.mj.mj_forward(self.m, self.d)
        for k in range(600):
            if k == self.SOFT_RELEASE_STEPS and saved_solref is not None:
                self.m.geom_solref[gid] = saved_solref
            self.mj.mj_step(self.m, self.d)
            self.steps += 1

    def frame(self, rng):
        return self.rend.frame(self.d, noise="tof", rng=rng)


class ArmCell(Cell):
    """UR10e 가 실제로 움직이는 셀. 목표는 TCP 위치+자세, 관절 속도 한계로 보간해 위치 서보로 구동."""

    def __init__(self, layout, seed, arm_cfg, layout_cfg=None):
        from arm import Arm, box_body_geom_ids, rot_from_approach_yaw  # noqa: F401
        super().__init__(layout, seed, arm_cfg=dict(arm_cfg, meshes=False), layout_cfg=layout_cfg)
        self.arm = Arm(self.m, self.d)
        self.q_cmd = self.arm.get_q().copy()          # 홈 (Cell.__init__ 에서 정착 전에 맞춰 둠)
        self.box_gids = box_body_geom_ids(self.m)
        self._rot = rot_from_approach_yaw
        for _ in range(200):
            self.mj.mj_step(self.m, self.d)

    def ee_pos(self):
        return self.d.site_xpos[self.arm.site_id]

    @contextmanager
    def _payload_aware(self):
        """IK·접촉 검사 동안 들고 있는 박스를 **툴의 일부**로 본다: 충돌을 켜고, 후보 자세마다 TCP 에 맞춰 옮긴다.

        이송 중 박스의 충돌은 꺼 둔다(운동학적 이송). 그래서 IK 가 팔뚝(forearm)이 박스를 관통하는 자세를 골라도 아무 신호가
        없었고, 놓는 순간 충돌이 켜지며 40 mm 침투의 접촉 임펄스로 박스가 튕겨 나갔다(seed 507 oracle: 박스와 forearm_link_col1
        접촉 -40 mm 를 release 직후 실측). 실제 로봇도 자기 페이로드와 부딪히면 안 되므로, 검사 때만 박스를 켜서 컵 이외 링크가
        박스에 닿는 자세를 거른다. yield 값 = 검사에서 '컵 접촉 허용' 에 추가할 geom id 목록."""
        gid = getattr(self, "_held_gid", None)
        if self.held is None or gid is None:
            yield []
            return
        self.m.geom_contype[gid], self.m.geom_conaffinity[gid] = self._saved_con
        self.arm.payload_sync = self._carry
        try:
            yield [gid]
        finally:
            self.m.geom_contype[gid] = 0
            self.m.geom_conaffinity[gid] = 0
            self.arm.payload_sync = None

    def _restore(self, q, qv):
        self.d.qpos[self.arm.qadr] = q
        self.d.qvel[self.arm.dadr] = qv
        self.mj.mj_forward(self.m, self.d)
        self._carry()                       # 검사 중 옮겨졌던 박스를 실제 TCP 로

    def _nearest_yaw(self, yaw_deg: float) -> float:
        """컵은 원형이라 요 y 와 y±180 은 같은 집기다. 현재 툴 x 축 방위각에 가까운 쪽을 골라 불필요한 손목 180도 회전을 피한다.
        (검출 요는 [0,180) 으로 정규화돼 −1도짜리 박스가 179도로 온다 — 리뷰 지적.) 든 박스는 툴에 강체 부착이라 놓는 자세는 안 바뀐다."""
        _, R = self.arm.tcp()
        az = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        return min((yaw_deg + k for k in (-360.0, -180.0, 0.0, 180.0, 360.0)), key=lambda y: abs(y - az))

    def move_to(self, xyz_w, speed_mps=0.6, settle_steps=100, yaw_deg=0.0, allow_box_contact=False,
                check_path=True, **_):
        """IK -> (경로 충돌 검사) -> 관절 보간 실행. 반환 (ok, reason, err_mm).

        speed_mps 는 mocap 과의 인터페이스 호환용: 0.3 미만이면 '천천히'(속도 스케일 0.4) 로 해석한다.
        allow_box_contact: 하강·안착 구간에서 컵이 박스에 닿는 것은 정상이므로 그 접촉은 충돌로 보지 않는다.
        """
        goal = np.asarray(xyz_w, float)
        R = self._rot((0, 0, -1), self._nearest_yaw(yaw_deg))
        q_now = self.arm.get_q()
        qv_now = self.d.qvel[self.arm.dadr].copy()
        with self._payload_aware() as extra:
            allowed = (list(self.box_gids.values()) if allow_box_contact else []) + extra
            # 충돌 회피 IK: 목표 자세와 (check_path 면) 현재 명령 자세에서의 관절 보간 경로가 환경(들고 있는 박스 포함)에 닿지 않는 해
            r, hits = self.arm.solve_ik_free(goal, R, self.q_cmd, allowed_gids=allowed,
                                             path_from=(self.q_cmd if check_path else None), seeds=6, iters=200)
            self._restore(q_now, qv_now)
        if not r.ok:
            return False, "ik_unreachable", r.pos_err_m * 1000.0
        if hits and check_path:
            return False, "path_collision", 0.0
        self._execute(r.q, speed_scale=(0.4 if speed_mps < 0.3 else 1.0), settle_steps=settle_steps)
        return True, "ok", float(np.linalg.norm(self.ee_pos() - goal)) * 1000.0

    def move_linear(self, xyz_w, speed_mps=0.6, settle_steps=100, yaw_deg=0.0, allow_box_contact=False,
                    step_m=0.05, **_):
        """TCP 를 **직선**으로 옮긴다: 경로를 step_m 이하 구간으로 나눠 구간마다 현재 관절에서 IK 를 풀고 실행한다.

        move_to 의 관절 공간 보간은 이동이 크면 TCP 가 곡선을 그린다(팔꿈치·손목 각도와 TCP 위치의 관계가 비선형).
        놓은 박스 16 mm 위에서 52 cm 를 관절 보간으로 후퇴하자 컵이 박스를 54 mm 밀었고, 밀린 박스 위에 다음 박스가 놓이며
        접촉 임펄스로 튕겨 나갔다(seed 507 oracle 추적 — 이전에 '운동학적 이송 아티팩트' 라고만 적었던 튕김의 실제 원인).
        박스 근처의 하강·상승·후퇴는 이 함수로, 멀리 가는 이동은 move_to(경로 충돌 검사 포함)로 한다.
        구간마다 접촉을 검사해(allowed 제외) 닿으면 그 자리에서 멈추고 False 를 돌려준다."""
        goal = np.asarray(xyz_w, float)
        R = self._rot((0, 0, -1), self._nearest_yaw(yaw_deg))
        start = self.ee_pos().copy()
        n = max(int(np.ceil(float(np.linalg.norm(goal - start)) / max(step_m, 1e-3))), 1)
        q_now = self.arm.get_q()
        qv_now = self.d.qvel[self.arm.dadr].copy()
        qs, q_prev = [], self.q_cmd.copy()
        with self._payload_aware() as extra:
            allowed = (list(self.box_gids.values()) if allow_box_contact else []) + extra
            for i in range(1, n + 1):
                wp = start + (goal - start) * (i / n)
                r = self.arm.solve_ik(wp, R, q0=q_prev, iters=150)
                hits = self.arm.contacts(r.q, allowed) if r.ok else []
                if not r.ok or hits:
                    # 가까운 해가 없거나 닿으면: 여러 시드로 자세를 바꿔 보되, 직전 경유점에서의 관절 경로도 검사한다
                    r, hits = self.arm.solve_ik_free(wp, R, q_prev, allowed_gids=allowed, path_from=q_prev, seeds=4, iters=150)
                if not r.ok:
                    self._restore(q_now, qv_now)
                    return False, "ik_unreachable", r.pos_err_m * 1000.0
                if hits:
                    self._restore(q_now, qv_now)
                    return False, "path_collision", 0.0
                qs.append(np.asarray(r.q, float).copy())
                q_prev = qs[-1]
            self._restore(q_now, qv_now)
        self._execute_path(qs, speed_scale=(0.4 if speed_mps < 0.3 else 1.0), settle_steps=settle_steps)
        return True, "ok", float(np.linalg.norm(self.ee_pos() - goal)) * 1000.0

    def _execute_path(self, qs, speed_scale=1.0, settle_steps=100):
        """경유점 관절 자세들을 하나의 smoothstep 프로파일로 이어 실행한다 (경유점마다 멈추지 않는다).

        진행 변수 = 누적 관절 이동량(최대 관절 기준). 시간은 관절별 총 이동량의 사다리꼴 추정과 최고 속도 한계 중 큰 쪽."""
        pts = [self.q_cmd.copy()] + [np.asarray(q, float) for q in qs]
        seg = [float(np.max(np.abs(pts[i + 1] - pts[i]))) for i in range(len(pts) - 1)]
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        L = float(cum[-1])
        if L < 1e-9:
            for _ in range(settle_steps):
                self.mj.mj_step(self.m, self.d)
                self.steps += 1
                self._carry()
            self.q_cmd = pts[-1].copy()
            return
        segs = [np.abs(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
        total_dq = np.sum(segs, axis=0)
        # 구간 k 에서 관절 j 의 속도 = (ds/dt) * dq_jk / seg_k, ds/dt 의 최대는 1.5 L / T. 어느 구간·관절도 vmax 를 넘지 않게
        # (총 이동량 기준으로만 잡으면 구간마다 다른 관절이 지배할 때 넘는다 — 리뷰 지적)
        ratio = max((float(np.max(dq / self.arm.vmax)) / sk) for dq, sk in zip(segs, seg) if sk > 1e-12)
        t_peak = 1.5 * L * ratio
        T = max(self.arm.segment_time(np.zeros_like(total_dq), total_dq), t_peak) / max(speed_scale, 1e-3)
        steps = max(int(T * CTRL_HZ), 50)
        for i in range(steps):
            t = (i + 1) / steps
            s = (t * t * (3 - 2 * t)) * L
            k = int(np.searchsorted(cum, s, side="right")) - 1
            k = min(max(k, 0), len(seg) - 1)
            frac = (s - cum[k]) / seg[k] if seg[k] > 1e-12 else 1.0
            self.arm.set_ctrl(pts[k] + (pts[k + 1] - pts[k]) * min(frac, 1.0))
            self.mj.mj_step(self.m, self.d)
            self.steps += 1
            self._carry()
        for _ in range(settle_steps):
            self.mj.mj_step(self.m, self.d)
            self.steps += 1
            self._carry()
        self.q_cmd = pts[-1].copy()

    def _execute(self, q_goal, speed_scale=1.0, settle_steps=100):
        q0 = self.q_cmd.copy()
        # smoothstep 의 최고 속도는 평균의 1.5배다. 구간 시간을 사다리꼴 추정치로만 잡으면 최고 속도가 사양 한계를
        # 1.5배 넘는다 (검토에서 지적) -> 최고 속도가 vmax 를 넘지 않는 시간과 큰 쪽을 쓴다.
        dq = np.abs(np.asarray(q_goal, float) - q0)
        t_peak = 1.5 * float(np.max(dq / self.arm.vmax)) if dq.size else 0.0
        T = max(self.arm.segment_time(q0, q_goal), t_peak) / max(speed_scale, 1e-3)
        steps = max(int(T * CTRL_HZ), 50)
        for i in range(steps):
            t = (i + 1) / steps
            a = t * t * (3 - 2 * t)
            self.arm.set_ctrl(q0 + (q_goal - q0) * a)
            self.mj.mj_step(self.m, self.d)
            self.steps += 1
            self._carry()
        for _ in range(settle_steps):
            self.mj.mj_step(self.m, self.d)
            self.steps += 1
            self._carry()
        self.q_cmd = np.asarray(q_goal, float).copy()

    def _attach(self, best):
        """박스를 TCP 프레임에 강체 부착 (툴이 회전하면 박스도 따라 돈다)."""
        super()._attach(best)
        p, R = self.arm.tcp()
        bid = self.box_bid(best)
        self.hold_off_tool = R.T @ (self.d.xpos[bid] - p)
        Rb = self.d.xmat[bid].reshape(3, 3)
        self.hold_R_rel = R.T @ Rb

    def _carry(self):
        if self.held is None:
            return
        jid = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_JOINT, f"box{self.held}_j")
        if jid < 0:
            return
        p, R = self.arm.tcp()
        adr = self.m.jnt_qposadr[jid]
        self.d.qpos[adr:adr + 3] = p + R @ self.hold_off_tool
        q = np.zeros(4)
        self.mj.mju_mat2Quat(q, (R @ self.hold_R_rel).reshape(-1))
        self.d.qpos[adr + 3:adr + 7] = q
        vadr = self.m.jnt_dofadr[jid]
        self.d.qvel[vadr:vadr + 6] = 0.0
        self.mj.mj_forward(self.m, self.d)


def pick_order(boxes, dest_xy_mm, col_tol=80.0):
    """열 스캔 규칙(실제 픽 순서와 81% 일치)으로 픽 후보를 우선순위 순으로 나열.

    1순위는 기존과 동일(목적지 최근접 열에서 가장 먼 박스). 그 뒤로는 같은 규칙을
    남은 박스에 반복 적용한다. 풋프린트 게이트에 걸리면 다음 후보로 넘어가기 위함 —
    한 후보가 거부됐다고 사이클을 통째로 버리면 실제 셀에서는 처리량 손실이 된다.
    """
    if not boxes:
        return []
    c = np.array([b["center_mm"][:2] for b in boxes], float)
    dist = np.hypot(c[:, 0] - dest_xy_mm[0], c[:, 1] - dest_xy_mm[1])
    remaining, order = list(range(len(boxes))), []
    while remaining:
        p = min(remaining, key=lambda i: dist[i])
        col = [i for i in remaining if abs(c[i, 0] - c[p, 0]) < col_tol]
        nxt = max(col, key=lambda i: dist[i])
        order.append(nxt)
        remaining.remove(nxt)
    return order


def run_episode(layout, seed, oracle=False, conf_min=0.0, use_footprint=True, verbose=False, arm_cfg=None,
                arm_layout=False, layout_cfg=None, layer_roi_mm=None, temporal_prior=False):
    cell = (ArmCell(layout, seed, arm_cfg, layout_cfg=layout_cfg) if arm_cfg
            else Cell(layout, seed, arm_layout=arm_layout, layout_cfg=layout_cfg))
    rng = np.random.default_rng(10_000 + seed)
    T = topdown_camera_transform(cam_height_mm=CAM_H * 1000.0)   # 트윈은 외참을 정확히 안다
    dest_w = np.array(dest_world_xy())
    dest_cam_xy = (dest_w[0] * 1000.0, -dest_w[1] * 1000.0)
    n_start = len(layout)
    log = {"picks": [], "seed": seed, "oracle": oracle, "n_boxes": n_start, "arm": bool(arm_cfg)}
    placed = 0
    stack_h = DECK_H
    prior_top = None             # 시간 사전: 직전 프레임에서 박스가 나온 층 깊이 (temporal_prior 일 때)
    filled = set()               # 목적지에서 이미 쓴 (층, 슬롯)
    unreachable_ids = set()      # 이 에피소드에서 도달 불가로 판정된 박스 (같은 박스를 매 사이클 다시 시도하지 않게)
    try:
        for step in range(n_start):
            step_start = cell.steps
            f = cell.frame(rng)
            if oracle:
                gt = ground_truth(cell.m, cell.d)
                boxes = [{"center_mm": (g["center_mm"][0], g["center_mm"][1], g["top_d_mm"]),
                          "depth_mm": g["top_d_mm"], "dims_mm": g["dims_mm"],
                          "ang_deg": g["ang_deg"], "normal": (0, 0, -1), "confidence": 1.0,
                          "rect_px": None} for g in gt]
            else:
                top_sel, _, det = detect_boxes_v2(f, layer_roi_mm=layer_roi_mm,
                                                 prior_top_mm=(prior_top if temporal_prior else None))
                prior_top = float(top_sel) if (det and np.isfinite(top_sel)) else None
                boxes = [b for b in det if b.get("confidence", 0) >= conf_min]
                for b in boxes:
                    b["center_mm"] = (b["center_mm"][0], b["center_mm"][1], b["depth_mm"])
                    b["normal"] = (0.0, 0.0, -1.0)
            # 소스 팔레트 ROI 로 제한. 목적지 스택은 소스와 같은 높이가 되므로 같은 층으로 검출되고,
            # ROI 가 없으면 플래너가 방금 옮긴 박스를 다시 집는다(1차 시도에서 발생: 같은 박스 4회 픽).
            # 실제 셀도 인식 ROI 를 소스 팔레트로 한정한다.
            boxes = [b for b in boxes
                     if abs(b["center_mm"][0]) < SOURCE_ROI_MM and abs(b["center_mm"][1]) < SOURCE_ROI_MM]
            # 앞서 도달 불가로 판정된 위치(반경 10 cm)는 후보에서 뺀다
            boxes = [b for b in boxes if not any(np.hypot(b["center_mm"][0] - u[0], b["center_mm"][1] - u[1]) < 100.0
                                                 for u in unreachable_ids)]
            if not boxes:
                log["picks"].append({"step": step, "result": "source_empty" if not unreachable_ids else "only_unreachable_left"})
                break
            order = pick_order(boxes, dest_cam_xy)
            b, n_rej, reach_fail = None, 0, None
            for i in order:
                cand = boxes[i]
                if use_footprint and not oracle and cand.get("rect_px") is not None:
                    fr = {"D": f["D"], "valid": f["D"] < 16000}
                    fp = suction_footprint_ok(fr, cand)
                    if not fp["ok"]:
                        n_rej += 1
                        if verbose:
                            print(f"    step {step}: 후보 거부 ({fp['reason']}) -> 다음 후보")
                        continue
                tgt_c = cam_to_world(np.array(cand["center_mm"], float))
                yaw_c = float(box_to_pick_pose(cand, T, clearance_mm=180.0).yaw_deg)
                if arm_cfg:
                    ok, why, _ = cell.move_to(tgt_c + np.array([0, 0, 0.20]), yaw_deg=yaw_c)
                    if not ok:
                        reach_fail = why
                        unreachable_ids.add((cand["center_mm"][0], cand["center_mm"][1]))
                        if verbose:
                            print(f"    step {step}: 후보 {why} -> 다음 후보")
                        continue
                b = cand
                break
            if b is None:
                log["picks"].append({"step": step, "result": reach_fail or "rejected_footprint",
                                     "candidates_rejected": n_rej})
                continue
            pose = box_to_pick_pose(b, T, clearance_mm=180.0)
            tgt = cam_to_world(np.array(b["center_mm"], float))
            yaw = float(pose.yaw_deg)          # base 좌표의 장축 요 (인식 노드와 같은 규약)
            if not arm_cfg:
                cell.move_to(tgt + np.array([0, 0, 0.20]))
            # 박스 근처(하강·상승·후퇴)는 TCP 직선 이동, 멀리 가는 이송은 관절 보간 + 경로 충돌 검사 (ArmCell.move_linear 주석)
            ok, why, _ = cell.move_linear(tgt + np.array([0, 0, 0.016]), speed_mps=0.25, yaw_deg=yaw, allow_box_contact=True)
            if not ok:
                log["picks"].append({"step": step, "result": f"descent_{why}"})
                cell.move_linear(tgt + np.array([0, 0, 0.35]), yaw_deg=yaw, allow_box_contact=True)
                continue
            held, gap = cell.grasp()
            if held is None:
                log["picks"].append({"step": step, "result": "grasp_miss",
                                     "gap_mm": round(gap, 1)})
                cell.move_linear(tgt + np.array([0, 0, 0.35]), yaw_deg=yaw, allow_box_contact=True)
                if verbose:
                    print(f"    step {step}: 흡착 실패 (gap {gap:.0f}mm)")
                continue
            before = cell.d.xpos[cell.box_bid(held)].copy()
            t_pick0 = cell.steps
            cell.move_linear(tgt + np.array([0, 0, 0.45]), yaw_deg=yaw, allow_box_contact=True)
            # 목적지 배치: 실제 이적재 공정처럼 4x3 격자 슬롯에 층층이 쌓는다.
            # (모든 박스를 목적지 중심 한 점에 떨어뜨리면 서로 부딪혀 무너진다 — 1차 시도에서 발생)
            # 팔이 있으면 '팔이 닿는 슬롯' 을 골라야 한다: 고정 받침대는 먼 슬롯에 못 닿으므로, 같은 층의 빈 슬롯을
            # 받침대에서 가까운 순으로 시도한다 (한 슬롯에 갇혀 모든 픽이 실패하던 1차 시도 교훈).
            slots = dest_slots()                 # 3열 x 3행 (4열째는 소스 박스와 겹쳐 제외 — cell_scene.DEST_COLS)
            layer = len(filled) // len(slots)
            free_slots = [s for s in range(len(slots)) if (layer, s) not in filled]
            if arm_cfg:
                bx, by = arm_cfg["base_xy"]
                free_slots.sort(key=lambda s: np.hypot(dest_w[0] + grid_xy(*slots[s])[0] - bx,
                                                       dest_w[1] + grid_xy(*slots[s])[1] - by))
            stack_h = DECK_H + BOX[2] * layer + BOX[2]
            ok, why, slot = False, "place_ik_unreachable", None
            for s in free_slots:
                gx, gy = grid_xy(*slots[s])
                drop = np.array([dest_w[0] + gx, dest_w[1] + gy, 0.0])
                # 집은 자세(요) 그대로 이송한다. 목적지에서 요를 0 으로 되돌리면 검출 각도가 90 으로 나온 박스가 90도 돌아간 채
                # 이웃 위에 놓여 튕겨 나간다 (ROS 트윈 연결 실행에서 발견 — 12박스 중 6 이 그렇게 misplaced 됐다)
                ok, why, _ = cell.move_to(np.array([drop[0], drop[1], stack_h + 0.45]), yaw_deg=yaw)
                if not ok:
                    continue
                # 직선 하강도 성공해야 그 슬롯에 놓는다. 하강 결과를 안 보고 놓으면 팔뚝이 박스를 관통한 채 놓는 일이 생긴다
                ok, why, _ = cell.move_linear(np.array([drop[0], drop[1], stack_h + 0.03]), speed_mps=0.25, yaw_deg=yaw,
                                              allow_box_contact=True)
                if ok:
                    slot = s
                    break
                why = f"descent_{why}"
                cell.move_linear(np.array([drop[0], drop[1], stack_h + 0.45]), yaw_deg=yaw, allow_box_contact=True)
            if not ok:
                # 목적지에 못 닿는다: 박스를 제자리에 돌려놓고 실패로 기록 (실제 셀이면 다른 슬롯 계획이 필요)
                cell.move_to(tgt + np.array([0, 0, 0.45]), yaw_deg=yaw, check_path=False)
                cell.move_linear(tgt + np.array([0, 0, 0.02]), speed_mps=0.25, yaw_deg=yaw, allow_box_contact=True)
                cell.release()
                cell.move_linear(tgt + np.array([0, 0, 0.45]), yaw_deg=yaw, allow_box_contact=True)
                log["picks"].append({"step": step, "result": f"place_{why}", "box": held})
                unreachable_ids.add((b["center_mm"][0], b["center_mm"][1]))
                continue
            cell.release()
            cell.move_linear(np.array([drop[0], drop[1], stack_h + 0.55]), yaw_deg=yaw, allow_box_contact=True)
            after = cell.d.xpos[cell.box_bid(held)]
            moved = (float(np.linalg.norm(after[:2] - drop[:2])) < 0.16
                     and after[2] > DECK_H - 0.05)
            placed += int(moved)
            filled.add((layer, slot))            # 잘못 놓였어도 슬롯은 쓴 것으로 본다 (그 위에 또 놓지 않게)
            log["picks"].append({"step": step, "result": "placed" if moved else "misplaced",
                                 "box": held, "gap_mm": round(gap, 1),
                                 "cycle_s": round((cell.steps - step_start) / CTRL_HZ, 2),
                                 "transfer_s": round((cell.steps - t_pick0) / CTRL_HZ, 2),
                                 "place_err_mm": round(float(np.linalg.norm(after[:2] - drop[:2])) * 1000, 1),
                                 "landed_z": round(float(after[2]), 3),
                                 "travel_mm": round(float(np.linalg.norm(after - before)) * 1000, 1),
                                 "yaw_deg": pose.yaw_deg, "conf": round(b.get("confidence", 0), 3)})
            if verbose:
                print(f"    step {step}: {'배치' if moved else '배치실패'} box{held} gap {gap:.0f}mm")
    finally:
        cell.close()
    log["placed"] = placed
    log["success_rate"] = round(placed / max(n_start, 1), 3)
    return log


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--boxes", type=int, default=6, help="에피소드당 상층 박스 수")
    ap.add_argument("--arm", choices=["none", "fixed", "track"], default="none")
    ap.add_argument("--arm-layout", action="store_true",
                    help="팔 없이(mocap) 돌리되 팔 씬과 같은 설비 배치를 쓴다 — 실행기 비교를 같은 장면에서 하기 위함")
    ap.add_argument("--layer-roi", type=float, default=0.0, help="층 히스토그램을 팔레트 영역(카메라 XY 반경 mm)으로 한정. 0 = 기존(화면 중앙)")
    ap.add_argument("--prior", action="store_true", help="직전 프레임의 층 깊이를 다음 프레임의 사전으로 (잔여 1~3개 층 점프 방지)")
    ap.add_argument("--tag", default="", help="결과 파일 이름 접미사 (예: _prior)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    arm_cfg = None if args.arm == "none" else arm_cfg_from_results(track=(args.arm == "track"))
    # 팔 변형(고정·트랙)과 --arm-layout mocap 은 모두 같은 설비 제외 구역(트랙 구성의 것)을 써서 장면이 정확히 같다
    layout_cfg = shared_layout_cfg() if (arm_cfg or args.arm_layout) else None
    if arm_cfg:
        print(f"팔: UR10e {args.arm} cfg={arm_cfg}  layout={layout_cfg}")
    elif args.arm_layout:
        print(f"mocap 석션 EE, 팔 씬과 같은 설비 배치 layout={layout_cfg}")

    cells = [(c, r) for r in range(N_ROW) for c in range(N_COL)]
    results = {"arm": args.arm, "arm_cfg": ({k: (list(v) if isinstance(v, tuple) else v) for k, v in arm_cfg.items()}
                                            if arm_cfg else None)}
    for mode in ("perception", "oracle"):
        eps = []
        print(f"--- {mode}")
        for e in range(args.episodes):
            rng = np.random.default_rng(500 + e)
            idx = sorted(rng.permutation(N_COL * N_ROW)[:args.boxes])
            layout = [(cells[i][0], cells[i][1], 0) for i in idx]
            log = run_episode(layout, seed=500 + e, oracle=(mode == "oracle"),
                              verbose=args.verbose, arm_cfg=arm_cfg, arm_layout=args.arm_layout,
                              layout_cfg=layout_cfg, layer_roi_mm=(args.layer_roi or None), temporal_prior=args.prior)
            eps.append(log)
            print(f"  ep{e}: {log['placed']}/{log['n_boxes']} 배치 "
                  f"({', '.join(sorted({p['result'] for p in log['picks']}))})")
        tot = sum(e["n_boxes"] for e in eps)
        ok = sum(e["placed"] for e in eps)
        reasons = {}
        for e in eps:
            for p in e["picks"]:
                reasons[p["result"]] = reasons.get(p["result"], 0) + 1
        cyc = [p["cycle_s"] for e in eps for p in e["picks"] if "cycle_s" in p]
        results[mode] = {"episodes": len(eps), "boxes": tot, "placed": ok,
                         "success_rate": round(ok / max(tot, 1), 3),
                         "outcomes": reasons, "logs": eps,
                         "cycle_s": ({"mean": round(float(np.mean(cyc)), 2),
                                      "p95": round(float(np.percentile(cyc, 95)), 2),
                                      "n": len(cyc)} if cyc else None)}
        c = results[mode]["cycle_s"]
        print(f"  => {mode}: {ok}/{tot} = {results[mode]['success_rate']:.1%}  {reasons}"
              + (f"  cycle {c['mean']:.1f}s (p95 {c['p95']:.1f})" if c else ""))

    gap = results["oracle"]["success_rate"] - results["perception"]["success_rate"]
    results["perception_cost"] = round(gap, 3)
    print(f"\n인식 때문에 잃는 성공률: {gap:+.1%} "
          f"(oracle {results['oracle']['success_rate']:.1%} vs 인식 {results['perception']['success_rate']:.1%})")
    if args.arm != "none":
        name = f"closed_loop_arm_{args.arm}{args.tag}.json"
    else:
        name = ("closed_loop_mocap_armlayout" if args.arm_layout else "closed_loop") + args.tag + ".json"
    results["layer_roi_mm"] = args.layer_roi or None
    results["temporal_prior"] = bool(args.prior)
    results["arm_layout"] = bool(args.arm_layout)
    results["layout_cfg"] = ({k: (list(v) if isinstance(v, tuple) else v) for k, v in layout_cfg.items()}
                             if layout_cfg else None)
    (OUT / name).write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved", OUT / name)


if __name__ == "__main__":
    main()
