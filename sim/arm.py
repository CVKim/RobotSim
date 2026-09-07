# -*- coding: utf-8 -*-
"""UR10e 팔을 셀 트윈에 얹는 유틸 + 역기구학(IK) + 충돌 검사 + 관절 속도 기반 사이클 시간.

지금까지 트윈의 석션 EE 는 mocap 으로 순간이동했다. 그래서 도달 범위, 관절 한계, 팔이 옆 박스나 설비에
부딪히는지, 관절 속도로 정해지는 사이클 시간이 전부 평가되지 않았다. 이 모듈이 그 자리를 채운다.

  팔 모델   MuJoCo Menagerie 의 UR10e (도달 1.3 m, 6축). sim/assets/ur10e/ur10e.xml (BSD-3). 메시는 선택.
  좌표계    트윈 월드 = 탑다운 카메라의 base_link (x = ToF X, y = −ToF Y, z = CAM_H − D). 팔 베이스는 받침대 위.
  IK        감쇠 최소제곱(DLS). 목표 = TCP 위치 + 자세(툴 z 축 = 접근 방향, x 축 = 요). 관절 한계 클램프.
  충돌      mj_forward 로 접촉을 계산해 팔 지오메트리가 환경(박스·데크·설비·컨베이어)에 닿는지 본다.
            흡착 컵과 '집으려는 박스' 의 접촉만 허용.
  시간      관절별 최대 속도(UR10e 사양: 베이스·숄더 120°/s, 나머지 180°/s) 의 사다리꼴 프로파일로 구간 시간 추정.

이 모듈은 '평가·계획' 용이고, 물리로 팔을 실제 구동하는 것은 tools/twin_closed_loop.py 의 --arm 모드가 한다.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

UR10E_DIR = Path(__file__).resolve().parent / "assets" / "ur10e"
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
ACTUATORS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
VMAX_RAD_S = np.radians([120.0, 120.0, 180.0, 180.0, 180.0, 180.0])   # UR10e 사양
ACCEL_TIME_S = 0.25            # 최대 속도까지 걸리는 시간 (사다리꼴 프로파일 가속 구간)
TRACK_VMAX = 1.0               # 리니어 트랙 m/s
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
TOOL_LEN = 0.15                # 플랜지 -> 흡착 컵 면 (m)
CUP_R = 0.055
TRACK_JOINT = "track_joint"


# ------------------------------------------------------------------ MJCF 조립

def ur10e_parts(with_meshes: Optional[bool] = None) -> dict:
    """Menagerie ur10e.xml 을 읽어 우리 씬에 끼워 넣을 조각(문자열)들을 돌려준다.

    반환 dict: default(클래스 트리), asset(재질·메시), body(base 바디 트리 + 툴), actuator, meshdir
    with_meshes=None 이면 assets/*.obj 가 있을 때만 메시를 쓴다. 없으면 시각 메시를 빼고 충돌 캡슐을 보이게 한다.
    """
    root = ET.parse(UR10E_DIR / "ur10e.xml").getroot()
    mesh_dir = UR10E_DIR / "assets"
    if with_meshes is None:
        with_meshes = mesh_dir.exists() and any(mesh_dir.glob("*.obj"))

    default = root.find("default")
    asset = root.find("asset")
    body = root.find("worldbody").find("body[@name='base']")
    actuator = root.find("actuator")

    if not with_meshes:
        for m in list(asset.findall("mesh")):
            asset.remove(m)
        for g in body.iter("geom"):
            if g.get("class") == "visual":
                g.set("_drop", "1")
        for parent in body.iter():
            for g in list(parent):
                if g.tag == "geom" and g.get("_drop") == "1":
                    parent.remove(g)
        for d in default.iter("default"):
            if d.get("class") == "collision":
                geom = d.find("geom")
                geom.set("group", "1")
                geom.set("rgba", "0.75 0.77 0.80 1")

    # 흡착 툴: 플랜지 사이트(attachment_site, z 축 = 플랜지 바깥)와 같은 자세로 툴 바디를 단다
    wrist3 = None
    for b in body.iter("body"):
        if b.get("name") == "wrist_3_link":
            wrist3 = b
    site = wrist3.find("site[@name='attachment_site']")
    tool = ET.SubElement(wrist3, "body", name="tool", pos=site.get("pos"), quat=site.get("quat"))
    ET.SubElement(tool, "geom", name="tool_tube", type="cylinder", size="0.03 0.06", pos="0 0 0.06",
                  rgba="0.2 0.2 0.22 1", contype="1", conaffinity="1", group="1", mass="0.6")
    ET.SubElement(tool, "geom", name="suction_g", type="cylinder", size=f"{CUP_R} 0.015", pos=f"0 0 {TOOL_LEN - 0.015}",
                  rgba="0.15 0.6 0.8 1", contype="1", conaffinity="1", group="1", mass="0.4",
                  friction="1.2 0.02 0.001")
    ET.SubElement(tool, "site", name="tcp", pos=f"0 0 {TOOL_LEN}", size="0.006", rgba="1 0.2 0.2 1", group="1")
    # 중력 보상: 위치 서보(kp 5000)만으로는 12.9 kg 상완이 1 m 밖에서 1~2 cm 처진다. 실제 로봇 제어기는 중력을
    # 보상하므로 바디마다 gravcomp=1 을 켜서 서보가 자세 오차만 다루게 한다.
    for b in body.iter("body"):
        b.set("gravcomp", "1")
    # 충돌 캡슐: 환경과 접촉하도록 contype/conaffinity 를 켜고, 접촉 로그를 읽기 쉽게 링크 이름을 붙인다
    # (인접 링크 캡슐의 겹침은 MuJoCo 가 부모-자식 바디 쌍을 걸러 주므로 접촉으로 나오지 않는다. 그래서 Arm.contacts 는
    #  팔-팔 접촉도 자기 충돌로 보고한다 — 검토에서 '전부 무시' 가 비인접 링크 충돌을 숨긴다고 지적받아 고침)
    for b in body.iter("body"):
        k = 0
        for g in b.findall("geom"):
            if g.get("class") in ("collision", "eef_collision"):
                g.set("contype", "1")
                g.set("conaffinity", "1")
                if not g.get("name"):
                    g.set("name", f"{b.get('name', 'link')}_col{k}")
                    k += 1

    s = lambda el: ET.tostring(el, encoding="unicode")  # noqa: E731
    return dict(default="".join(s(c) for c in default),
                asset="".join(s(c) for c in asset),
                body=s(body), actuator="".join(s(c) for c in actuator),
                meshdir=str(mesh_dir) if with_meshes else "")


TRACK_STACK_H = 0.12       # 레일(0.06) + 캐리지(0.06): 트랙 구성의 팔 베이스는 받침대보다 이만큼 높다


def arm_mount_xml(parts: dict, base_xy=(-0.6, 0.85), pedestal_h=0.8, track_range: float = 0.0,
                  pedestal_r=0.16) -> str:
    """받침대(+선택 리니어 트랙) 위에 UR10e 를 얹은 바디 트리 문자열.

    주의: 트랙이 있으면 팔 베이스 높이 = pedestal_h + TRACK_STACK_H. 고정 구성과 베이스 높이를 맞춰 비교하려면
    트랙 쪽 pedestal_h 를 그만큼 낮춘다 (tools/twin_arm_reach.py 가 그렇게 한다)."""
    bx, by = base_xy
    ped = (f'    <body name="arm_mount" pos="{bx:.4f} {by:.4f} 0">\n'
           f'      <geom name="pedestal" type="cylinder" size="{pedestal_r:.3f} {pedestal_h / 2:.4f}" '
           f'pos="0 0 {pedestal_h / 2:.4f}" rgba="0.5 0.52 0.55 1"/>\n')
    if track_range > 0:
        # 레일과 캐리지는 서로 충돌하지 않게 둔다(contype/conaffinity 0). arm_mount 는 월드에 고정된 바디라 MuJoCo 의
        # 부모-자식 충돌 제외가 적용되지 않고, 받침대 높이에 따라 두 상자가 부동소수점 수준으로 겹치면 접촉 마찰이
        # 액추에이터 힘(3000 N)을 통째로 먹어 트랙이 움직이지 않았다 (받침대 1.08 m 에서 실제로 발생, 1.2 m 에서는 우연히 무접촉).
        ped += (f'      <geom name="track_rail" type="box" size="{track_range + 0.2:.3f} 0.08 0.03" '
                f'pos="0 0 {pedestal_h + 0.03:.4f}" rgba="0.35 0.36 0.38 1" contype="0" conaffinity="0"/>\n'
                f'      <body name="carriage" pos="0 0 {pedestal_h + 0.06:.4f}">\n'
                f'        <joint name="{TRACK_JOINT}" type="slide" axis="1 0 0" range="{-track_range:.3f} {track_range:.3f}" '
                f'damping="200"/>\n'
                f'        <geom name="carriage_g" type="box" size="0.18 0.18 0.03" pos="0 0 0.03" rgba="0.3 0.3 0.32 1" '
                f'contype="0" conaffinity="0"/>\n'
                f'        <body pos="0 0 0.06">\n{parts["body"]}\n        </body>\n'
                f'      </body>\n')
    else:
        ped += f'      <body pos="0 0 {pedestal_h:.4f}">\n{parts["body"]}\n      </body>\n'
    ped += '    </body>\n'
    return ped


def track_actuator_xml(track_range: float) -> str:
    if track_range <= 0:
        return ""
    return (f'    <general name="track" joint="{TRACK_JOINT}" biastype="affine" gainprm="20000" '
            f'biasprm="0 -20000 -2000" ctrlrange="{-track_range:.3f} {track_range:.3f}" forcerange="-3000 3000"/>\n')


# ------------------------------------------------------------------ 운동학 / IK

def rot_from_approach_yaw(approach=(0.0, 0.0, -1.0), yaw_deg: float = 0.0) -> np.ndarray:
    """툴 자세: z 축 = 접근 방향(박스 쪽), x 축 = 요 방향(상면 장축). 열 = 툴 축을 월드로."""
    z = np.asarray(approach, float)
    z = z / np.linalg.norm(z)
    x0 = np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg)), 0.0])
    x = x0 - z * float(x0 @ z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def rotvec_between(R_target: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """R_cur 를 R_target 으로 돌리는 회전벡터(월드 프레임)."""
    R = R_target @ R_cur.T
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = math.acos(cos)
    if ang < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-9:                       # 180도 근방: 대각 성분에서 축 복원
        w, v = np.linalg.eigh(R)
        axis = v[:, np.argmin(np.abs(w - 1.0))]
        return axis / np.linalg.norm(axis) * ang
    return axis / n * ang


@dataclass
class IKResult:
    q: np.ndarray
    ok: bool
    pos_err_m: float
    rot_err_deg: float
    iters: int


class Arm:
    """모델 안의 UR10e(+트랙) 핸들: 관절 인덱스, FK, IK, 충돌 검사, 시간 추정."""

    def __init__(self, model, data, site: str = "tcp"):
        import mujoco
        self.mj, self.m, self.d = mujoco, model, data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site_id < 0:
            raise ValueError("tcp site not found — build the scene with arm=...")
        names = list(JOINTS)
        tj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, TRACK_JOINT)
        self.has_track = tj >= 0
        if self.has_track:
            names = [TRACK_JOINT] + names
        self.jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.jids])
        self.lo = np.array([model.jnt_range[j][0] for j in self.jids])
        self.hi = np.array([model.jnt_range[j][1] for j in self.jids])
        self.vmax = np.concatenate([[TRACK_VMAX], VMAX_RAD_S]) if self.has_track else VMAX_RAD_S.copy()
        self.n = len(self.jids)
        act_names = (["track"] if self.has_track else []) + ACTUATORS
        self.act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in act_names]
        # 팔 지오메트리: 'base' 바디 서브트리 + 캐리지 (받침대는 정적이라 제외)
        base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        sub = set()
        for b in range(model.nbody):
            k = b
            while k > 0:
                if k == base_bid:
                    sub.add(b)
                    break
                k = model.body_parentid[k]
        self.arm_gids = {g for g in range(model.ngeom) if model.geom_bodyid[g] in sub}
        self.cup_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "suction_g")
        self.home = np.concatenate([[0.0], HOME]) if self.has_track else HOME.copy()

    # ---- 상태 ----------------------------------------------------------
    def get_q(self) -> np.ndarray:
        return self.d.qpos[self.qadr].copy()

    def set_q(self, q):
        self.d.qpos[self.qadr] = np.asarray(q, float)
        self.d.qvel[self.dadr] = 0.0
        self.mj.mj_forward(self.m, self.d)

    def set_ctrl(self, q):
        for a, v in zip(self.act_ids, np.asarray(q, float)):
            self.d.ctrl[a] = v

    def tcp(self):
        """(pos(3), R(3x3)) — 현재 qpos 기준 (mj_forward 된 상태여야 함)."""
        return self.d.site_xpos[self.site_id].copy(), self.d.site_xmat[self.site_id].reshape(3, 3).copy()

    def fk(self, q):
        saved = self.get_q()
        self.set_q(q)
        p, R = self.tcp()
        self.set_q(saved)
        return p, R

    # ---- IK -------------------------------------------------------------
    def solve_ik(self, target_pos, target_R, q0=None, iters=300, tol_pos=1e-3, tol_rot_deg=0.5,
                 damping=1e-2, max_step=0.3, track_weight=0.3) -> IKResult:
        """감쇠 최소제곱 IK. q0 에서 시작해 관절 한계 안에서 수렴. 트랙이 있으면 트랙 축은 가중치를 낮춰 팔을 먼저 쓴다."""
        q = np.array(self.home if q0 is None else q0, float)
        saved = self.get_q()
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        W = np.ones(self.n)
        if self.has_track:
            W[0] = track_weight
        best = None
        for it in range(iters):
            self.set_q(q)
            p, R = self.tcp()
            e_p = np.asarray(target_pos, float) - p
            e_r = rotvec_between(np.asarray(target_R, float), R)
            pe, re = float(np.linalg.norm(e_p)), float(np.degrees(np.linalg.norm(e_r)))
            if best is None or pe + 0.01 * re < best[0]:
                best = (pe + 0.01 * re, q.copy(), pe, re, it)
            if pe < tol_pos and re < tol_rot_deg:
                self.set_q(saved)
                return IKResult(q, True, pe, re, it)
            self.mj.mj_jacSite(self.m, self.d, jacp, jacr, self.site_id)
            J = np.vstack([jacp[:, self.dadr], jacr[:, self.dadr]]) * W[None, :]
            e = np.concatenate([e_p, e_r])
            lam = damping * (1.0 + 10.0 * min(pe, 0.5))
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e) * W
            nrm = float(np.max(np.abs(dq)))
            if nrm > max_step:
                dq *= max_step / nrm
            q = np.clip(q + dq, self.lo, self.hi)
        self.set_q(saved)
        _, qb, pe, re, it = best
        return IKResult(qb, False, pe, re, it)

    def solve_ik_multi(self, target_pos, target_R, q0=None, seeds: int = 4, rng=None, **kw) -> IKResult:
        """q0 에서 먼저 풀고, 실패하면 홈·무작위 시드로 재시도. 성공한 해 중 q0 에서 가장 가까운 것을 고른다."""
        rng = rng or np.random.default_rng(0)
        q_ref = np.array(self.home if q0 is None else q0, float)
        cands = []
        r = self.solve_ik(target_pos, target_R, q_ref, **kw)
        if r.ok:
            return r
        cands.append(r)
        starts = [self.home.copy()]
        for _ in range(seeds):
            s = self.home + rng.normal(0, 0.6, self.n)
            if self.has_track:
                s[0] = rng.uniform(self.lo[0], self.hi[0])
            starts.append(np.clip(s, self.lo, self.hi))
        for s in starts:
            r = self.solve_ik(target_pos, target_R, s, **kw)
            cands.append(r)
            if r.ok:
                break
        oks = [c for c in cands if c.ok]
        if oks:
            return min(oks, key=lambda c: float(np.linalg.norm(c.q - q_ref)))
        return min(cands, key=lambda c: c.pos_err_m)

    def solve_ik_free(self, target_pos, target_R, q0=None, allowed_gids: Sequence[int] = (), path_from=None,
                      seeds: int = 8, rng=None, path_n: int = 10, **kw):
        """충돌을 피하는 IK: 여러 시드(홈, 팔꿈치 위/아래 변형, 무작위)로 해를 구해 목표 자세와 (path_from 이 있으면)
        관절 보간 경로가 환경과 닿지 않는 첫 해를 돌려준다. 6축 팔은 같은 목표에 여러 자세가 있어서, 처음 수렴한 해가
        팔꿈치를 아래로 떨어뜨려 옆 박스를 치는 경우가 많다 — 실제 플래너가 하듯 다른 자세를 찾아본다.

        반환 (IKResult, contacts) — contacts 가 빈 리스트면 무충돌. 무충돌 해가 없으면 접촉이 가장 적은 해."""
        rng = rng or np.random.default_rng(0)
        q_ref = np.array(self.home if q0 is None else q0, float)
        off = 1 if self.has_track else 0
        # (시작 자세, 트랙 가중치) 후보. 트랙 가중치 0.3 은 '팔을 먼저 쓰라'는 뜻인데, 작업 공간 가장자리에서는 그 때문에
        # 트랙이 안 움직여 수렴에 실패한다(실측 1층 포즈 8건) -> 같은 시작 자세를 가중치 1.0 으로도 시도한다.
        starts = [(q_ref, 0.3)]
        if self.has_track:
            starts.append((q_ref, 1.0))
        starts.append((np.clip(q_ref + rng.normal(0, 0.25, self.n), self.lo, self.hi), 1.0))
        starts.append((self.home.copy(), 0.3))
        for k in range(seeds):
            s = self.home.copy()
            # 팔꿈치 위(elbow-up)/아래 두 가지 패밀리를 번갈아 시도하고 손목을 흔든다
            s[off + 1] = rng.uniform(-2.6, -0.4)                  # shoulder_lift
            s[off + 2] = rng.uniform(0.4, 2.6) * (1 if k % 2 == 0 else -1)   # elbow
            s[off + 3] = rng.uniform(-3.0, 0.0)                   # wrist_1
            s[off + 0] = q_ref[off + 0] + rng.normal(0, 0.5)      # shoulder_pan (목표 방향 근처)
            if self.has_track:
                s[0] = np.clip(q_ref[0] + rng.normal(0, 0.3), self.lo[0], self.hi[0])
            starts.append((np.clip(s, self.lo, self.hi), 1.0 if k % 3 == 2 else 0.3))
        best, best_n = None, 10 ** 9
        for s, tw in starts:
            r = self.solve_ik(target_pos, target_R, s, track_weight=tw, **kw)
            if not r.ok:
                continue
            c = self.contacts(r.q, allowed_gids)
            if not c and path_from is not None:
                c = self.path_collides(path_from, r.q, n=path_n, allowed_gids=allowed_gids) or []
            if not c:
                return r, []
            if len(c) < best_n:
                best, best_n = (r, c), len(c)
        if best is not None:
            return best
        r = self.solve_ik_multi(target_pos, target_R, q_ref, seeds=2, rng=rng, **kw)
        if r.ok:                                   # 마지막 시도가 풀렸으면 진짜 접촉 목록을 돌려준다
            c = self.contacts(r.q, allowed_gids)
            if not c and path_from is not None:
                c = self.path_collides(path_from, r.q, n=path_n, allowed_gids=allowed_gids) or []
            return r, c
        return r, [("unreachable", "", 0.0)]

    # ---- 충돌 ------------------------------------------------------------
    def contacts(self, q=None, allowed_gids: Sequence[int] = ()) -> list:
        """q 자세에서 팔 지오메트리가 환경과 이루는 접촉 (팔-팔 자기 접촉과 allowed 지오메트리와의 컵 접촉은 제외).
        반환 [(팔 geom 이름, 상대 geom 이름, 침투 깊이 m)]"""
        if q is not None:
            self.set_q(q)
        else:
            self.mj.mj_forward(self.m, self.d)
        out = []
        allowed = set(int(g) for g in allowed_gids)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            a1, a2 = g1 in self.arm_gids, g2 in self.arm_gids
            if not a1 and not a2:           # 둘 다 환경
                continue
            if c.dist > 0.0:                # 마진 접촉(실제 침투 없음)
                continue
            if a1 and a2:                   # 팔-팔 자기 충돌 (비인접 링크; 인접 쌍은 MuJoCo 가 이미 걸러 준다)
                out.append((self.geom_name(g1), self.geom_name(g2), float(-c.dist)))
                continue
            arm_g, other = (g1, g2) if a1 else (g2, g1)
            if arm_g == self.cup_gid and other in allowed:
                continue
            out.append((self.geom_name(arm_g), self.geom_name(other), float(-c.dist)))
        return out

    def geom_name(self, gid: int) -> str:
        n = self.mj.mj_id2name(self.m, self.mj.mjtObj.mjOBJ_GEOM, gid)
        return n or f"geom{gid}"

    def path_collides(self, q_a, q_b, n=12, allowed_gids=(), max_step_rad=0.05) -> Optional[list]:
        """관절 공간 직선 보간 경로에서 첫 충돌. 없으면 None.

        샘플 수는 n 과 '관절 이동량 / max_step_rad' 중 큰 쪽 — 고정 10점이면 긴 이동에서 TCP 가 샘플 사이 0.5 m 를 건너뛰어
        이웃 박스를 스치는 것을 놓쳤다(검토에서 발견, 167포즈 중 2건)."""
        dq = float(np.max(np.abs(np.asarray(q_b, float) - np.asarray(q_a, float)))) if len(q_a) else 0.0
        n = max(int(n), int(math.ceil(dq / max_step_rad)))
        for t in np.linspace(0.0, 1.0, n + 1):
            c = self.contacts(np.asarray(q_a) * (1 - t) + np.asarray(q_b) * t, allowed_gids)
            if c:
                return c
        return None

    # ---- 시간 ------------------------------------------------------------
    def segment_time(self, q_a, q_b) -> float:
        """사다리꼴 속도 프로파일로 구간 시간: 가장 느린 관절이 결정한다."""
        dq = np.abs(np.asarray(q_b, float) - np.asarray(q_a, float))
        t = 0.0
        for d, v in zip(dq, self.vmax):
            if d < 1e-9:
                continue
            a = v / ACCEL_TIME_S
            if d < v * v / a:               # 최대 속도에 못 미치는 짧은 구간 (삼각 프로파일)
                t = max(t, 2.0 * math.sqrt(d / a))
            else:
                t = max(t, d / v + v / a)
        return t


def box_body_geom_ids(model, name_prefix="box"):
    import mujoco
    out = {}
    for g in range(model.ngeom):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if n and n.startswith(name_prefix) and n.endswith("_g"):
            out[n[len(name_prefix):-2]] = g
    return out
