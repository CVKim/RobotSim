# -*- coding: utf-8 -*-
"""인식 -> 계획 -> 실행 폐루프 (MuJoCo 셀 트윈).

지금까지 인식-정책 연동은 depth 오버레이에 PICK/PLACE 를 그리는 데서 끝났고,
실제로 무언가를 집어 옮긴 적이 없었다. 이 스크립트가 루프를 닫는다:

    렌더(ToF) -> detect_boxes_v2 -> 픽 순서 계획 -> 6-DoF 포즈(로봇 베이스 좌표)
             -> 석션 EE 이동/흡착/이송/해제(물리) -> 결과 판정 -> 다음 프레임 재촬영

핵심: 정책이 **인식 출력만** 보고 움직인다. 시뮬레이터의 정답 위치를 쓰지 않는다.
따라서 인식 오차(병합·누락·중심 오차)가 실제 픽 실패로 어떻게 이어지는지 측정된다.
정답 위치로 집는 'oracle' 모드를 함께 돌려 인식 때문에 잃는 성능을 분리한다.

한계: 팔 기구학·충돌회피가 없다. 석션 EE 를 mocap 으로 직접 구동하므로
도달범위·특이점·주변 충돌은 평가되지 않는다(docs/41 참조).

실행:  .venv\\Scripts\\python.exe tools/twin_closed_loop.py --episodes 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from binpick_topface import detect_boxes_v2  # noqa: E402
from cell_scene import (BOX, CAM_H, DECK_H, N_COL, N_ROW, build_xml,  # noqa: E402
                        dest_world_xy, grid_xy)
from cell_twin import TwinRenderer, ground_truth, settle  # noqa: E402
from robotsim_perception.pose import (box_to_pick_pose, suction_footprint_ok,  # noqa: E402
                                      topdown_camera_transform)

OUT = ROOT / "explore" / "twin"
CTRL_HZ = 500.0          # timestep 0.002
SUCTION_REACH_MM = 45.0  # 이 거리 안이면 흡착 성립
SOURCE_ROI_MM = 620.0    # 소스 팔레트 반경 (카메라 좌표 mm) — 목적지 스택 제외용
LATERAL_TOL_MM = 90.0    # 흡착 성립 측면 허용 오차 (컵 아래 면에만 붙음)


def cam_to_world(p_mm):
    """카메라 좌표 mm -> 월드 m (트윈 관례: ToF_X=world X, ToF_Y=-world Y, D=CAM_H-z)."""
    return np.array([p_mm[0] / 1000.0, -p_mm[1] / 1000.0, CAM_H - p_mm[2] / 1000.0])


class Cell:
    def __init__(self, layout, seed):
        import mujoco
        self.mj = mujoco
        xml, _ = build_xml(layout, seed=seed)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        settle(self.m, self.d, 1200)
        self.rend = TwinRenderer(self.m)
        self.mocap_id = self.m.body_mocapid[
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "suction_target")]
        self.suction_bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "suction")
        self.grip_eq = {}
        for k in range(self.m.neq):
            nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, k)
            if nm and nm.startswith("grip"):
                self.grip_eq[int(nm[4:])] = k
                self.d.eq_active[k] = 0          # weld 는 쓰지 않는다 (아래 주석 참조)
        self.held = None
        self.hold_offset = None

    def close(self):
        self.rend.close()

    def box_bid(self, i):
        return self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_BODY, f"box{i}")

    def move_to(self, xyz_w, speed_mps=0.6, settle_steps=150):
        """mocap 을 목표까지 S-커브로 이동(거리에 비례한 시간) 후 안정화.

        박스를 문 채로 급가속하면 흡착 weld 가 버티지 못해 박스가 뒤처진다.
        이동 시간을 거리/속도로 잡고 smoothstep 프로파일을 써서 가속도를 제한한다.
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
            self._carry()
        for _ in range(settle_steps):
            self.mj.mj_step(self.m, self.d)
            self._carry()
        return float(np.linalg.norm(self.d.xpos[self.suction_bid] - goal))

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
        ee = self.d.xpos[self.suction_bid]
        self.d.qpos[adr:adr + 3] = ee + self.hold_offset
        self.d.qpos[adr + 3:adr + 7] = self.hold_quat
        vadr = self.m.jnt_dofadr[jid]
        self.d.qvel[vadr:vadr + 6] = 0.0
        self.mj.mj_forward(self.m, self.d)

    def grasp(self):
        """EE 근처에서 가장 가까운 박스를 흡착. 성공 시 box index, 실패 시 None."""
        ee = self.d.xpos[self.suction_bid]
        best, bd = None, 1e9
        for i in self.grip_eq:
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
        bid = self.box_bid(best)
        self.hold_offset = (self.d.xpos[bid] - self.d.xpos[self.suction_bid]).copy()
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
        for _ in range(60):
            self.mj.mj_step(self.m, self.d)
            self._carry()
        return best, bd * 1000.0

    def release(self):
        """부착 해제 후 물리로 안착시킨다 (스택 위 낙하·정렬은 물리 계산)."""
        if getattr(self, "_held_gid", None) is not None:
            self.m.geom_contype[self._held_gid] = self._saved_con[0]
            self.m.geom_conaffinity[self._held_gid] = self._saved_con[1]
            self._held_gid = None
        self.held = None
        self.hold_offset = None
        self.mj.mj_forward(self.m, self.d)
        for _ in range(600):
            self.mj.mj_step(self.m, self.d)

    def frame(self, rng):
        return self.rend.frame(self.d, noise="tof", rng=rng)


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


def run_episode(layout, seed, oracle=False, conf_min=0.0, use_footprint=True, verbose=False):
    cell = Cell(layout, seed)
    rng = np.random.default_rng(10_000 + seed)
    T = topdown_camera_transform(cam_height_mm=CAM_H * 1000.0)   # 트윈은 외참을 정확히 안다
    dest_w = np.array(dest_world_xy())
    dest_cam_xy = (dest_w[0] * 1000.0, -dest_w[1] * 1000.0)
    n_start = len(layout)
    log = {"picks": [], "seed": seed, "oracle": oracle, "n_boxes": n_start}
    placed = 0
    stack_h = DECK_H
    try:
        for step in range(n_start):
            f = cell.frame(rng)
            if oracle:
                gt = ground_truth(cell.m, cell.d)
                boxes = [{"center_mm": (g["center_mm"][0], g["center_mm"][1], g["top_d_mm"]),
                          "depth_mm": g["top_d_mm"], "dims_mm": g["dims_mm"],
                          "ang_deg": g["ang_deg"], "normal": (0, 0, -1), "confidence": 1.0,
                          "rect_px": None} for g in gt]
            else:
                _, _, det = detect_boxes_v2(f)
                boxes = [b for b in det if b.get("confidence", 0) >= conf_min]
                for b in boxes:
                    b["center_mm"] = (b["center_mm"][0], b["center_mm"][1], b["depth_mm"])
                    b["normal"] = (0.0, 0.0, -1.0)
            # 소스 팔레트 ROI 로 제한. 목적지 스택은 소스와 같은 높이가 되므로 같은 층으로 검출되고,
            # ROI 가 없으면 플래너가 방금 옮긴 박스를 다시 집는다(1차 시도에서 발생: 같은 박스 4회 픽).
            # 실제 셀도 인식 ROI 를 소스 팔레트로 한정한다.
            boxes = [b for b in boxes
                     if abs(b["center_mm"][0]) < SOURCE_ROI_MM and abs(b["center_mm"][1]) < SOURCE_ROI_MM]
            if not boxes:
                log["picks"].append({"step": step, "result": "source_empty"})
                break
            order = pick_order(boxes, dest_cam_xy)
            b, n_rej = None, 0
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
                b = cand
                break
            if b is None:
                log["picks"].append({"step": step, "result": "rejected_footprint",
                                     "candidates_rejected": n_rej})
                continue
            pose = box_to_pick_pose(b, T, clearance_mm=180.0)
            tgt = cam_to_world(np.array(b["center_mm"], float))
            cell.move_to(tgt + np.array([0, 0, 0.20]))
            cell.move_to(tgt + np.array([0, 0, 0.016]), speed_mps=0.25)
            held, gap = cell.grasp()
            if held is None:
                log["picks"].append({"step": step, "result": "grasp_miss",
                                     "gap_mm": round(gap, 1)})
                cell.move_to(tgt + np.array([0, 0, 0.35]))
                if verbose:
                    print(f"    step {step}: 흡착 실패 (gap {gap:.0f}mm)")
                continue
            before = cell.d.xpos[cell.box_bid(held)].copy()
            cell.move_to(tgt + np.array([0, 0, 0.45]))
            # 목적지 배치: 실제 이적재 공정처럼 4x3 격자 슬롯에 층층이 쌓는다.
            # (모든 박스를 목적지 중심 한 점에 떨어뜨리면 서로 부딪혀 무너진다 — 1차 시도에서 발생)
            slot = placed % (N_COL * N_ROW)
            layer = placed // (N_COL * N_ROW)
            gx, gy = grid_xy(slot % N_COL, slot // N_COL)
            drop = np.array([dest_w[0] + gx, dest_w[1] + gy, 0.0])
            stack_h = DECK_H + BOX[2] * layer + BOX[2]
            cell.move_to(np.array([drop[0], drop[1], stack_h + 0.45]))
            cell.move_to(np.array([drop[0], drop[1], stack_h + 0.03]), speed_mps=0.25)
            cell.release()
            cell.move_to(np.array([drop[0], drop[1], stack_h + 0.55]))
            after = cell.d.xpos[cell.box_bid(held)]
            moved = (float(np.linalg.norm(after[:2] - drop[:2])) < 0.16
                     and after[2] > DECK_H - 0.05)
            placed += int(moved)
            log["picks"].append({"step": step, "result": "placed" if moved else "misplaced",
                                 "box": held, "gap_mm": round(gap, 1),
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--boxes", type=int, default=6, help="에피소드당 상층 박스 수")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from cell_scene import N_COL, N_ROW
    cells = [(c, r) for r in range(N_ROW) for c in range(N_COL)]
    results = {}
    for mode in ("perception", "oracle"):
        eps = []
        print(f"--- {mode}")
        for e in range(args.episodes):
            rng = np.random.default_rng(500 + e)
            idx = sorted(rng.permutation(N_COL * N_ROW)[:args.boxes])
            layout = [(cells[i][0], cells[i][1], 0) for i in idx]
            log = run_episode(layout, seed=500 + e, oracle=(mode == "oracle"),
                              verbose=args.verbose)
            eps.append(log)
            print(f"  ep{e}: {log['placed']}/{log['n_boxes']} 배치 "
                  f"({', '.join(sorted({p['result'] for p in log['picks']}))})")
        tot = sum(e["n_boxes"] for e in eps)
        ok = sum(e["placed"] for e in eps)
        reasons = {}
        for e in eps:
            for p in e["picks"]:
                reasons[p["result"]] = reasons.get(p["result"], 0) + 1
        results[mode] = {"episodes": len(eps), "boxes": tot, "placed": ok,
                         "success_rate": round(ok / max(tot, 1), 3),
                         "outcomes": reasons, "logs": eps}
        print(f"  => {mode}: {ok}/{tot} = {results[mode]['success_rate']:.1%}  {reasons}")

    gap = results["oracle"]["success_rate"] - results["perception"]["success_rate"]
    results["perception_cost"] = round(gap, 3)
    print(f"\n인식 때문에 잃는 성공률: {gap:+.1%} "
          f"(oracle {results['oracle']['success_rate']:.1%} vs 인식 {results['perception']['success_rate']:.1%})")
    (OUT / "closed_loop.json").write_text(json.dumps(results, indent=1, ensure_ascii=False),
                                          encoding="utf-8")
    print("saved", OUT / "closed_loop.json")


if __name__ == "__main__":
    main()
