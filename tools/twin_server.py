# -*- coding: utf-8 -*-
"""트윈 서버 — MuJoCo 셀 트윈을 TCP 로 열어 ROS2 노드(WSL2)가 '카메라' 와 '로봇' 으로 쓰게 한다.

    .venv\\Scripts\\python.exe tools/twin_server.py --arm track --boxes 12 --seed 500      # Windows 쪽에서 띄운다
    (WSL2)  ros2 launch twin_bridge twin_cycle.launch.py                                  # 인식 노드 + pick_executor + 브리지

역할 둘.
  카메라: `frame` 요청마다 트윈을 ToF 카메라로 렌더해 X/Y/D/I 프레임을 준다(실측 노이즈 모델 포함). 인식 노드의 소스가 이것을 받는다.
  로봇  : `execute` 요청(pick_executor 가 낸 pre-pick/pick/lift 3점 + 툴 자세, base_link)을 UR10e 가 IK·관절 보간으로 수행하고
          흡착 → 목적지 슬롯에 놓기 → 후퇴까지 한 뒤 결과(placed / grasp_miss / ik_unreachable / …)와 TCP 경로를 돌려준다.
          실제 셀에서 로봇 드라이버(MoveIt 액션 서버)가 하는 일을 시뮬레이터가 대신한다.

프로토콜은 robotsim_perception/twin_link.py. 좌표: 인식 노드의 base_link(탑다운 T_base_cam)와 트윈 월드가 같은 프레임이라는 것은
execute 마다 `pick_err_mm`(명령 픽 위치 vs 실제로 집은 박스 상면 중심)로 검증된다 — 외참이 어긋나면 이 값이 그만큼 커진다.
기동 시의 gt_selfcheck_mm 은 서버 안의 정답 좌표 변환 자기 일관성일 뿐이다. 실행 로직은 tools/twin_closed_loop.run_episode 의 픽 1회분과 같다 — 다른 것은
'무엇을 집을지' 를 이 프로세스가 아니라 ROS 그래프(인식 노드 → pick_executor)가 정한다는 점이다.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from cell_scene import BOX, CAM_H, DECK_H, N_COL, N_ROW, dest_slots, dest_world_xy, grid_xy  # noqa: E402
from cell_twin import ground_truth  # noqa: E402
from robotsim_perception.pose import apply, topdown_camera_transform  # noqa: E402
from robotsim_perception.twin_link import DEFAULT_PORT, frame_to_blob, serve  # noqa: E402
from twin_closed_loop import CTRL_HZ, SOURCE_ROI_MM, ArmCell, Cell, arm_cfg_from_results, shared_layout_cfg  # noqa: E402

LOG_DIR = ROOT / "explore" / "twin"


def quat_to_matrix(q):
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-6:
        raise ValueError("zero quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class StepRecorder:
    """mujoco 모듈을 감싸 mj_step 마다 TCP 위치를 기록한다 (rviz 에 경로를 그리기 위해). 나머지 속성은 그대로 넘긴다."""

    def __init__(self, mj, cell, every: int = 25):
        self._mj, self.cell, self.every = mj, cell, int(every)
        self.k, self.on, self.path = 0, False, []

    def __getattr__(self, name):
        return getattr(self._mj, name)

    def mj_step(self, m, d):
        self._mj.mj_step(m, d)
        self.k += 1
        if self.on and self.k % self.every == 0:
            p = self.cell.ee_pos()
            self.path.append([round(self.k / CTRL_HZ, 3), round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)])

    def start(self):
        self.k, self.on, self.path = 0, True, []

    def stop(self):
        self.on = False
        return self.path


class TwinServer:
    def __init__(self, arm: str, n_boxes: int, seed: int, noise: str, log_path: Path):
        self.arm_mode, self.noise = arm, noise
        self.log_path = log_path
        self.T = topdown_camera_transform(CAM_H * 1000.0)
        self.busy = False
        self.build(seed, n_boxes)

    # ------------------------------------------------------------------ 장면
    def build(self, seed: int, n_boxes: int):
        self.seed, self.n_boxes = int(seed), int(n_boxes)
        rng = np.random.default_rng(self.seed)
        cells = [(c, r) for r in range(N_ROW) for c in range(N_COL)]
        idx = sorted(rng.permutation(N_COL * N_ROW)[:self.n_boxes])
        layout = [(cells[i][0], cells[i][1], 0) for i in idx]
        self.arm_cfg = None if self.arm_mode == "none" else arm_cfg_from_results(track=(self.arm_mode == "track"))
        layout_cfg = shared_layout_cfg()
        if getattr(self, "cell", None) is not None:
            self.cell.close()
        self.cell = (ArmCell(layout, self.seed, self.arm_cfg, layout_cfg=layout_cfg) if self.arm_cfg
                     else Cell(layout, self.seed, arm_layout=True, layout_cfg=layout_cfg))
        self.rec = StepRecorder(self.cell.mj, self.cell)
        self.cell.mj = self.rec
        self.rng = np.random.default_rng(10_000 + self.seed)
        self.filled, self.placed, self.seq, self.n_exec = set(), 0, 0, 0
        self.check_frames()
        self.log({"event": "start", "arm": self.arm_mode, "seed": self.seed, "n_boxes": self.n_boxes, "noise": self.noise,
                  "arm_cfg": self.info()["arm_cfg"]})

    def check_frames(self):
        """정답(ground_truth) 카메라 좌표 -> 탑다운 T_base_cam -> 월드 가 박스 실제 위치와 맞는지 — 서버 안의 자기 일관성 검사.
        인식 노드의 실제 외참은 여기 오지 않으므로 이 값으로 노드-트윈 프레임 일치를 주장할 수 없다(리뷰 지적). 그 검증은 execute 의
        pick_err_mm(명령 픽 위치 vs 집은 박스 상면 중심)가 한다."""
        errs = []
        for g in ground_truth(self.cell.m, self.cell.d, top_only=False):
            c_cam = np.array([g["center_mm"][0], g["center_mm"][1], g["top_d_mm"]])
            p_base = apply(self.T, c_cam) / 1000.0
            bid = self.cell.mj.mj_name2id(self.cell.m, self.cell.mj.mjtObj.mjOBJ_BODY, g["name"])
            R = self.cell.d.xmat[bid].reshape(3, 3)
            top_w = self.cell.d.xpos[bid] + R @ np.array([0, 0, BOX[2] / 2])
            errs.append(float(np.linalg.norm(p_base - top_w)) * 1000.0)
        self.gt_selfcheck_mm = round(max(errs), 2) if errs else None
        if errs and max(errs) > 5.0:
            print(f"WARNING: ground-truth transform self-check {max(errs):.1f} mm", flush=True)

    def remaining(self) -> int:
        """소스 팔레트 위에 남은 박스 수 (정답). 실제 셀의 '팔레트 비움' PLC 신호에 해당."""
        n = 0
        for i in self.cell.box_ids:
            p = self.cell.d.xpos[self.cell.box_bid(i)]
            if abs(p[0]) < SOURCE_ROI_MM / 1000.0 and abs(p[1]) < SOURCE_ROI_MM / 1000.0 and p[2] > DECK_H - 0.05:
                n += 1
        return n

    def info(self) -> dict:
        return {"ok": True, "arm": self.arm_mode, "arm_cfg": ({k: (list(v) if isinstance(v, tuple) else v) for k, v in self.arm_cfg.items()}
                                                              if self.arm_cfg else None),
                "n_boxes": self.n_boxes, "seed": self.seed, "cam_height_mm": CAM_H * 1000.0,
                "shape": [480, 640], "seq": self.seq, "remaining": self.remaining(), "placed": self.placed,
                "gt_selfcheck_mm": self.gt_selfcheck_mm, "noise": self.noise}

    def state(self) -> dict:
        p = self.cell.ee_pos()
        st = {"ok": True, "remaining": self.remaining(), "placed": self.placed, "executed": self.n_exec,
              "tcp": [round(float(v), 4) for v in p], "busy": self.busy, "seq": self.seq, "sim_time_s": round(self.cell.steps / CTRL_HZ, 2)}
        if self.arm_cfg:
            st["q"] = [round(float(v), 4) for v in self.cell.arm.get_q()]
        return st

    # ------------------------------------------------------------------ 카메라
    def frame(self):
        self.seq += 1
        t0 = time.perf_counter()
        f = self.cell.frame(self.rng) if self.noise == "tof" else self.cell.rend.frame(self.cell.d)
        blob = frame_to_blob(f)
        meta = {"ok": True, "name": f"twin#{self.seq}", "seq": self.seq, "remaining": self.remaining(),
                "placed": self.placed, "render_ms": round((time.perf_counter() - t0) * 1e3, 1), "bytes": len(blob)}
        self.log({"event": "frame", **{k: v for k, v in meta.items() if k != "ok"}})
        return meta, blob

    # ------------------------------------------------------------------ 로봇
    def execute(self, pre_pick, pick, lift, quat) -> dict:
        cell, arm = self.cell, bool(self.arm_cfg)
        R = quat_to_matrix(quat)
        approach = R[:, 2] / max(np.linalg.norm(R[:, 2]), 1e-9)
        yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        pre, pk, lf = (np.asarray(v, float) for v in (pre_pick, pick, lift))
        self.busy, self.n_exec = True, self.n_exec + 1
        step0, wall0 = cell.steps, time.perf_counter()
        self.rec.start()

        def done(ok, result, **kw):
            self.busy = False
            out = {"ok": bool(ok), "result": result, "cycle_s": round((cell.steps - step0) / CTRL_HZ, 2),
                   "wall_s": round(time.perf_counter() - wall0, 1), "tcp_path": self.rec.stop(),
                   "remaining": self.remaining(), "placed": self.placed, "seq": self.n_exec, **kw}
            self.log({"event": "execute", **{k: v for k, v in out.items() if k != "tcp_path"},
                      "pick": [round(float(v), 4) for v in pk], "yaw_deg": round(yaw, 1)})
            return out

        try:
            ok, why, _ = cell.move_to(pre, yaw_deg=yaw)                       # 접근 (경로 충돌 검사 포함)
            if not ok:
                return done(False, why)
            # 컵 면을 상면 16 mm 위에 둔다 (run_episode 와 같음): 접촉 임펄스 없이 흡착 판정 45 mm 안에 든다
            # 박스 근처(하강·상승·후퇴)는 TCP 직선 이동(ArmCell.move_linear), 멀리 가는 이송은 관절 보간 + 경로 충돌 검사
            ok, why, _ = cell.move_linear(pk - approach * 0.016, speed_mps=0.25, yaw_deg=yaw, allow_box_contact=True)
            if not ok:
                cell.move_linear(lf, yaw_deg=yaw, allow_box_contact=True)
                return done(False, f"descent_{why}")
            held, gap = cell.grasp()
            if held is None:
                cell.move_linear(lf, yaw_deg=yaw, allow_box_contact=True)
                return done(False, "grasp_miss", gap_mm=round(gap, 1))
            # 명령 픽 위치 vs 실제 집은 박스 상면 중심: 인식 오차 + 노드-트윈 좌표 불일치가 여기 다 나타난다
            bid_h = cell.box_bid(held)
            top_h = cell.d.xpos[bid_h] + cell.d.xmat[bid_h].reshape(3, 3) @ np.array([0, 0, BOX[2] / 2])
            pick_err_mm = round(float(np.linalg.norm(top_h - pk)) * 1000.0, 1)
            cell.move_linear(lf, yaw_deg=yaw, allow_box_contact=True)         # 들어올림 (pick_executor 의 lift 점)
            before = cell.d.xpos[cell.box_bid(held)].copy()

            dest_w = np.array(dest_world_xy())
            slots = dest_slots()                 # 3열 x 3행 (cell_scene.DEST_COLS)
            layer = len(self.filled) // len(slots)
            free_slots = [s for s in range(len(slots)) if (layer, s) not in self.filled]
            if arm:
                bx, by = self.arm_cfg["base_xy"]
                free_slots.sort(key=lambda s: np.hypot(dest_w[0] + grid_xy(*slots[s])[0] - bx,
                                                       dest_w[1] + grid_xy(*slots[s])[1] - by))
            stack_h = DECK_H + BOX[2] * layer + BOX[2]
            ok, why, slot, drop = False, "place_ik_unreachable", None, None
            for s in free_slots:
                gx, gy = grid_xy(*slots[s])
                drop = np.array([dest_w[0] + gx, dest_w[1] + gy, 0.0])
                ok, why, _ = cell.move_to(np.array([drop[0], drop[1], stack_h + 0.45]), yaw_deg=yaw)   # 집은 자세 그대로 이송
                if not ok:
                    continue
                # 직선 하강(들고 있는 박스를 포함한 접촉 검사)도 성공해야 그 슬롯에 놓는다
                ok, why, _ = cell.move_linear(np.array([drop[0], drop[1], stack_h + 0.03]), speed_mps=0.25, yaw_deg=yaw,
                                              allow_box_contact=True)
                if ok:
                    slot = s
                    break
                why = f"descent_{why}"
                cell.move_linear(np.array([drop[0], drop[1], stack_h + 0.45]), yaw_deg=yaw, allow_box_contact=True)
            if not ok:
                # 목적지에 못 닿는다: 박스를 제자리에 돌려놓는다
                cell.move_to(pk + np.array([0, 0, 0.45]), yaw_deg=yaw, check_path=False)
                cell.move_linear(pk + np.array([0, 0, 0.02]), speed_mps=0.25, yaw_deg=yaw, allow_box_contact=True)
                cell.release()
                cell.move_linear(pk + np.array([0, 0, 0.45]), yaw_deg=yaw, allow_box_contact=True)
                return done(False, f"place_{why}", held=held, pick_err_mm=pick_err_mm)
            cell.release()
            cell.move_linear(np.array([drop[0], drop[1], stack_h + 0.55]), yaw_deg=yaw, allow_box_contact=True)
            after = cell.d.xpos[cell.box_bid(held)]
            err_mm = float(np.linalg.norm(after[:2] - drop[:2])) * 1000.0
            moved = err_mm < 160.0 and after[2] > DECK_H - 0.05
            self.placed += int(moved)
            self.filled.add((layer, slot))
            return done(moved, "placed" if moved else "misplaced", held=held, gap_mm=round(gap, 1), pick_err_mm=pick_err_mm,
                        place_err_mm=round(err_mm, 1), landed_z=round(float(after[2]), 3), slot=[layer, slot],
                        travel_mm=round(float(np.linalg.norm(after - before)) * 1000.0, 1))
        except Exception as e:  # noqa: BLE001
            self._recover(cell)          # 든 박스가 있으면 놓고(충돌 복원) 검사 훅을 끈다 — 안 하면 다음 픽까지 박스가 TCP 에 붙어 다닌다
            return done(False, f"error:{type(e).__name__}", error=repr(e))

    @staticmethod
    def _recover(cell):
        try:
            if cell.held is not None:
                cell.release()
        except Exception:  # noqa: BLE001
            cell.held = None
            gid = getattr(cell, "_held_gid", None)
            if gid is not None:
                cell.m.geom_contype[gid], cell.m.geom_conaffinity[gid] = cell._saved_con
                cell._held_gid = None
        if hasattr(cell, "arm"):
            cell.arm.payload_sync = None

    # ------------------------------------------------------------------ 요청 분배
    def handle(self, obj: dict):
        cmd = obj.get("cmd")
        if cmd == "hello":
            return self.info(), None
        if cmd == "frame":
            return self.frame()
        if cmd == "state":
            return self.state(), None
        if cmd == "reset":
            self.build(obj.get("seed", self.seed), obj.get("n_boxes", self.n_boxes))
            self.log({"event": "reset", "seed": self.seed, "n_boxes": self.n_boxes})
            return self.info(), None
        if cmd == "execute":
            for k in ("pre_pick", "pick", "lift", "quat"):
                if k not in obj or len(obj[k]) not in (3, 4):
                    return {"ok": False, "error": f"missing/invalid field {k}"}, None
            res = self.execute(obj["pre_pick"], obj["pick"], obj["lift"], obj["quat"])
            print(f"  execute #{res['seq']}: {res['result']}  sim {res['cycle_s']} s  wall {res['wall_s']} s  "
                  f"pick_err {res.get('pick_err_mm', '-')} mm  remaining {res['remaining']}  placed {res['placed']}", flush=True)
            return res, None
        return {"ok": False, "error": f"unknown cmd {cmd!r}"}, None

    def log(self, rec: dict):
        rec = dict(rec, t=round(time.time(), 3))
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", choices=["none", "fixed", "track"], default="track")
    ap.add_argument("--boxes", type=int, default=12, help="소스 상층 박스 수 (최대 12)")
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--noise", choices=["tof", "none"], default="tof")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--log", default=str(LOG_DIR / "twin_server.jsonl"))
    ap.add_argument("--pidfile", default="", help="기동 시 PID 를 적는 파일 (스크립트가 종료시킬 때 씀)")
    ap.add_argument("--exit-after-s", type=float, default=0.0, help="0 이 아니면 이 시간 뒤 스스로 종료 (테스트용)")
    args = ap.parse_args()
    if args.pidfile:
        import os
        Path(args.pidfile).write_text(str(os.getpid()), encoding="utf-8")
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # 같은 포트에 이미 서버가 있으면 바로 종료 (Windows 에서는 bind 가 통과할 수 있어 소켓으로 직접 확인한다)
    import socket
    try:
        with socket.create_connection(("127.0.0.1", args.port), timeout=1.0):
            pass
        print(f"ERROR: a twin server already answers on port {args.port} — stop it first (pidfile/taskkill)", flush=True)
        sys.exit(2)
    except OSError:
        pass
    log_path.write_text("", encoding="utf-8")
    t0 = time.perf_counter()
    srv = TwinServer(args.arm, args.boxes, args.seed, args.noise, log_path)
    info = srv.info()
    print(f"twin ready in {time.perf_counter() - t0:.1f} s: arm={args.arm} boxes={info['remaining']} seed={args.seed} "
          f"noise={args.noise} gt_selfcheck={info['gt_selfcheck_mm']} mm  log={log_path}", flush=True)
    t_start = time.perf_counter()
    stop = (lambda: time.perf_counter() - t_start > args.exit_after_s) if args.exit_after_s > 0 else None
    serve(srv.handle, host=args.host, port=args.port, log=lambda s: print(s, flush=True), stop=stop)


if __name__ == "__main__":
    main()
