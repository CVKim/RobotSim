# -*- coding: utf-8 -*-
"""UR10e 팔 도달·충돌·사이클 시간 평가 (셀 트윈 + 실측 픽 포즈).

지금까지 셀 트윈의 석션 EE 는 mocap 으로 순간이동해 도달 범위·관절 한계·충돌·사이클 시간이 평가되지 않았다.
이 도구가 그 셋을 처음으로 잰다.

  A. 받침대 위치 sweep      팔 베이스 (x, y, 받침대 높이) 격자에서 소스 상면 24곳(2층 12 + 1층 12) + 목적지 24곳(2층째 12 + 1층째 12)의
                            pre-pick/pick 이 IK 로 풀리고 충돌이 없는 비율. 최적 위치 선정.
  B. 리니어 트랙            최적 위치에 x 방향 트랙(±0.6 m)을 더하면 어디까지 닿나 (팔 베이스 높이는 고정 구성과 맞춤).
  C. 실측 픽 포즈           실측 30프레임의 검출 박스(167개)를 로봇 좌표로 옮긴 픽 포즈에 대해
                            도달 가능 / 하강 경로 충돌 없음 / 접근 경로 충돌 없음 비율 (층별). 이웃 박스는 검출 위치에 세운다.
  D. 사이클 시간            관절 속도 한계(UR10e 사양)의 사다리꼴 프로파일로 pre-pick → pick → lift → place 시간.

실행 (Windows, .venv):  .venv\\Scripts\\python.exe tools/twin_arm_reach.py [--fast]
산출: explore/twin/arm_reach.json (로컬), results/twin_arm_reach.json (집계·익명), assets/twin_arm_reach.png
실측 데이터가 없으면(tools/local_paths.py 없음) C, D 는 건너뛴다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from arm import Arm, box_body_geom_ids, rot_from_approach_yaw  # noqa: E402
from cell_scene import BOX, CAM_H, DECK_H, N_COL, N_ROW, build_xml, dest_slots, dest_world_xy, full_layout, grid_xy  # noqa: E402
from cell_twin import settle  # noqa: E402

OUT_LOCAL = ROOT / "explore" / "twin" / "arm_reach.json"
OUT_PUB = ROOT / "results" / "twin_arm_reach.json"
OUT_PNG = ROOT / "assets" / "twin_arm_reach.png"
PRE_M = 0.15            # pre-pick 높이
UR_REACH = 1.30         # UR10e 사양 도달 반경 (m)
DWELL_S = 0.3           # 흡착 / 해제 대기


def _targets_for(scene: str):
    """scene 'top2': 소스 2층 상면 12 + 목적지(1층 위) 2층째 9 / 'top1': 소스 1층 상면 12 + 목적지(빈 데크) 1층째 9.
    두 장면을 합쳐 42목표: 소스 상면 24 (2층 12 + 1층 12) + 목적지 18 (2층째 9 + 1층째 9).
    목적지는 3열(cell_scene.DEST_COLS): 4열째 슬롯은 소스 박스와 1.6 mm 겹쳐 놓을 수 없는 자리라 목표에서도 뺀다.
    반환 [(tag, xyz_pick(m), yaw_deg)]"""
    dx, dy = dest_world_xy()
    out = []
    n_layer = 2 if scene == "top2" else 1
    z_src = DECK_H + 0.004 + n_layer * BOX[2] + 0.0015
    z_dst = DECK_H + n_layer * BOX[2] + 0.0015
    for r in range(N_ROW):
        for c in range(N_COL):
            x, y = grid_xy(c, r)
            out.append(("src", np.array([x, y, z_src]), 0.0))
    for c, r in dest_slots():
        x, y = grid_xy(c, r)
        out.append(("dst", np.array([dx + x, dy + y, z_dst + 0.01]), 0.0))
    return out


def _quick_unreachable(arm, m, target) -> bool:
    """숄더 원점에서 목표까지 거리가 도달 반경 + 여유보다 크면 IK 를 돌리지 않는다 (sweep 시간 절약)."""
    import mujoco
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "shoulder_link")
    if arm.has_track:
        return False
    sh = arm.d.xbody[sid] if hasattr(arm.d, "xbody") else arm.d.xpos[sid]
    return float(np.linalg.norm(np.asarray(target) - sh)) > UR_REACH + 0.25


def eval_config(cfg: dict, fast: bool = False) -> dict:
    """한 받침대 구성에서 두 장면(top2/top1)의 소스·목적지 목표 도달 여부를 센다."""
    import mujoco
    res = dict(cfg=cfg, n=0, ik_ok=0, free_ok=0, src=[0, 0], dst=[0, 0], fails={})
    Rt = rot_from_approach_yaw((0, 0, -1), 0.0)
    for scene in ("top2", "top1"):
        layout = full_layout(2 if scene == "top2" else 1)
        xml, _ = build_xml(layout, seed=7, dest_stack=(len(dest_slots()) if scene == "top2" else 0),
                           arm=dict(base_xy=tuple(cfg["base_xy"]), pedestal_h=cfg["pedestal_h"],
                                    track_range=cfg.get("track_range", 0.0), meshes=False))
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        arm = Arm(m, d)
        arm.set_q(arm.home)                 # 정착 전에 홈으로 (qpos=0 은 팔이 수평으로 뻗어 목적지 스택을 휩쓴다)
        arm.set_ctrl(arm.home)
        settle(m, d, 300)
        box_gids = box_body_geom_ids(m)
        box_xy = {g: d.geom_xpos[g][:2].copy() for g in box_gids.values()}
        q_prev = arm.home.copy()
        for tag, pick, yaw in _targets_for(scene):
            res["n"] += 1
            k = 0 if tag == "src" else 1
            (res["src"] if tag == "src" else res["dst"])[1] += 1
            pre = pick + np.array([0, 0, PRE_M])
            if _quick_unreachable(arm, m, pre):
                res["fails"]["out_of_reach"] = res["fails"].get("out_of_reach", 0) + 1
                continue
            # 컵 접촉을 허용하는 박스 = 목표 바로 아래 박스 하나 (없으면 없음)
            near = [g for g, xy in box_xy.items() if np.hypot(xy[0] - pick[0], xy[1] - pick[1]) < 0.12]
            boxes = near[:1]
            iters = 120 if fast else 200
            # 충돌 회피 IK: 팔꿈치 위/아래 등 여러 자세 중 환경에 닿지 않는 해를 고른다
            r1, c1 = arm.solve_ik_free(pre, Rt, q_prev, allowed_gids=boxes, seeds=3 if fast else 8, iters=iters)
            if not r1.ok:
                res["fails"]["ik"] = res["fails"].get("ik", 0) + 1
                continue
            r2, c2 = arm.solve_ik_free(pick, Rt, r1.q, allowed_gids=boxes, seeds=2 if fast else 6, iters=iters)
            if not r2.ok:
                res["fails"]["ik"] = res["fails"].get("ik", 0) + 1
                continue
            res["ik_ok"] += 1
            if c1 or c2:
                res["fails"]["collision"] = res["fails"].get("collision", 0) + 1
                continue
            res["free_ok"] += 1
            (res["src"] if tag == "src" else res["dst"])[0] += 1
            q_prev = r1.q
    res["frac_ok"] = round(res["free_ok"] / max(res["n"], 1), 3)
    res["src_frac"] = round(res["src"][0] / max(res["src"][1], 1), 3)
    res["dst_frac"] = round(res["dst"][0] / max(res["dst"][1], 1), 3)
    return res


# ------------------------------------------------------------------ 실측 픽 포즈

def _real_sessions():
    try:
        from local_paths import BINPICK_DIR
    except ImportError:
        return []
    root = Path(BINPICK_DIR)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "4_tof_D.mim").exists())


def eval_real(cfg: dict, fast: bool = False, max_frames: int | None = None) -> dict:
    """실측 프레임마다: 검출 박스를 그 자리에 세운 팔 씬을 만들고, 각 박스의 픽 포즈에 대해
    IK(pre/pick) → 하강 경로 충돌 → 접근 경로(홈→pre) 충돌 → 사이클 시간(목적지 슬롯 순환) 을 잰다."""
    import mujoco
    from robotsim_perception import load_frame
    from robotsim_perception.pose import box_to_pick_pose, topdown_camera_transform
    from robotsim_perception.runtime import Thresholds, decide
    sessions = _real_sessions()
    if max_frames:
        sessions = sessions[:max_frames]
    T = topdown_camera_transform(cam_height_mm=CAM_H * 1000.0)
    Rt0 = rot_from_approach_yaw((0, 0, -1), 0.0)
    dx, dy = dest_world_xy()
    rows = []
    slot = 0
    for si, sp in enumerate(sessions):
        fr = load_frame(sp)
        dec = decide(fr, Thresholds(lattice=True, source_roi_mm=620.0))
        if dec.status != "OK" or not dec.boxes:
            continue
        pps = [(b, box_to_pick_pose(b, T, clearance_mm=PRE_M * 1000.0)) for b in dec.boxes]
        statics = []
        for b, pp in pps:
            x, y, z = (v / 1000.0 for v in pp.position_mm)
            statics.append(dict(x=x, y=y, z_top=z, yaw_rad=np.radians(pp.yaw_deg), L=b.dims_mm[0] / 1000.0,
                                W=b.dims_mm[1] / 1000.0))
        xml, _ = build_xml([], seed=100 + si, tier_sheet=False, distractors=True, static_boxes=statics,
                           arm=dict(base_xy=tuple(cfg["base_xy"]), pedestal_h=cfg["pedestal_h"],
                                    track_range=cfg.get("track_range", 0.0), meshes=False))
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        arm = Arm(m, d)
        arm.set_q(arm.home)
        sbox = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"sbox{k}_g") for k in range(len(statics))}
        layer = "2층" if statics and statics[0]["z_top"] > 1.05 else "1층"
        q_ret = arm.home.copy()
        for k, (b, pp) in enumerate(pps):
            pick = np.array(pp.position_mm) / 1000.0
            pre = pick + np.array([0, 0, PRE_M])
            Rt = rot_from_approach_yaw((0, 0, -1), pp.yaw_deg)
            row = dict(frame=si, layer=layer, conf=round(b.confidence, 2), src=getattr(b, "source", "detected"),
                       reach=False, descent_free=None, approach_free=None, cycle_s=None,
                       fail="")
            iters = 150 if fast else 350
            allowed = [sbox[k]]
            # 접근: 직전 후퇴 자세에서 pre-pick 까지 관절 보간 경로가 환경(이웃 박스·설비)에 닿지 않는 자세를 고른다
            r1, c_app = arm.solve_ik_free(pre, Rt, q_ret, allowed_gids=allowed, path_from=q_ret,
                                          seeds=4 if fast else 10, iters=iters)
            if not r1.ok:
                row["fail"] = "ik_pre"
                rows.append(row)
                continue
            r2, c_pick = arm.solve_ik_free(pick, Rt, r1.q, allowed_gids=allowed, seeds=3 if fast else 10, iters=iters)
            if not r2.ok:
                row["fail"] = "ik_pick"
                rows.append(row)
                continue
            row["reach"] = True
            row["approach_free"] = not c_app
            # 하강 경로: pre -> pick 직선 (IK 워밍). 컵-목표 박스 접촉만 허용
            desc_free, q = not c_pick, r1.q
            if desc_free:
                for t in np.linspace(0.0, 1.0, 6)[1:]:
                    r = arm.solve_ik(pre * (1 - t) + pick * t, Rt, q, iters=80)
                    q = r.q
                    if arm.contacts(q, allowed_gids=allowed):
                        desc_free = False
                        break
            row["descent_free"] = desc_free
            if not desc_free:
                row["fail"] = "descent_collision"
            elif not row["approach_free"]:
                row["fail"] = "approach_collision"
            # 사이클 시간: 목적지 슬롯 순환 (1층째). 슬롯은 포즈마다 넘어간다 (안 닿는 슬롯에 갇히지 않게)
            slots = dest_slots()
            gx, gy = grid_xy(*slots[slot % len(slots)])
            slot += 1
            place = np.array([dx + gx, dy + gy, DECK_H + BOX[2] + 0.01])
            pre_place = place + np.array([0, 0, PRE_M])
            rp1, _ = arm.solve_ik_free(pre_place, Rt0, r1.q, allowed_gids=[], seeds=3, iters=iters)
            rp2, _ = arm.solve_ik_free(place, Rt0, rp1.q, allowed_gids=[], seeds=2, iters=iters) if rp1.ok else (None, None)
            row["place_reach"] = bool(rp1.ok and rp2 is not None and rp2.ok)
            if row["place_reach"]:
                motion = (arm.segment_time(q_ret, r1.q) + arm.segment_time(r1.q, r2.q)
                          + arm.segment_time(r2.q, r1.q) + arm.segment_time(r1.q, rp1.q)
                          + arm.segment_time(rp1.q, rp2.q) + arm.segment_time(rp2.q, rp1.q))
                row["motion_s"] = round(float(motion), 2)              # 순수 이동
                row["cycle_s"] = round(float(motion + 2 * DWELL_S), 2)   # + 흡착/해제 대기 0.3 s x 2
                q_ret = rp1.q
            rows.append(row)
    return dict(n_frames=len(sessions), rows=rows)


def summarize_real(real: dict) -> dict:
    rows = real["rows"]
    if not rows:
        return {}

    def frac(sel, key):
        v = [r[key] for r in sel if r.get(key) is not None]
        return round(sum(1 for x in v if x) / len(v), 3) if v else None

    reached = [r for r in rows if r["reach"]]
    out = dict(n_poses=len(rows), reachable=frac(rows, "reach"),
               descent_free=frac(reached, "descent_free"),
               approach_free=frac(reached, "approach_free"),
               # 목적지 도달: 픽에 닿은 포즈 중 목적지 슬롯(pre-place·place)까지 IK 가 풀린 비율
               place_reachable=frac(reached, "place_reach"),
               # 픽과 목적지 둘 다 닿아 사이클 시간이 계산된 비율 (전체 포즈 기준)
               pick_and_place=round(sum(1 for r in rows if r.get("cycle_s") is not None) / len(rows), 3),
               fails={}, by_layer={})
    for r in rows:
        if r["fail"]:
            out["fails"][r["fail"]] = out["fails"].get(r["fail"], 0) + 1
    for layer in sorted({r["layer"] for r in rows}):
        sel = [r for r in rows if r["layer"] == layer]
        out["by_layer"][layer] = dict(n=len(sel), reachable=frac(sel, "reach"),
                                      descent_free=frac([r for r in sel if r["reach"]], "descent_free"),
                                      approach_free=frac([r for r in sel if r["reach"]], "approach_free"))
    cyc = [r["cycle_s"] for r in rows if r.get("cycle_s") is not None]
    mot = [r["motion_s"] for r in rows if r.get("motion_s") is not None]
    if cyc:
        out["cycle_s"] = dict(n=len(cyc), mean=round(float(np.mean(cyc)), 2), p50=round(float(np.median(cyc)), 2),
                              p95=round(float(np.percentile(cyc, 95)), 2), min=round(float(min(cyc)), 2),
                              max=round(float(max(cyc)), 2), dwell_included_s=2 * DWELL_S,
                              motion_only_mean=round(float(np.mean(mot)), 2), mocap_reference_s=11.7)
    return out


# ------------------------------------------------------------------ 차트

def chart(sweep, best, track, real_sum, path):
    n_targets = len(_targets_for('top2')) + len(_targets_for('top1'))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    for cand in (r"C:\Windows\Fonts\malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        if Path(cand).exists():
            fm.fontManager.addfont(cand)
            plt.rcParams["font.family"] = fm.FontProperties(fname=cand).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # (a) 받침대 위치 히트맵 (최적 높이에서)
    h = best["cfg"]["pedestal_h"]
    xs = sorted({c["cfg"]["base_xy"][0] for c in sweep})
    ys = sorted({c["cfg"]["base_xy"][1] for c in sweep})
    grid = np.full((len(ys), len(xs)), np.nan)
    for c in sweep:
        if abs(c["cfg"]["pedestal_h"] - h) < 1e-6:
            grid[ys.index(c["cfg"]["base_xy"][1]), xs.index(c["cfg"]["base_xy"][0])] = c["frac_ok"] * 100
    ax = axes[0]
    im = ax.imshow(grid, origin="lower", cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"{x:.1f}" for x in xs])
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([f"{y:.2f}" for y in ys])
    for i in range(len(ys)):
        for j in range(len(xs)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", color="w" if grid[i, j] < 60 else "k", fontsize=9)
    ax.set_xlabel("팔 베이스 x (m)")
    ax.set_ylabel("팔 베이스 y (m)")
    ax.set_title(f"UR10e 고정 베이스: {n_targets}목표 중 도달·무충돌 비율 %  (받침대 {h:.1f} m)", fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.046)
    # (b) 최적 / 트랙 / 실측
    ax = axes[1]
    labels = ["고정 베이스\n소스", "고정 베이스\n목적지", "트랙 ±0.6 m\n소스", "트랙 ±0.6 m\n목적지"]
    vals = [best["src_frac"] * 100, best["dst_frac"] * 100, track["src_frac"] * 100, track["dst_frac"] * 100]
    bars = ax.bar(labels, vals, color=["#4c72b0", "#4c72b0", "#dd8452", "#dd8452"])
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("도달·무충돌 %")
    ax.set_title(f"트윈 목표 {n_targets} (소스 상면 2층 12·1층 12 + 목적지 3열: 2층째 9·1층째 9)", fontsize=9.5)
    # (c) 실측 픽 포즈
    ax = axes[2]
    if real_sum:
        keys = ["reachable", "descent_free", "approach_free", "place_reachable", "pick_and_place"]
        names = ["픽 도달", "하강\n무충돌", "접근\n무충돌", "목적지 도달\n(픽 도달 중)", "픽+목적지\n둘 다"]
        v = [(real_sum.get(k) or 0) * 100 for k in keys]
        bars = ax.bar(names, v, color="#55a868")
        for b_, x in zip(bars, v):
            ax.text(b_.get_x() + b_.get_width() / 2, x + 1, f"{x:.0f}%", ha="center", fontsize=9)
        ax.set_ylim(0, 110)
        c = real_sum.get("cycle_s")
        sub = f"이동 {c['motion_only_mean']:.1f} s + 대기 0.6 s = {c['mean']:.1f} s (p95 {c['p95']:.1f}); mocap 11.7 s" if c else ""
        ax.set_title(f"실측 픽 포즈 {real_sum['n_poses']}개 (트랙 포함 최적 구성)\n{sub}", fontsize=9.5)
    else:
        ax.text(0.5, 0.5, "실측 데이터 없음", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")      # Windows cp949 콘솔에서 한글·기호 출력
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="적은 IK 반복·시드 (개발용)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--chart-only", action="store_true", help="results/twin_arm_reach.json 으로 차트만 다시 그림")
    args = ap.parse_args()
    if args.chart_only:
        pub = json.loads(OUT_PUB.read_text(encoding="utf-8"))
        sweep = [dict(cfg=dict(base_xy=tuple(r["base_xy"]), pedestal_h=r["pedestal_h"]), frac_ok=r["frac_ok"],
                      src_frac=r["src_frac"], dst_frac=r["dst_frac"]) for r in pub["config_sweep"]]
        best = dict(cfg=dict(base_xy=tuple(pub["best_fixed"]["cfg"]["base_xy"]), pedestal_h=pub["best_fixed"]["cfg"]["pedestal_h"]),
                    frac_ok=pub["best_fixed"]["frac_ok"], src_frac=pub["best_fixed"]["src_frac"], dst_frac=pub["best_fixed"]["dst_frac"])
        chart(sweep, best, pub["track"], pub.get("real_poses") or {}, OUT_PNG)
        print("saved", OUT_PNG)
        return

    t0 = time.perf_counter()
    xs = [-1.0, -0.8, -0.6, -0.4, -0.2] if not args.fast else [-0.8, -0.6, -0.4]
    ys = [0.75, 0.85, 0.95] if not args.fast else [0.85]
    hs = [0.6, 0.8, 1.0, 1.2] if not args.fast else [0.8, 1.0]
    cfgs = [dict(base_xy=(x, y), pedestal_h=h) for h in hs for y in ys for x in xs]
    print(f"A. 받침대 sweep: {len(cfgs)} 구성 x {len(_targets_for('top2')) + len(_targets_for('top1'))} 목표")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        sweep = list(ex.map(eval_config, cfgs, [args.fast] * len(cfgs)))
    sweep.sort(key=lambda r: (-r["frac_ok"], -r["src_frac"], r["cfg"]["pedestal_h"]))
    best = sweep[0]
    for r in sweep[:6]:
        print(f"   base {r['cfg']['base_xy']} h={r['cfg']['pedestal_h']}: ok {r['frac_ok']:.0%} "
              f"(src {r['src_frac']:.0%}, dst {r['dst_frac']:.0%}) fails={r['fails']}")
    print(f"   -> 최적 고정 베이스 {best['cfg']}  [{time.perf_counter() - t0:.0f} s]")

    print("B. 리니어 트랙 ±0.6 m (레일+캐리지 0.12 m 만큼 받침대를 낮춰 팔 베이스 높이를 고정 구성과 맞춤)")
    from arm import TRACK_STACK_H
    track_cfg = dict(best["cfg"], track_range=0.6, pedestal_h=round(best["cfg"]["pedestal_h"] - TRACK_STACK_H, 3))
    track = eval_config(track_cfg, args.fast)
    print(f"   track: ok {track['frac_ok']:.0%} (src {track['src_frac']:.0%}, dst {track['dst_frac']:.0%}) fails={track['fails']}")

    real_sum, real = {}, None
    if not args.skip_real and _real_sessions():
        print("C/D. 실측 픽 포즈 (트랙 포함 구성)")
        real = eval_real(track_cfg, args.fast, args.max_frames)
        real_sum = summarize_real(real)
        print(f"   {real_sum.get('n_poses')} poses: reach {real_sum.get('reachable')}, descent_free {real_sum.get('descent_free')}, "
              f"approach_free {real_sum.get('approach_free')}, place {real_sum.get('place_reachable')}, fails {real_sum.get('fails')}")
        if real_sum.get("cycle_s"):
            print(f"   cycle {real_sum['cycle_s']}")
        print("   by layer:", real_sum.get("by_layer"))
        # 고정 베이스로도 실측 포즈를 평가해 트랙의 효과를 분리
        real_fixed = eval_real(best["cfg"], args.fast, args.max_frames)
        real_sum["fixed_base"] = {k: v for k, v in summarize_real(real_fixed).items() if k in ("reachable", "descent_free", "approach_free", "place_reachable", "fails", "cycle_s")}
        print(f"   fixed base: {real_sum['fixed_base']}")
    else:
        print("C/D. 실측 데이터 없음 — 건너뜀")

    pub = dict(meta=dict(date=time.strftime("%Y-%m-%d"), arm="UR10e (MuJoCo Menagerie, reach 1.3 m)", units="m, s",
                         method=("DLS IK (pos<1 mm, rot<0.5 deg) with joint limits; collision = MuJoCo contacts between arm geoms "
                                 "and environment (cup vs target box allowed); cycle = trapezoidal joint profiles at UR10e "
                                 "speed limits (120/180 deg/s) + 0.3 s dwell x2; real poses = detector output on the 30 real frames "
                                 "mapped to base_link via the top-down T_base_cam; neighbours rebuilt as static boxes"),
                         fast=args.fast),
               config_sweep=[dict(base_xy=list(r["cfg"]["base_xy"]), pedestal_h=r["cfg"]["pedestal_h"], frac_ok=r["frac_ok"],
                                  src_frac=r["src_frac"], dst_frac=r["dst_frac"], fails=r["fails"]) for r in sweep],
               best_fixed=dict(cfg=dict(base_xy=list(best["cfg"]["base_xy"]), pedestal_h=best["cfg"]["pedestal_h"]),
                               frac_ok=best["frac_ok"], src_frac=best["src_frac"], dst_frac=best["dst_frac"], fails=best["fails"]),
               track=dict(cfg=dict(base_xy=list(track_cfg["base_xy"]), pedestal_h=track_cfg["pedestal_h"], track_range=0.6,
                                   arm_base_height_m=round(track_cfg["pedestal_h"] + TRACK_STACK_H, 3)),
                          frac_ok=track["frac_ok"], src_frac=track["src_frac"], dst_frac=track["dst_frac"], fails=track["fails"]),
               real_poses=real_sum)
    OUT_PUB.parent.mkdir(parents=True, exist_ok=True)
    OUT_PUB.write_text(json.dumps(pub, indent=1, ensure_ascii=False), encoding="utf-8")
    OUT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOCAL.write_text(json.dumps(dict(pub, real_rows=(real or {}).get("rows")), indent=1, ensure_ascii=False), encoding="utf-8")
    chart(sweep, best, track, real_sum, OUT_PNG)
    print(f"saved {OUT_PUB}, {OUT_PNG}  [{time.perf_counter() - t0:.0f} s]")


if __name__ == "__main__":
    main()
