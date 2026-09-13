# -*- coding: utf-8 -*-
"""장애물을 돌아가는 경로 계획을 넣고 직선 이동과 비교한다 (셀 트윈).

    .venv\\Scripts\\python.exe tools/twin_path_plan.py                 # results/twin_path_plan.json + assets/twin_path_plan.png
    .venv\\Scripts\\python.exe tools/twin_path_plan.py --seeds 3 --dest-stack 6

지금까지 팔 실행기는 TCP 직선 이동만 했다. 직선 위에 박스나 설비가 있으면 그 픽을 통째로 포기했고, 문서에도
'장애물을 돌아가는 경로 계획은 없다' 고 적어 두었다. 여기서 관절 공간 RRT-Connect(+단축)를 넣고 같은 이동을 두 방식으로 재 본다.

  비교 대상: 소스 팔레트의 각 박스 위 자세 -> 목적지 슬롯 위 자세 (이송 구간)
  지표: 성공률(충돌 없이 갈 수 있는가), 계획 시간, 관절 이동량, 관절 속도 한계로 계산한 실행 시간

충돌 판정은 트윈이 쓰는 것과 같다(sim/arm.py). 계획기가 실행기와 다른 기준을 보면 '계획은 됐는데 돌리면 부딪히는' 일이 생긴다.
주의: 이 실험은 **빈 팔** 기준이다. 박스를 든 상태의 충돌(페이로드)은 실행기 쪽에서 따로 검사한다(docs/41 7절).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT))

import plan_rrt  # noqa: E402
from arm import JOINTS as ARM_JOINTS  # noqa: E402
from arm import Arm, rot_from_approach_yaw  # noqa: E402
from cell_scene import BOX, DECK_H, build_xml, dest_slots, dest_world_xy, full_layout, grid_xy  # noqa: E402
from cell_twin import settle  # noqa: E402

ARM_CFG = dict(base_xy=(-0.6, 0.75), pedestal_h=1.08, track_range=0.6, meshes=False)
OUT_JSON = ROOT / "results" / "twin_path_plan.json"
OUT_PNG = ROOT / "assets" / "twin_path_plan.png"
CLEAR = 0.15          # 박스 상면 위 접근 높이 (실행기와 같은 clearance)


def make_cell(seed: int, dest_stack: int):
    import mujoco
    xml, _ = build_xml(full_layout(2), seed=seed, dest_stack=dest_stack, arm=ARM_CFG)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    settle(m, d, 1500)
    arm = Arm(m, d)
    arm.set_q(arm.home)
    return m, d, arm


def transfer_pairs(arm, rng):
    """소스 박스 위 자세 -> 목적지 슬롯 위 자세 쌍. IK 가 풀리는 것만 남긴다."""
    R = rot_from_approach_yaw((0.0, 0.0, -1.0), 0.0)
    z_src = DECK_H + 2 * BOX[2] + CLEAR
    z_dst = DECK_H + BOX[2] + CLEAR
    dx, dy = dest_world_xy()
    out = []
    slots = dest_slots()
    for k, (c, r) in enumerate([(c, r) for r in range(3) for c in range(4)]):
        x, y = grid_xy(c, r)
        sc, sr = slots[k % len(slots)]
        sx, sy = grid_xy(sc, sr)
        # 충돌을 피하는 IK 를 쓴다(solve_ik_multi 는 자세만 맞추므로 팔꿈치가 옆 박스에 박힌 해를 준다 —
        # 실제로 12칸 중 8칸이 그랬다). 실행기도 같은 함수를 쓴다.
        a, ca = arm.solve_ik_free(np.array([x, y, z_src]), R, rng=rng)
        b, cb = arm.solve_ik_free(np.array([dx + sx, dy + sy, z_dst]), R, rng=rng)
        if a.ok and b.ok and not ca and not cb:
            out.append((f"c{c}r{r}", a.q.copy(), b.q.copy()))
    return out


def evaluate(arm, pairs, seed: int):
    rows = []
    for name, qa, qb in pairs:
        straight_hit = arm.path_collides(qa, qb)
        t0 = time.perf_counter()
        path = plan_rrt.plan(arm, qa, qb, seed=seed)
        t_plan = time.perf_counter() - t0
        smooth = plan_rrt.shortcut(arm, path, seed=seed) if path else None
        row = {"pair": name, "straight_ok": straight_hit is None,
               "straight_hit": None if straight_hit is None else str(straight_hit[0][0]),
               "planned_ok": smooth is not None, "plan_s": round(t_plan, 3)}
        if smooth is not None:
            row["waypoints"] = len(smooth)
            row["cost_rad"] = round(plan_rrt.path_cost(smooth), 3)
            row["cost_rad_raw"] = round(plan_rrt.path_cost(path), 3)
            row["exec_s"] = round(float(sum(arm.segment_time(x, y) for x, y in zip(smooth[:-1], smooth[1:]))), 2)
            row["straight_exec_s"] = round(float(arm.segment_time(qa, qb)), 2)
            row["recheck_ok"] = not any(arm.path_collides(x, y) for x, y in zip(smooth[:-1], smooth[1:]))
        rows.append(row)
    return rows


def export_scene(m, d, arm):
    """트윈의 충돌 세계를 MoveIt 계획 장면으로 옮길 수 있게 원시 도형으로 뽑는다.

    팔 자신의 링크는 URDF 에 이미 있으므로 뺀다. 나머지(데크·팔레트 박스·설비 블록·바닥)는 위치·크기를 그대로 넘긴다 —
    두 계획기가 **같은 세계**를 보게 하는 것이 비교의 전제다."""
    import mujoco
    arm_gids = set(int(g) for g in getattr(arm, "arm_gids", []) or [])
    # URDF 에 이미 있는 구조물은 빼야 한다 — 같은 것을 계획 장면에도 올리면 로봇이 '자기 받침대와 충돌' 상태로 시작해
    # MoveIt 이 시작 자세를 억지로 빼내며 한참 돌아가는 경로를 낸다(실제로 비용이 3 -> 13 rad 로 뛰었다).
    in_urdf = {"pedestal", "track_rail", "carriage_g"}
    out = []
    for gid in range(m.ngeom):
        if gid in arm_gids:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"
        if name in in_urdf:
            continue
        t = int(m.geom_type[gid])
        kind = {mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
                mujoco.mjtGeom.mjGEOM_SPHERE: "sphere", mujoco.mjtGeom.mjGEOM_CAPSULE: "cylinder",
                mujoco.mjtGeom.mjGEOM_PLANE: "plane"}.get(t)
        if kind is None:
            continue
        out.append({"name": name, "kind": kind,
                    "size": [float(v) for v in m.geom_size[gid][:3]],
                    "pos": [float(v) for v in d.geom_xpos[gid]],
                    "mat": [float(v) for v in d.geom_xmat[gid]]})
    return out


def chart(rows, path):
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
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))

    n = len(rows)
    s_ok = sum(1 for r in rows if r["straight_ok"])
    p_ok = sum(1 for r in rows if r["planned_ok"])
    ax[0].bar(["직선 이동", "경로 계획"], [100 * s_ok / n, 100 * p_ok / n], color=["#9aa3ad", "#1d4ed8"], width=0.55)
    for i, v in enumerate([100 * s_ok / n, 100 * p_ok / n]):
        ax[0].text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10)
    ax[0].set_ylim(0, 112)
    ax[0].set_ylabel("갈 수 있는 이동 %")
    ax[0].set_title(f"같은 이동 {n}건: 직선이면 막히는 것이 {n - s_ok}건")

    blocked = [r for r in rows if not r["straight_ok"] and r["planned_ok"]]
    if blocked:
        ax[1].hist([r["exec_s"] - r["straight_exec_s"] for r in blocked], bins=12, color="#d97706")
        ax[1].set_xlabel("직선 대비 늘어난 실행 시간 (s)")
        ax[1].set_ylabel("건수")
        ax[1].set_title("돌아가는 값: 직선이 막힌 이동만")
    ok = [r for r in rows if r["planned_ok"]]
    ax[2].hist([r["plan_s"] * 1000 for r in ok], bins=12, color="#059669")
    ax[2].set_xlabel("계획 시간 (ms)")
    ax[2].set_ylabel("건수")
    ax[2].set_title("RRT-Connect + 단축 계획 시간")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dest-stack", type=int, default=6, help="목적지에 미리 쌓아 둔 박스 수 (장애물 역할)")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument("--dump-problems", type=Path, default=None,
                    help="MoveIt 비교용으로 같은 문제(시작·목표 관절각)와 장면을 JSON 으로 뽑는다")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    rows = []
    dump = {"joint_names": None, "scenes": []} if args.dump_problems else None
    for s in range(args.seeds):
        m, d, arm = make_cell(s, args.dest_stack)
        pairs = transfer_pairs(arm, np.random.default_rng(100 + s))
        r = evaluate(arm, pairs, seed=s)
        for x in r:
            x["seed"] = s
        rows += r
        if dump is not None:
            dump["joint_names"] = (["track_joint"] if arm.has_track else []) + list(ARM_JOINTS)
            dump["scenes"].append({
                "seed": s,
                "geoms": export_scene(m, d, arm),
                "problems": [{"pair": name, "start": [float(v) for v in qa], "goal": [float(v) for v in qb],
                              "straight_ok": bool(arm.path_collides(qa, qb) is None)}
                             for name, qa, qb in pairs],
            })
        n = len(r)
        print(f"  시드 {s}: 이동 {n}건 · 직선 가능 {sum(1 for x in r if x['straight_ok'])} · "
              f"계획 가능 {sum(1 for x in r if x['planned_ok'])} · "
              f"계획 시간 중앙 {np.median([x['plan_s'] for x in r])*1000:.0f} ms", flush=True)

    n = len(rows)
    s_ok = sum(1 for r in rows if r["straight_ok"])
    p_ok = sum(1 for r in rows if r["planned_ok"])
    blocked = [r for r in rows if not r["straight_ok"] and r["planned_ok"]]
    summary = {
        "what": "직선 이동 대비 관절 공간 RRT-Connect 경로 계획 (셀 트윈, 빈 팔 기준)",
        "scene": {"layers": 2, "dest_stack": args.dest_stack, "arm": ARM_CFG, "seeds": args.seeds},
        "n_moves": n,
        "straight_ok": s_ok, "planned_ok": p_ok,
        "straight_rate": round(s_ok / n, 3), "planned_rate": round(p_ok / n, 3),
        "unblocked": len(blocked),
        "plan_ms_median": round(float(np.median([r["plan_s"] for r in rows])) * 1000, 1),
        "plan_ms_p95": round(float(np.percentile([r["plan_s"] for r in rows], 95)) * 1000, 1),
        "extra_exec_s_median": (round(float(np.median([r["exec_s"] - r["straight_exec_s"] for r in blocked])), 2)
                                if blocked else None),
        "shortcut_gain": (round(float(np.mean([r["cost_rad_raw"] / max(r["cost_rad"], 1e-6) for r in rows
                                               if r.get("cost_rad")])), 2)),
        "recheck_all_ok": all(r.get("recheck_ok", True) for r in rows),
        "runs": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    chart(rows, OUT_PNG)
    if args.dump_problems:
        args.dump_problems.parent.mkdir(parents=True, exist_ok=True)
        args.dump_problems.write_text(json.dumps(dump, ensure_ascii=False), encoding="utf-8", newline=chr(10))
        print("문제·장면 내보냄:", args.dump_problems)
    print(f"직선 {s_ok}/{n} ({100*s_ok/n:.0f}%) · 계획 {p_ok}/{n} ({100*p_ok/n:.0f}%) · "
          f"직선이 막힌 {len(blocked)}건을 계획이 열었다 · 계획 시간 중앙 {summary['plan_ms_median']} ms")
    print("saved", args.out, "및", OUT_PNG)


if __name__ == "__main__":
    main()
