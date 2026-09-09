# -*- coding: utf-8 -*-
"""셀 트윈의 UR10e(+리니어 트랙)를 rviz 가 그릴 수 있는 URDF 로 옮긴다.

    .venv\\Scripts\\python.exe tools/make_ur10e_urdf.py            # ros2_ws/src/twin_bridge/urdf/ur10e_track.urdf

왜: 트윈은 MuJoCo MJCF 로 팔을 돌리고, ROS 쪽 rviz 는 URDF 만 안다. 두 파일에 같은 기구학을 손으로 두 번 적으면 반드시 어긋나므로
MJCF(sim/assets/ur10e/ur10e.xml, Menagerie BSD-3)에서 바디 위치·쿼터니언·관절 축·한계를 그대로 읽어 URDF 를 생성한다.
받침대·트랙 위치도 sim/arm.py 의 arm_mount_xml 과 같은 값을 쓴다(둘이 어긋나면 rviz 의 팔이 트윈과 다른 자리에 선다).

생성물의 검증: tests/test_urdf_fk.py 가 무작위 관절각에서 URDF 순기구학과 MuJoCo 순기구학의 TCP 를 비교한다(< 0.1 mm).
시각 요소는 링크를 잇는 원기둥으로 간략히 만든다 — 메시(.obj)는 레포에 두지 않는다.
"""
from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MJCF = ROOT / "sim" / "assets" / "ur10e" / "ur10e.xml"
OUT = ROOT / "ros2_ws" / "src" / "twin_bridge" / "urdf" / "ur10e_track.urdf"

CHAIN = ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]
LIMITED = {"elbow_joint": 3.1415}           # MJCF class="size3_limited"
DEFAULT_RANGE = 6.28319
DEFAULT_AXIS = (0.0, 1.0, 0.0)              # MJCF default class="ur10e" 의 joint axis
VEL, EFFORT = 3.14, 330.0                   # rviz 표시용 (제어에는 쓰지 않는다)


def quat_to_rpy(q) -> tuple:
    """MuJoCo 쿼터니언 (w, x, y, z) -> URDF rpy (고정축 XYZ)."""
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(R[2, 0]) < 0.999999:
        roll, yaw = math.atan2(R[2, 1], R[2, 2]), math.atan2(R[1, 0], R[0, 0])
    else:                                    # 짐벌 근방
        roll, yaw = math.atan2(-R[1, 2], R[1, 1]), 0.0
    return roll, pitch, yaw


def parse_mjcf() -> dict:
    """MJCF 에서 링크 체인의 (pos, quat, 관절 이름·축·한계) 를 읽는다."""
    root = ET.parse(MJCF).getroot()
    base = root.find("worldbody").find("body[@name='base']")
    links, body = {}, base
    links["base"] = dict(pos=[float(v) for v in (base.get("pos") or "0 0 0").split()],
                         quat=[float(v) for v in (base.get("quat") or "1 0 0 0").split()], joint=None)
    for name in CHAIN:
        child = None
        for b in body.iter("body"):
            if b.get("name") == name:
                child = b
                break
        if child is None:
            raise SystemExit(f"MJCF 에 body {name} 이 없다")
        j = child.find("joint")
        axis = tuple(float(v) for v in (j.get("axis") or " ".join(str(a) for a in DEFAULT_AXIS)).split())
        lim = LIMITED.get(j.get("name"), DEFAULT_RANGE)
        if j.get("range"):
            lo, hi = (float(v) for v in j.get("range").split())
        else:
            lo, hi = -lim, lim
        links[name] = dict(pos=[float(v) for v in (child.get("pos") or "0 0 0").split()],
                           quat=[float(v) for v in (child.get("quat") or "1 0 0 0").split()],
                           joint=dict(name=j.get("name"), axis=axis, lower=lo, upper=hi))
        body = child
    site = body.find("site[@name='attachment_site']")
    links["tool"] = dict(pos=[float(v) for v in site.get("pos").split()],
                         quat=[float(v) for v in site.get("quat").split()], joint=None)
    return links


def urdf(links: dict, base_xy, pedestal_h: float, track_range: float, tool_len: float, pedestal_r: float) -> str:
    """arm_mount_xml 과 같은 배치로 base_link -> 받침대 -> (트랙) -> UR10e -> 툴 -> tcp 를 만든다."""
    o = []
    a = o.append

    def origin(pos, quat=(1, 0, 0, 0), pad="    "):
        r, p, y = quat_to_rpy(quat)
        return f'{pad}<origin xyz="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" rpy="{r:.6f} {p:.6f} {y:.6f}"/>'

    def cylinder_link(name, length, radius, color="0.82 0.82 0.82 1", offset=(0, 0, 0)):
        a(f'  <link name="{name}">')
        if length > 0:
            a('    <visual>')
            a(f'      <origin xyz="{offset[0]:.4f} {offset[1]:.4f} {offset[2]:.4f}" rpy="0 0 0"/>')
            a(f'      <geometry><cylinder length="{length:.4f}" radius="{radius:.4f}"/></geometry>')
            a(f'      <material name="{name}_m"><color rgba="{color}"/></material>')
            a('    </visual>')
        a('  </link>')

    a('<?xml version="1.0"?>')
    a('<!-- tools/make_ur10e_urdf.py 가 sim/assets/ur10e/ur10e.xml(MJCF, Menagerie BSD-3)에서 생성한다. 손으로 고치지 말 것. -->')
    a('<robot name="ur10e_track">')
    # 받침대 (셀 좌표 base_link 기준)
    a('  <link name="base_link"/>')
    cylinder_link("pedestal", pedestal_h, pedestal_r, "0.50 0.52 0.55 1", (0, 0, pedestal_h / 2))
    a('  <joint name="pedestal_fixed" type="fixed">')
    a('    <parent link="base_link"/><child link="pedestal"/>')
    a(origin([base_xy[0], base_xy[1], 0.0]))
    a('  </joint>')

    if track_range > 0:
        cylinder_link("rail", 0.06, 0.05, "0.35 0.36 0.38 1")
        a('  <joint name="rail_fixed" type="fixed">')
        a('    <parent link="pedestal"/><child link="rail"/>')
        a(origin([0.0, 0.0, pedestal_h + 0.03]))
        a('  </joint>')
        cylinder_link("carriage", 0.06, 0.18, "0.30 0.30 0.32 1", (0, 0, 0.03))
        a('  <joint name="track_joint" type="prismatic">')
        a('    <parent link="rail"/><child link="carriage"/>')
        a(origin([0.0, 0.0, 0.03]))
        a('    <axis xyz="1 0 0"/>')
        a(f'    <limit lower="{-track_range:.3f}" upper="{track_range:.3f}" effort="3000" velocity="1.0"/>')
        a('  </joint>')
        arm_parent, arm_z = "carriage", 0.06
    else:
        arm_parent, arm_z = "pedestal", pedestal_h

    # UR10e 체인
    b = links["base"]
    cylinder_link("ur_base", 0.12, 0.08, "0.28 0.28 0.28 1", (0, 0, 0.06))
    a('  <joint name="ur_base_fixed" type="fixed">')
    a(f'    <parent link="{arm_parent}"/><child link="ur_base"/>')
    a(origin([0.0, 0.0, arm_z], b["quat"]))
    a('  </joint>')

    parent = "ur_base"
    names = CHAIN + ["tool"]
    for i, name in enumerate(names):
        L = links[name]
        nxt = links[names[i + 1]] if i + 1 < len(names) else None
        seg = float(np.linalg.norm(nxt["pos"])) if nxt is not None else tool_len
        radius = 0.075 if i < 3 else 0.05
        color = "0.49 0.678 0.8 1" if i in (1, 2) else "0.82 0.82 0.82 1"
        # 시각화: 자식 관절 원점 쪽으로 뻗은 원기둥 하나 (메시 대신)
        if nxt is not None and seg > 1e-4:
            d = np.asarray(nxt["pos"], float) / seg
            mid = (np.asarray(nxt["pos"], float) / 2.0).tolist()
        else:
            d, mid = np.array([0.0, 0.0, 1.0]), [0.0, 0.0, seg / 2.0]
        a(f'  <link name="{name}">')
        a('    <visual>')
        # 원기둥 축(+z)을 d 방향으로 돌린다
        axis = np.cross([0.0, 0.0, 1.0], d)
        s, c = float(np.linalg.norm(axis)), float(np.dot([0.0, 0.0, 1.0], d))
        if s < 1e-9:
            rpy = (0.0, 0.0, 0.0) if c > 0 else (math.pi, 0.0, 0.0)
        else:
            ang = math.atan2(s, c)
            k = axis / s
            # 축-각 -> rpy
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
            pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
            rpy = (math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0]))
        a(f'      <origin xyz="{mid[0]:.4f} {mid[1]:.4f} {mid[2]:.4f}" rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>')
        a(f'      <geometry><cylinder length="{max(seg, 0.02):.4f}" radius="{radius:.4f}"/></geometry>')
        a(f'      <material name="{name}_m"><color rgba="{color}"/></material>')
        a('    </visual>')
        a('  </link>')
        j = L["joint"]
        if j is None:                        # tool: 플랜지 사이트에 고정
            a(f'  <joint name="{name}_fixed" type="fixed">')
            a(f'    <parent link="{parent}"/><child link="{name}"/>')
            a(origin(L["pos"], L["quat"]))
            a('  </joint>')
        else:
            a(f'  <joint name="{j["name"]}" type="revolute">')
            a(f'    <parent link="{parent}"/><child link="{name}"/>')
            a(origin(L["pos"], L["quat"]))
            a(f'    <axis xyz="{j["axis"][0]:.0f} {j["axis"][1]:.0f} {j["axis"][2]:.0f}"/>')
            a(f'    <limit lower="{j["lower"]:.5f}" upper="{j["upper"]:.5f}" effort="{EFFORT}" velocity="{VEL}"/>')
            a('  </joint>')
        parent = name

    a('  <link name="tcp"/>')                # 흡착 컵 끝 (트윈의 site "tcp" 와 같은 자리)
    a('  <joint name="tcp_fixed" type="fixed">')
    a(f'    <parent link="tool"/><child link="tcp"/>')
    a(origin([0.0, 0.0, tool_len]))
    a('  </joint>')
    a('</robot>')
    return "\n".join(o) + "\n"


def main():
    import sim.arm as arm_mod                                   # TOOL_LEN 등 같은 상수를 쓴다
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-xy", type=float, nargs=2, default=(-0.6, 0.75))
    ap.add_argument("--pedestal-h", type=float, default=1.08)
    ap.add_argument("--track-range", type=float, default=0.6)
    ap.add_argument("--pedestal-r", type=float, default=0.16)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    links = parse_mjcf()
    text = urdf(links, args.base_xy, args.pedestal_h, args.track_range, arm_mod.TOOL_LEN, args.pedestal_r)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"saved {args.out}  (base_xy={tuple(args.base_xy)} pedestal_h={args.pedestal_h} track={args.track_range} m)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
