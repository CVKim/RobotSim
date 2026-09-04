# -*- coding: utf-8 -*-
"""디팔레타이징 셀 디지털 트윈 (MuJoCo MJCF 생성기).

실측 30프레임에서 역산한 지오메트리를 그대로 옮긴다 (단위: m, 월드 Z+ = 위):

  카메라        탑다운, 바닥 위 4.183 m (실측 빈 프레임 깊이 4183 mm)
                f = 517 px @ 640x480  ->  fovy = 2*atan(240/517) = 49.79 deg
  팔레트 데크   바닥 위 0.643 m  (= 4.183 - 3.540; 실측 1층 상면 3257 mm + 박스높이 283 mm)
  박스          0.293 x 0.219 x 0.283 m, 격자 피치 0.30 x 0.225 m (isaac_sdg_boxdrop.py 와 동일)
  적재 배치     4열 x 3행 x N층 (실측 세션당 층별 12개)
  목적지        카메라 좌표 (-1191.4, -38.3) mm  (robotsim_perception/planner.py DEFAULT_DEST_XY_MM)

좌표 대응: 카메라는 월드 -Z 를 내려다본다(identity quat). 따라서
  카메라 X(이미지 오른쪽+) = 월드 +X,   카메라 Y(이미지 아래+) = 월드 -Y,   카메라 D(전방+) = 아래 방향 거리.

석션 EE 는 mocap 바디로 구동한다 (팔 기구학 없음 — 한계는 docs/41 참조).
"""
from __future__ import annotations

import numpy as np

# --- 실측 역산 상수 (mm -> m) --------------------------------------------
CAM_H = 4.183           # 바닥 위 카메라 높이
DECK_H = 0.643          # 바닥 위 팔레트 데크 상면
BOX = (0.293, 0.219, 0.283)
PITCH = (0.30, 0.225)
FOVY_DEG = float(np.degrees(2 * np.arctan(240.0 / 517.0)))   # 49.79
IMG_W, IMG_H = 640, 480
FOCAL_PX = 517.0
DEST_XY_MM = (-1191.4, -38.3)   # 카메라 좌표계 mm
N_COL, N_ROW = 4, 3


def grid_xy(col, row):
    """격자 (col,row) -> 월드 (x,y) m. 소스 팔레트 중심 = 원점."""
    x = (col - (N_COL - 1) / 2.0) * PITCH[0]
    y = (row - (N_ROW - 1) / 2.0) * PITCH[1]
    return x, y


def dest_world_xy():
    """목적지 스택 중심: 카메라 좌표 mm -> 월드 m (Y 부호 반전)."""
    return DEST_XY_MM[0] / 1000.0, -DEST_XY_MM[1] / 1000.0


def build_xml(layout, seed=0, jitter_mm=4.0, yaw_jitter_deg=1.2, dest_stack=0,
              tier_sheet=True, distractors=True):
    """layout: [(col,row,layer), ...] 소스 팔레트에 놓을 박스들. dest_stack: 목적지에 미리 쌓인 개수.

    반환 (xml_str, gt) — gt['boxes'] = [{'name','xyz_m','yaw_deg','layer'}...] 배치 시점 정답.
    실제 정답은 물리 안정화 후 mjData 에서 다시 읽는다 (cell_twin.settle 참조).
    """
    rng = np.random.default_rng(seed)
    j = jitter_mm / 1000.0
    hx, hy, hz = BOX[0] / 2, BOX[1] / 2, BOX[2] / 2

    parts = [f'''<mujoco model="depal_cell">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <global offwidth="{IMG_W}" offheight="{IMG_H}"/>
    <quality offsamples="4"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.22 0.24 0.26" rgb2="0.28 0.30 0.32"/>
    <material name="floor" texture="grid" texrepeat="8 8" reflectance="0.02"/>
    <material name="deck" rgba="0.55 0.30 0.14 1" reflectance="0.02"/>
    <material name="tier" rgba="0.30 0.32 0.36 1" reflectance="0.01"/>
    <material name="steel" rgba="0.62 0.64 0.68 1" reflectance="0.05"/>
''']
    # 박스마다 살짝 다른 색 — ToF 강도(I) 채널의 이음새 에지 신호를 만들기 위함
    n_box = len(layout) + dest_stack
    for i in range(n_box):
        g = 0.42 + 0.10 * rng.random()
        parts.append(f'    <material name="card{i}" rgba="{g:.3f} {g*0.63:.3f} {g*0.38:.3f} 1" reflectance="0.02"/>\n')
    parts.append('''  </asset>

  <worldbody>
    <light pos="0 0 4.0" dir="0 0 -1" diffuse="0.9 0.9 0.9" specular="0.15 0.15 0.15"/>
    <light pos="1.2 1.0 3.4" dir="-0.3 -0.25 -1" diffuse="0.35 0.35 0.35"/>
    <geom name="floor" type="plane" size="6 6 0.05" material="floor"/>
''')
    parts.append(f'    <camera name="tof" pos="0 0 {CAM_H:.4f}" quat="1 0 0 0" fovy="{FOVY_DEG:.4f}"/>\n')

    # 소스 팔레트 데크 (T-11 1100x1100, 두께 40mm) + 리프트 기둥
    dz = 0.02
    parts.append(f'    <geom name="deck_src" type="box" pos="0 0 {DECK_H - dz:.4f}" '
                 f'size="0.55 0.55 {dz:.4f}" material="deck"/>\n')
    parts.append(f'    <geom name="lift_src" type="box" pos="0 0 {(DECK_H - 2*dz)/2:.4f}" '
                 f'size="0.40 0.40 {(DECK_H - 2*dz)/2:.4f}" material="steel"/>\n')
    dx, dy = dest_world_xy()
    parts.append(f'    <geom name="deck_dst" type="box" pos="{dx:.4f} {dy:.4f} {DECK_H - dz:.4f}" '
                 f'size="0.55 0.55 {dz:.4f}" material="deck"/>\n')
    parts.append(f'    <geom name="lift_dst" type="box" pos="{dx:.4f} {dy:.4f} {(DECK_H - 2*dz)/2:.4f}" '
                 f'size="0.40 0.40 {(DECK_H - 2*dz)/2:.4f}" material="steel"/>\n')
    if tier_sheet:   # 층 사이 티어시트 (실측 빈 프레임에서 관측된 어두운 판)
        parts.append(f'    <geom name="tier_src" type="box" pos="0 0 {DECK_H + 0.002:.4f}" '
                     f'size="0.52 0.40 0.002" material="tier"/>\n')
    if distractors:
        # 주변 설비 (실측 장면의 컨베이어·리프트·프레임).
        # 중요: 실측 셀은 시야가 다양한 높이의 기구물로 차 있어 깊이 히스토그램에 지배적인
        # 단일 피크가 없다. 평평한 바닥만 두면 바닥 피크가 박스 상면 피크를 이겨
        # find_top_layer 가 바닥을 최상층으로 잡는다(트윈 1차 시도에서 실제 발생).
        parts.append('    <geom name="conv_l" type="box" pos="0 -1.05 0.35" size="1.7 0.22 0.35" material="steel"/>\n')
        parts.append('    <geom name="conv_r" type="box" pos="0 1.05 0.30" size="1.7 0.18 0.30" material="steel"/>\n')
        k = 0
        for (ex, ey) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            for _t in range(3):
                hxx, hyy = float(rng.uniform(0.10, 0.30)), float(rng.uniform(0.10, 0.30))
                hzz = float(rng.uniform(0.12, 0.62))
                px = ex * float(rng.uniform(0.75, 1.75)) + float(rng.normal(0, 0.10))
                py = ey * float(rng.uniform(0.70, 1.60)) + float(rng.normal(0, 0.10))
                # 소스 팔레트와 목적지 팔레트 위에는 설비를 두지 않는다.
                # (목적지 반경을 비우지 않으면 설비 블록이 적재 위치와 충돌해 놓은 박스가 튕겨 나간다)
                if abs(px) < 0.62 and abs(py) < 0.62:
                    continue
                if np.hypot(px - dx, py - dy) < 0.75:
                    continue
                parts.append(f'    <geom name="equip{k}" type="box" pos="{px:.3f} {py:.3f} {hzz:.3f}" '
                             f'size="{hxx:.3f} {hyy:.3f} {hzz:.3f}" material="steel"/>\n')
                k += 1

    gt = {"boxes": [], "camera": {"height_m": CAM_H, "fovy_deg": FOVY_DEG,
                                  "focal_px": FOCAL_PX, "img": [IMG_W, IMG_H]},
          "deck_h_m": DECK_H, "box_m": list(BOX), "dest_world_xy_m": [dx, dy]}

    def add_box(i, x, y, layer, base_z, tag):
        yaw = float(rng.normal(0, yaw_jitter_deg))
        xx = x + float(rng.normal(0, j))
        yy = y + float(rng.normal(0, j))
        zz = base_z + hz + layer * BOX[2] + 0.0015
        name = f"box{i}"
        parts.append(
            f'    <body name="{name}" pos="{xx:.5f} {yy:.5f} {zz:.5f}" euler="0 0 {yaw:.3f}">\n'
            f'      <freejoint name="{name}_j"/>\n'
            f'      <geom name="{name}_g" type="box" size="{hx:.4f} {hy:.4f} {hz:.4f}" '
            f'mass="6.0" material="card{i}" friction="0.9 0.02 0.001" '
            f'solref="0.004 1" solimp="0.95 0.99 0.001"/>\n'
            f'    </body>\n')
        gt["boxes"].append({"name": name, "xyz_m": [xx, yy, zz], "yaw_deg": yaw,
                            "layer": layer, "site": tag})

    i = 0
    base_src = DECK_H + (0.004 if tier_sheet else 0.0)
    for (c, r, layer) in layout:
        x, y = grid_xy(c, r)
        add_box(i, x, y, layer, base_src, "source")
        i += 1
    for k in range(dest_stack):
        c, r = k % N_COL, (k // N_COL) % N_ROW
        x, y = grid_xy(c, r)
        add_box(i, dx + x, dy + y, k // (N_COL * N_ROW), DECK_H, "dest")
        i += 1

    # 석션 EE (mocap 구동) — 팔 기구학 없음. 흡착은 weld equality 로 모델링.
    parts.append(f'''    <body name="suction_target" pos="0 0 {DECK_H + 1.0:.4f}" quat="1 0 0 0" mocap="true">
      <geom name="mocap_viz" type="box" size="0.06 0.06 0.012" rgba="0.1 0.7 0.9 0.25" contype="0" conaffinity="0"/>
    </body>
    <body name="suction" pos="0 0 {DECK_H + 1.0:.4f}">
      <freejoint name="suction_j"/>
      <geom name="suction_g" type="cylinder" size="0.055 0.015" mass="1.2" rgba="0.15 0.6 0.8 1"
            contype="2" conaffinity="1" friction="1.2 0.02 0.001"/>
    </body>
  </worldbody>

  <equality>
    <weld name="ee_mocap" body1="suction" body2="suction_target" solref="0.01 1" solimp="0.95 0.99 0.001"/>
''')
    for k in range(len(layout) + dest_stack):   # 흡착용 weld — 기본 비활성
        parts.append(f'    <weld name="grip{k}" body1="suction" body2="box{k}" active="false" '
                     f'solref="0.006 1" solimp="0.97 0.99 0.001"/>\n')
    parts.append('  </equality>\n</mujoco>\n')
    return "".join(parts), gt


def full_layout(n_layers=2, n_per_layer=None):
    """가득 찬 적재: 층당 4x3=12개. n_per_layer 로 층별 개수 축소 가능."""
    out = []
    for layer in range(n_layers):
        cells = [(c, r) for r in range(N_ROW) for c in range(N_COL)]
        if n_per_layer is not None:
            cells = cells[:n_per_layer]
        out += [(c, r, layer) for (c, r) in cells]
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    xml, gt = build_xml(full_layout(2), seed=0)
    out = Path(__file__).parent / "depal_cell.xml"
    out.write_text(xml, encoding="utf-8")
    print(f"wrote {out}  ({len(gt['boxes'])} boxes, fovy={FOVY_DEG:.2f}deg, cam_h={CAM_H}m)")
