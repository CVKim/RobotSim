import blenderproc as bproc  # 반드시 파일 첫 문장 (bproc 규칙)
# -*- coding: utf-8 -*-
# T5 합성데이터 1차: 실측 지오메트리 매칭 박스 적재 씬 (BlenderProc2).
#   박스 293×219×283mm (σ 지터), 카메라 탑다운 f≈517px·높이 2.973m (실측 역산)
#   실행: blenderproc run tools/sdg_boxdrop.py -- --n_scenes 5 --out explore/sdg_test
import argparse
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--n_scenes", type=int, default=5)
parser.add_argument("--out", default=r"E:\Robot_Sim\explore\sdg_test")
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

BOX = np.array([0.293, 0.219, 0.283])  # m
rng = np.random.default_rng(0)

bproc.init()

# 바닥(팔레트 면) + 주변 벽 없는 심플 씬
ground = bproc.object.create_primitive("PLANE", scale=[1.2, 1.2, 1])
ground.enable_rigidbody(active=False, collision_shape="BOX")

light = bproc.types.Light()
light.set_type("AREA")
light.set_location([0.5, -0.5, 2.5])
light.set_energy(rng.uniform(80, 200))

# 카메라 내파라미터: 실측 역산 (640x480, f=517px)
K = np.array([[517.0, 0, 320.0], [0, 517.0, 240.0], [0, 0, 1.0]])
bproc.camera.set_intrinsics_from_K_matrix(K, 640, 480)

for scene_i in range(args.n_scenes):
    boxes = []
    n_layer = rng.integers(6, 13)  # 상층 박스 수 랜덤 (픽킹 진행 상태 재현)
    grid = [(gx, gy) for gx in range(4) for gy in range(3)]
    rng.shuffle(grid)
    for k, (gx, gy) in enumerate(grid[:n_layer]):
        dims = BOX * (1 + rng.normal(0, [0.031, 0.045, 0.007]))
        b = bproc.object.create_primitive("CUBE", scale=list(dims / 2))
        x = (gx - 1.5) * 0.30 + rng.normal(0, 0.004)
        y = (gy - 1.0) * 0.225 + rng.normal(0, 0.004)
        b.set_location([x, y, dims[2] / 2 + 0.02 + 0.01 * k])
        b.set_rotation_euler([0, 0, rng.normal(0, 0.02)])
        mat = bproc.material.create(f"card_{scene_i}_{k}")
        base = rng.uniform(0.35, 0.55)
        mat.set_principled_shader_value("Base Color",
                                        [base, base * 0.62, base * 0.38, 1])
        mat.set_principled_shader_value("Roughness", rng.uniform(0.6, 0.9))
        b.replace_materials(mat)
        b.enable_rigidbody(active=True, collision_shape="BOX")
        b.set_cp("category_id", 1)
        boxes.append(b)

    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=0.5, max_simulation_time=2.0, check_object_interval=0.25)

    if scene_i == 0:  # 렌더 패스 활성화는 1회만 허용
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"], default_values={"category_id": 0})

    # 탑다운 카메라: 상면(0.283m) 위 2.973m
    cam_h = 0.283 + 2.973 + rng.normal(0, 0.01)
    # Blender 카메라는 기본이 로컬 -Z 방향(아래)을 봄 — 회전 불필요
    pose = bproc.math.build_transformation_mat(
        [rng.normal(0, 0.02), rng.normal(0, 0.02), cam_h],
        [0, 0, rng.normal(0, 0.03)])
    bproc.camera.add_camera_pose(pose)

    data = bproc.renderer.render()
    bproc.writer.write_hdf5(args.out, data, append_to_existing_output=True)

    # 다음 씬 준비: 박스 제거, 카메라 포즈 초기화
    for b in boxes:
        b.delete()
    bproc.utility.reset_keyframes()

print(f"done: {args.n_scenes} scenes -> {args.out}")
