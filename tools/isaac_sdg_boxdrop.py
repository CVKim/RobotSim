# -*- coding: utf-8 -*-
"""Isaac Sim 4.5 Replicator 합성데이터 — BlenderProc 씬(sdg_boxdrop.py v4)과 동일 조건 재현.

실측 매칭: 640x480, f=517px -> focal 16.93mm @ aperture 20.955mm, 박스 293x219x283mm.
랜덤화: 카메라 높이(=바닥 깊이 3.26~4.26m 등가), 박스 가시 수(슬롯별 near/far 선택),
방해물 8개, 조명. 물리 드롭은 생략(박스는 바닥에 정치) — BlenderProc과의 차이로 명시.
실행: OMNI_KIT_ACCEPT_EULA=YES python tools/isaac_sdg_boxdrop.py --frames 300 --out E:/Robot_Sim/explore/isaac_data
"""
import argparse
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=5)
ap.add_argument("--out", default=r"E:\Robot_Sim\explore\isaac_test")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402

BOX = (0.293, 0.219, 0.283)
FAR = (15.0, 15.0, 0.15)
rng = np.random.default_rng(0)

with rep.new_layer():
    cam = rep.create.camera(position=(0, 0, 3.256), look_at=(0, 0, 0),
                            focal_length=16.93, horizontal_aperture=20.955,
                            clipping_range=(0.05, 30.0))
    rp = rep.create.render_product(cam, (640, 480))

    ground = rep.create.plane(scale=30, position=(0, 0, 0),
                              semantics=[("class", "ground")])

    # 12 슬롯 그리드 (BlenderProc과 동일 배치)
    slots = [((gx - 1.5) * 0.30, (gy - 1.0) * 0.225) for gx in range(4) for gy in range(3)]
    boxes = []
    for k, (sx, sy) in enumerate(slots):
        b = rep.create.cube(scale=BOX, position=(sx, sy, BOX[2] / 2),
                            semantics=[("class", f"box{k + 1}")])
        boxes.append((b, sx, sy))

    distractors = [rep.create.cube(scale=(0.3, 0.3, 0.3), position=(3, 3, 0.15),
                                   semantics=[("class", "distractor")]) for _ in range(8)]

    light = rep.create.light(light_type="Sphere", intensity=30000, position=(0, 0, 3.0),
                             scale=2.0)

    with rep.trigger.on_frame(num_frames=args.frames):
        # 바닥 깊이 랜덤화 등가: 카메라 높이 3.256~4.256m
        with cam:
            rep.modify.pose(position=rep.distribution.uniform((-0.02, -0.02, 3.256),
                                                             (0.02, 0.02, 4.256)),
                            look_at=(0, 0, 0))
        # 박스: 슬롯 위치 vs 원거리 대기 (가시 수 랜덤)
        for b, sx, sy in boxes:
            with b:
                rep.modify.pose(
                    position=rep.distribution.choice([(sx, sy, BOX[2] / 2), FAR],
                                                     weights=[0.75, 0.25]),
                    rotation=rep.distribution.uniform((0, 0, -2), (0, 0, 2)))
                rep.randomizer.color(colors=rep.distribution.uniform((0.35, 0.22, 0.13),
                                                                    (0.55, 0.34, 0.21)))
        # 방해물: 팔레트 밖 링 영역
        for d in distractors:
            with d:
                rep.modify.pose(position=rep.distribution.uniform((-1.6, -1.6, 0.05),
                                                                 (1.6, 1.6, 0.5)),
                                rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 180)),
                                scale=rep.distribution.uniform((0.05, 0.05, 0.05),
                                                               (0.5, 0.5, 0.5)))
                rep.randomizer.color(colors=rep.distribution.uniform((0.15, 0.15, 0.15),
                                                                    (0.5, 0.5, 0.55)))
        with light:
            rep.modify.pose(position=rep.distribution.uniform((-1, -1, 2), (1, 1, 3.2)))
            rep.modify.attribute("intensity", rep.distribution.uniform(15000, 60000))

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=args.out, rgb=True, distance_to_image_plane=True,
                      semantic_segmentation=True, colorize_semantic_segmentation=False)
    writer.attach([rp])

rep.orchestrator.run_until_complete()
print(f"done: {args.frames} frames -> {args.out}")
app.close()
