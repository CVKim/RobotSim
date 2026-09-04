# -*- coding: utf-8 -*-
"""셀 트윈 렌더러 — MuJoCo 씬 -> 실측과 같은 형식의 ToF 프레임(X/Y/D/I) + **진짜 정답**.

이 모듈의 핵심 가치: 실측 30프레임은 정답이 검출기 자체 출력(pseudo-GT)이라 순환 참조였지만,
트윈에서는 박스의 실제 위치·치수·자세를 mjData 에서 그대로 읽을 수 있다.
따라서 검출기의 위치·치수 오차를 처음으로 절대 기준으로 측정할 수 있다.

좌표 대응 (sim/cell_scene.py 참조):
    ToF_X = world_X ,  ToF_Y = -world_Y ,  ToF_D = (CAM_H - world_Z)
    픽셀:  X = (u - cx) * D / f ,  Y = (v - cy) * D / f      (f = 517 px, 640x480)

강도(I) 모델: 실측에서 I * D_m^2 의 중앙값이 ~3527 로 거의 일정 (능동조명 ToF 의 1/D^2 감쇠).
    I = K * lum * / D_m^2      (lum = 렌더 휘도 0..1, K 는 아래 I_K 로 보정)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cell_scene import CAM_H, FOCAL_PX, IMG_H, IMG_W, BOX  # noqa: E402

SENTINEL_D = 16383.75
SENTINEL_XY = 8191.75
I_K = 8200.0          # 강도 보정 상수 — 상면 중앙값이 실측(~320~400)과 맞도록 설정
_UV = None


def _uv_grid():
    global _UV
    if _UV is None:
        cx, cy = (IMG_W - 1) / 2.0, (IMG_H - 1) / 2.0
        u, v = np.meshgrid(np.arange(IMG_W, dtype=np.float32),
                           np.arange(IMG_H, dtype=np.float32))
        _UV = ((u - cx) / FOCAL_PX, (v - cy) / FOCAL_PX)
    return _UV


class TwinRenderer:
    """MuJoCo 모델 -> ToF 프레임. mujoco.Renderer 2개(RGB/Depth)를 재사용한다."""

    def __init__(self, model, width=IMG_W, height=IMG_H, cam="tof"):
        import mujoco
        self.mj = mujoco
        self.model = model
        self.cam = cam
        self.r_rgb = mujoco.Renderer(model, height=height, width=width)
        self.r_dep = mujoco.Renderer(model, height=height, width=width)
        self.r_dep.enable_depth_rendering()

    def close(self):
        for r in (self.r_rgb, self.r_dep):
            try:
                r.close()
            except Exception:
                pass

    def frame(self, data, noise=None, rng=None):
        """반환 dict(X, Y, D, I) — 실측 세션과 같은 키/단위(mm). noise: None | 'tof'."""
        self.r_dep.update_scene(data, camera=self.cam)
        depth_m = np.asarray(self.r_dep.render(), dtype=np.float32)
        self.r_rgb.update_scene(data, camera=self.cam)
        rgb = np.asarray(self.r_rgb.render(), dtype=np.float32) / 255.0

        far = depth_m > (CAM_H + 2.0)          # 배경/무한대
        D = depth_m * 1000.0
        lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            I = I_K * lum / np.maximum(depth_m, 1e-3) ** 2
        I = np.nan_to_num(I, nan=0.0, posinf=0.0).astype(np.float32)

        valid = ~far & np.isfinite(D)
        if noise == "tof":
            rng = rng or np.random.default_rng(0)
            # (1) 강도 의존 깊이 노이즈: sigma(mm) = 180.3 * I^-0.805 (tools/tof_noise_study.py 실측)
            sig = np.clip(180.3 * np.power(np.maximum(I, 1.0), -0.805), 0.3, 40.0)
            D = D + rng.normal(0, 1, D.shape).astype(np.float32) * sig
            # (2) 결손 구조: 에지 드롭아웃 + 스펙클 + 대면적 블롭
            #     tools/sdg_dataset.add_tof_noise 와 같은 레시피 (실측 유효율 ~60% 재현으로 검증됨).
            #     실측 결손은 공간적으로 뭉쳐 있어, 픽셀 단위 랜덤 드롭과는 성격이 다르다.
            import cv2
            g = cv2.morphologyEx(D.astype(np.float32), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            drop = (rng.random(D.shape) < 0.55) & (g > 30)
            drop |= rng.random(D.shape) < 0.04
            h, w = D.shape
            blob = np.zeros((h, w), np.uint8)
            for _ in range(int(rng.integers(6, 18))):
                cv2.ellipse(blob, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                            (int(rng.integers(8, 80)), int(rng.integers(8, 80))),
                            float(rng.uniform(0, 180)), 0, 360, 1, -1)
            drop |= blob > 0
            valid &= ~drop

        ux, uy = _uv_grid()
        X = (ux * D).astype(np.float32)
        Y = (uy * D).astype(np.float32)
        D = D.astype(np.float32)
        X[~valid], Y[~valid] = SENTINEL_XY, SENTINEL_XY
        D[~valid] = SENTINEL_D
        # 주의: 강도(I)는 무효 픽셀에서도 0 으로 만들지 않는다. 실제 ToF 는 깊이를 기각한
        # 픽셀에서도 진폭을 반환하며, I 를 0 으로 찍으면 인공적인 고대비 스펙클이 생겨
        # _intensity_edges 의 Canny 가 화면 전체를 에지로 채우고 분할이 무너진다
        # (트윈 2차 시도에서 실제로 검출 0 을 유발).
        return {"X": X, "Y": Y, "D": D, "I": I, "rgb": None}


def settle(model, data, steps=1200):
    import mujoco
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return data


def ground_truth(model, data, top_only=True, tol_mm=45.0):
    """물리 안정화 후의 **실제** 박스 상면 정보 (ToF 카메라 좌표 mm).

    반환 [{'name','center_mm'(X,Y),'top_d_mm','dims_mm'(L,W),'ang_deg','layer_top'}...]
    top_only=True 면 최상층(카메라 최근접 상면) ±tol 안의 박스만 — 검출기가 보는 대상과 일치.
    """
    import mujoco
    out = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if not name or not name.startswith("box"):
            continue
        p = data.xpos[i]
        R = data.xmat[i].reshape(3, 3)
        # 상면 중심 = 바디 중심 + 로컬 +Z * hz  (회전 반영)
        top_c = p + R @ np.array([0.0, 0.0, BOX[2] / 2.0])
        # 상면 평면상의 변 방향 -> 이미지 평면 투영 각도 및 실치수
        e1, e2 = R @ np.array([1.0, 0, 0]), R @ np.array([0, 1.0, 0])
        L = BOX[0] * float(np.hypot(e1[0], e1[1]))    # 기울면 투영 길이가 줄어듦
        W = BOX[1] * float(np.hypot(e2[0], e2[1]))
        ang = float(np.degrees(np.arctan2(-e1[1], e1[0])) % 180.0)   # ToF Y = -world Y
        if L < W:
            L, W, ang = W, L, (ang + 90.0) % 180.0
        out.append({"name": name,
                    "center_mm": (float(top_c[0] * 1000.0), float(-top_c[1] * 1000.0)),
                    "top_d_mm": float((CAM_H - top_c[2]) * 1000.0),
                    "dims_mm": (round(L * 1000.0, 1), round(W * 1000.0, 1)),
                    "ang_deg": round(ang, 1),
                    "tilt_deg": round(float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), 0, 1)))), 2)})
    if not out:
        return out
    if top_only:
        dmin = min(b["top_d_mm"] for b in out)
        out = [b for b in out if b["top_d_mm"] - dmin < tol_mm]
    return sorted(out, key=lambda b: b["center_mm"])


def match(gt, det, tol_mm=90.0):
    """정답-검출 1:1 탐욕 매칭 (중심거리). 반환 (pairs, missed_gt, false_pos)."""
    used, pairs, fp = set(), [], []
    for b in det:
        c = np.array(b["center_mm"][:2], float)
        best, bd = None, 1e9
        for i, g in enumerate(gt):
            if i in used:
                continue
            dist = float(np.hypot(*(c - np.array(g["center_mm"]))))
            if dist < tol_mm and dist < bd:
                bd, best = dist, i
        if best is None:
            fp.append(b)
        else:
            used.add(best)
            pairs.append((gt[best], b, bd))
    return pairs, [g for i, g in enumerate(gt) if i not in used], fp
