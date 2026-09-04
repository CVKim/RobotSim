# -*- coding: utf-8 -*-
"""카메라 좌표 -> 로봇 베이스 좌표 변환 + 6-DoF 픽 포즈.

지금까지 이 패키지의 출력(Box.center_mm, normal)은 전부 **ToF 카메라 좌표 mm** 이라
로봇이 그대로 실행할 수 없었다. 이 모듈이 그 배관을 채운다.

  T_base_cam : 카메라 프레임 -> 로봇 베이스 프레임 (4x4 동차변환, 단위 mm)
  PickPose   : 로봇 베이스 좌표의 위치 + 회전(요 포함) + 접근/후퇴 경로

좌표계 약속
  카메라 : X 오른쪽+, Y 아래+, D(=Z) 전방+   (실측 ToF .mim 와 동일)
  로봇   : 사용자가 정의 (통상 X 전방, Y 좌, Z 위)

요(yaw) 정의: 박스 상면 장축(L)의 방향. 카메라 좌표에서 `ang_deg`(0~180)로 주어지며
석션 그리퍼가 직사각형 패드/다중 컵 배열일 때 반드시 필요하다. 대칭 그리퍼면 무시해도 된다.

주의: 실제 T_base_cam 값은 로봇+카메라로 핸드아이 캘리브레이션을 해야 얻어진다.
이 모듈은 변환·포즈 구조와 왕복 검증만 제공하며, 기본값은 단위행렬(= 카메라 좌표 그대로)이다.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def rot_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def make_transform(R=None, t_mm=(0.0, 0.0, 0.0)) -> np.ndarray:
    """3x3 회전 + 이동(mm) -> 4x4 동차변환."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, float)
    T[:3, 3] = np.asarray(t_mm, float)
    return T


def invert(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def apply(T: np.ndarray, p_mm: Sequence[float]) -> np.ndarray:
    """점 변환 (mm)."""
    p = np.asarray(p_mm, float)
    return (T[:3, :3] @ p) + T[:3, 3]


def apply_vec(T: np.ndarray, v: Sequence[float]) -> np.ndarray:
    """방향벡터 변환 (이동 성분 제외)."""
    return T[:3, :3] @ np.asarray(v, float)


# 흔한 설치: 카메라가 셀 위에서 아래를 보고, 로봇 베이스는 바닥에 있음.
# 카메라 D(전방, 아래) -> 로봇 -Z,  카메라 Y(아래, 이미지) -> 로봇 -X 형태의 예시.
def topdown_camera_transform(cam_height_mm: float, yaw_deg: float = 0.0,
                             base_xy_mm: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    """탑다운 카메라의 전형적인 T_base_cam 예시 (실측 캘리브레이션 대체가 아님).

    로봇 베이스: X 전방, Y 좌, Z 위. 카메라는 (base_xy, cam_height) 에서 수직 하향.
    카메라 X(오른쪽)->로봇 +Y? 는 설치마다 다르므로 yaw_deg 로 맞춘다.
    """
    R_flip = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])   # Y,D 부호 반전
    R = rot_z(yaw_deg) @ R_flip
    return make_transform(R, (float(base_xy_mm[0]), float(base_xy_mm[1]), float(cam_height_mm)))


def load_transform(path) -> np.ndarray:
    """JSON 파일에서 T_base_cam 로드. {"T_base_cam": [[..4x4..]]} 또는 {"R":3x3,"t_mm":3}."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "T_base_cam" in d:
        T = np.asarray(d["T_base_cam"], float)
        if T.shape != (4, 4):
            raise ValueError(f"T_base_cam must be 4x4, got {T.shape}")
        return T
    return make_transform(d.get("R"), d.get("t_mm", (0, 0, 0)))


def save_transform(path, T: np.ndarray, meta: Optional[dict] = None):
    out = {"T_base_cam": np.asarray(T, float).tolist(), "units": "mm"}
    if meta:
        out["meta"] = meta
    Path(path).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


@dataclass
class PickPose:
    """로봇이 실행 가능한 6-DoF 픽 포즈 (베이스 좌표 mm)."""
    box_id: int
    position_mm: tuple          # 상면 중심 (X, Y, Z) 로봇 베이스 좌표
    approach: tuple             # 단위 접근 벡터 (박스로 향함) = -법선
    yaw_deg: float              # 상면 장축 방향 (베이스 좌표 XY 평면 기준)
    tilt_deg: float             # 접근 벡터와 베이스 -Z 사이 각
    pre_pick_mm: tuple          # 접근 시작점 = position - approach * clearance
    post_pick_mm: tuple         # 들어올린 뒤 지점
    clearance_mm: float
    dims_mm: tuple
    confidence: float

    def to_dict(self):
        d = asdict(self)
        for k in ("position_mm", "approach", "pre_pick_mm", "post_pick_mm", "dims_mm"):
            d[k] = [float(v) for v in d[k]]
        return d


def box_to_pick_pose(box, T_base_cam: np.ndarray, clearance_mm: float = 150.0,
                     lift_mm: float = 250.0) -> PickPose:
    """detect.Box (카메라 좌표) -> PickPose (로봇 베이스 좌표).

    box 는 center_mm(3), normal(3), ang_deg 또는 rect_px, dims_mm, confidence 를 갖는 객체/딕트.
    normal 은 카메라 쪽(-Z)을 향하므로 접근 벡터 = -normal.
    """
    g = (lambda k, d=None: box.get(k, d)) if isinstance(box, dict) else (lambda k, d=None: getattr(box, k, d))
    c_cam = np.asarray(g("center_mm"), float)
    if c_cam.size == 2:                       # tools 계열은 (X, Y) 만 -> depth 로 Z 보완
        c_cam = np.array([c_cam[0], c_cam[1], float(g("depth_mm", 0.0))])
    n_cam = np.asarray(g("normal", (0.0, 0.0, -1.0)), float)
    n_cam = n_cam / max(np.linalg.norm(n_cam), 1e-9)

    pos = apply(T_base_cam, c_cam)
    approach = apply_vec(T_base_cam, -n_cam)
    approach = approach / max(np.linalg.norm(approach), 1e-9)

    ang_cam = g("ang_deg")
    if ang_cam is None:
        rect = g("rect_px")
        ang_cam = float(rect[2]) if rect is not None else 0.0
    # 카메라 XY 의 장축 방향 벡터를 베이스로 옮겨 요를 다시 계산 (부호·회전 반영)
    d_cam = np.array([math.cos(math.radians(float(ang_cam))),
                      math.sin(math.radians(float(ang_cam))), 0.0])
    d_base = apply_vec(T_base_cam, d_cam)
    yaw = math.degrees(math.atan2(d_base[1], d_base[0])) % 180.0

    tilt = math.degrees(math.acos(float(np.clip(np.dot(approach, (0, 0, -1.0)), -1, 1))))
    pre = pos - approach * clearance_mm
    post = pos - approach * (clearance_mm + lift_mm)
    dims = g("dims_mm", (0.0, 0.0))
    return PickPose(
        box_id=int(g("id", g("box_id", -1)) or -1),
        position_mm=tuple(float(v) for v in pos),
        approach=tuple(float(v) for v in approach),
        yaw_deg=round(float(yaw), 2),
        tilt_deg=round(float(tilt), 2),
        pre_pick_mm=tuple(float(v) for v in pre),
        post_pick_mm=tuple(float(v) for v in post),
        clearance_mm=float(clearance_mm),
        dims_mm=(float(dims[0]), float(dims[1])),
        confidence=float(g("confidence", 0.0)),
    )


def suction_footprint_ok(frame, box, cup_radius_mm=45.0, min_valid=0.7,
                         max_plane_rms_mm=8.0) -> dict:
    """석션 컵 풋프린트 실행가능성: 컵 아래 상면 유효 픽셀 비율 + 평탄도.

    박스 단위 fill/plane_rms 만으로는 '중심부에 큰 결손이 있는 박스'를 걸러내지 못한다.
    반환 {'ok', 'valid_frac', 'plane_rms_mm', 'reason'}
    """
    import cv2
    g = (lambda k, d=None: box.get(k, d)) if isinstance(box, dict) else (lambda k, d=None: getattr(box, k, d))
    rect = g("rect_px")
    if rect is None:
        return {"ok": False, "valid_frac": 0.0, "plane_rms_mm": None, "reason": "no rect_px"}
    (cx, cy) = rect[0]
    D, valid = frame["D"] if isinstance(frame, dict) else frame.D, \
               frame["valid"] if isinstance(frame, dict) and "valid" in frame else getattr(frame, "valid", None)
    if valid is None:
        valid = (D < 16000)
    depth = float(g("depth_mm", np.median(D[valid]) if valid.any() else 0.0))
    # 컵 반경(mm) -> 픽셀 (f=517 기준, 깊이 비례)
    r_px = max(int(round(cup_radius_mm * 517.0 / max(depth, 1.0))), 3)
    m = np.zeros(D.shape, np.uint8)
    cv2.circle(m, (int(round(cx)), int(round(cy))), r_px, 1, -1)
    m = m > 0
    n = int(m.sum())
    if n == 0:
        return {"ok": False, "valid_frac": 0.0, "plane_rms_mm": None, "reason": "empty footprint"}
    vf = float((m & valid).sum() / n)
    rms = None
    sel = m & valid
    if sel.sum() >= 30:
        z = D[sel].astype(np.float64)
        rms = float(np.sqrt(np.mean((z - z.mean()) ** 2)))
    ok = vf >= min_valid and (rms is None or rms <= max_plane_rms_mm)
    reason = "ok" if ok else ("low valid pixels under cup" if vf < min_valid else "surface too rough")
    return {"ok": bool(ok), "valid_frac": round(vf, 3),
            "plane_rms_mm": None if rms is None else round(rms, 2), "reason": reason}
