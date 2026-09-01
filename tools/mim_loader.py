# -*- coding: utf-8 -*-
"""Matrox MIL .mim (TIFF 기반) 로더 + 캡처 세션 유틸.

세션 폴더 구조:
  1_rgb.{jpg,bmp,mim}     RGB (mim은 원본 비트뎁스일 가능성)
  2_tof_X.mim  3_tof_Y.mim  4_tof_D.mim  5_tof_I.mim
    → ToF 카메라의 X/Y 좌표맵, Depth(D), Intensity(I). 640x480.
    → X,Y,D 세 채널로 내파라미터 없이 포인트클라우드 복원 가능.
"""
from pathlib import Path

import numpy as np
import tifffile


def load_mim(path):
    """MIL .mim을 numpy 배열로. TIFF 호환이라 tifffile로 읽힌다."""
    return tifffile.imread(str(path))


def load_session(session_dir):
    """세션 폴더에서 존재하는 채널을 dict로 로드."""
    session_dir = Path(session_dir)
    out = {}
    for key, name in [("rgb", "1_rgb"), ("X", "2_tof_X"), ("Y", "3_tof_Y"),
                      ("D", "4_tof_D"), ("I", "5_tof_I")]:
        p = session_dir / f"{name}.mim"
        if p.exists():
            out[key] = load_mim(p)
    return out


def tof_pointcloud(sess, invalid_sentinels=(0,)):
    """X/Y/D 채널 → (N,3) 포인트클라우드 (mm 단위 가정, 센티넬 검증 필요)."""
    X, Y, D = sess["X"], sess["Y"], sess["D"]
    mask = np.isfinite(D)
    for s in invalid_sentinels:
        mask &= D != s
    pts = np.stack([X[mask], Y[mask], D[mask]], axis=-1).astype(np.float32)
    return pts, mask


# 무효 픽셀 센티넬 (실측 확인값): X/Y=8191.75, D=16383.75, D=0
SENTINEL_XY = 8191.0
SENTINEL_D = 16383.0


def valid_mask(sess):
    """D 채널 기준 유효 픽셀 마스크 (센티넬·0 제외)."""
    D = sess["D"]
    m = (D > 0) & (D < SENTINEL_D)
    if "X" in sess:
        m &= np.abs(sess["X"]) < SENTINEL_XY
    if "Y" in sess:
        m &= np.abs(sess["Y"]) < SENTINEL_XY
    return m


def imread_kr(path, flags=None):
    """한글 경로 대응 이미지 로드 (cv2.imread는 Windows 비ASCII 경로 실패)."""
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR if flags is None else flags)


def describe(arr, name=""):
    a = np.asarray(arr)
    finite = a[np.isfinite(a)] if a.dtype.kind == "f" else a
    q = np.percentile(finite, [0, 1, 50, 99, 100]) if finite.size else [None] * 5
    return (f"{name}: shape={a.shape} dtype={a.dtype} "
            f"min={q[0]:.1f} p1={q[1]:.1f} med={q[2]:.1f} p99={q[3]:.1f} max={q[4]:.1f} "
            f"uniq~{min(finite.size, len(np.unique(finite[:100000])))}")
