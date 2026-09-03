# -*- coding: utf-8 -*-
"""ToF 프레임 컨테이너 + Matrox MIL .mim (TIFF 호환) 세션 로더.

세션 폴더 구조 (tools/mim_loader.py 와 동일):
  2_tof_X.mim  3_tof_Y.mim  4_tof_D.mim  5_tof_I.mim   (640x480 float32, mm, 카메라 좌표)
무효 픽셀 센티넬: D=16383.75 또는 0, |X|,|Y|=8191.75
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

# 무효 픽셀 센티넬 (실측 확인값). tools/mim_loader.py 와 동일 임계값.
SENTINEL_XY = 8191.0
SENTINEL_D = 16383.0
SENTINEL_XY_VALUE = 8191.75
SENTINEL_D_VALUE = 16383.75

CHANNEL_FILES = {"X": "2_tof_X.mim", "Y": "3_tof_Y.mim", "D": "4_tof_D.mim", "I": "5_tof_I.mim"}


def valid_mask(X: np.ndarray, Y: np.ndarray, D: np.ndarray) -> np.ndarray:
    """유효 픽셀 마스크: 센티넬·0·비유한값 제외 (tools/mim_loader.valid_mask 와 동일 규칙)."""
    m = np.isfinite(D) & (D > 0) & (D < SENTINEL_D)
    m &= np.isfinite(X) & (np.abs(X) < SENTINEL_XY)
    m &= np.isfinite(Y) & (np.abs(Y) < SENTINEL_XY)
    return m


@dataclass
class Frame:
    """ToF 4채널 프레임. 모든 배열은 같은 (H,W) shape, float32, mm 단위."""
    X: np.ndarray
    Y: np.ndarray
    D: np.ndarray
    I: np.ndarray
    valid: np.ndarray = field(default=None)  # bool (H,W); None 이면 자동 계산
    source: str = ""

    def __post_init__(self):
        self.X = np.ascontiguousarray(self.X, dtype=np.float32)
        self.Y = np.ascontiguousarray(self.Y, dtype=np.float32)
        self.D = np.ascontiguousarray(self.D, dtype=np.float32)
        self.I = np.ascontiguousarray(self.I, dtype=np.float32)
        shapes = {self.X.shape, self.Y.shape, self.D.shape, self.I.shape}
        if len(shapes) != 1 or self.D.ndim != 2:
            raise ValueError(f"channel shapes differ or not 2-D: {shapes}")
        if self.valid is None:
            self.valid = valid_mask(self.X, self.Y, self.D)
        else:
            self.valid = np.asarray(self.valid, dtype=bool)
            if self.valid.shape != self.D.shape:
                raise ValueError("valid mask shape mismatch")

    @property
    def shape(self):
        return self.D.shape

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())

    @classmethod
    def from_session(cls, sess: Mapping[str, np.ndarray], source: str = "") -> "Frame":
        """tools.mim_loader.load_session() 결과(dict) → Frame."""
        missing = [k for k in ("X", "Y", "D", "I") if k not in sess]
        if missing:
            raise KeyError(f"session dict missing channels: {missing}")
        return cls(sess["X"], sess["Y"], sess["D"], sess["I"], source=source)

    def as_session(self) -> dict:
        """tools/ 스크립트 호환 dict (detect_boxes(sess) 등에 그대로 전달 가능)."""
        return {"X": self.X, "Y": self.Y, "D": self.D, "I": self.I}


def _read_mim(path: Path) -> np.ndarray:
    import tifffile  # 지연 임포트: 합성 프레임만 쓰는 경우 불필요
    return np.asarray(tifffile.imread(str(path)))


def load_frame(session_dir, require_all: bool = True) -> Frame:
    """세션 폴더에서 X/Y/D/I .mim 4채널을 읽어 Frame 으로 반환.

    require_all=True 이면 채널 하나라도 없을 때 FileNotFoundError.
    """
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        raise FileNotFoundError(f"session dir not found: {session_dir}")
    arrays: dict[str, Optional[np.ndarray]] = {}
    missing = []
    for key, name in CHANNEL_FILES.items():
        p = session_dir / name
        if p.exists():
            arrays[key] = _read_mim(p)
        else:
            missing.append(name)
    if missing and require_all:
        raise FileNotFoundError(f"missing ToF channels in {session_dir}: {missing}")
    if "D" not in arrays:
        raise FileNotFoundError(f"depth channel {CHANNEL_FILES['D']} required in {session_dir}")
    D = arrays["D"]
    for key in ("X", "Y", "I"):
        if key not in arrays:  # require_all=False 일 때 빈 채널 채움 (X/Y 없으면 mm 치수 불가)
            arrays[key] = np.zeros_like(D)
    return Frame(arrays["X"], arrays["Y"], D, arrays["I"], source=str(session_dir))
