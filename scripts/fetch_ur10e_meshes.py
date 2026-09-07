# -*- coding: utf-8 -*-
"""UR10e 메시(OBJ, 약 34 MB)를 MuJoCo Menagerie 에서 받아 sim/assets/ur10e/assets/ 에 둔다.

    python scripts/fetch_ur10e_meshes.py

git 의 sparse checkout 으로 universal_robots_ur10e 폴더만 받는다. 메시가 없어도 sim/arm.py 는
충돌용 캡슐을 보이게 해서 동작하므로, 그림을 예쁘게 뽑을 때만 필요하다.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "sim" / "assets" / "ur10e" / "assets"
REPO = "https://github.com/google-deepmind/mujoco_menagerie"


def main():
    if DST.exists() and any(DST.glob("*.obj")):
        print("already present:", DST)
        return 0
    tmp = Path(tempfile.mkdtemp(prefix="menagerie_"))
    try:
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(tmp / "m")],
                       check=True)
        subprocess.run(["git", "-C", str(tmp / "m"), "sparse-checkout", "set", "universal_robots_ur10e"], check=True)
        src = tmp / "m" / "universal_robots_ur10e" / "assets"
        DST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, DST, dirs_exist_ok=True)
        n = len(list(DST.glob("*.obj")))
        print(f"copied {n} meshes -> {DST}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
