# -*- coding: utf-8 -*-
"""pytest 설정: 레포 루트를 sys.path 에 추가해 설치 없이 robotsim_perception 을 임포트."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
