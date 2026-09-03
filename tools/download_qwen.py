# -*- coding: utf-8 -*-
"""Qwen2.5-VL-3B-Instruct 로컬 다운로드 (VLM 에이전트 데모용)."""
from huggingface_hub import snapshot_download

p = snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct",
                      local_dir=r"E:\Robot_Sim\models\qwen25vl3b")
print("DOWNLOAD DONE:", p)
