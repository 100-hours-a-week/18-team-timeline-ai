import os
from huggingface_hub import snapshot_download

MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
MODEL_NAME = MODEL_ID.split("/")[-1]
print(1)
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=MODEL_NAME,
    local_dir_use_symlinks=False,
    revision="main",
)
