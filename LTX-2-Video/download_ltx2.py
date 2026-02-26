import os
import sys
import traceback
from huggingface_hub import hf_hub_download

repo_id = "Lightricks/LTX-2"
local_dir = r"D:\hf_models\Lightricks\LTX-2"
os.makedirs(local_dir, exist_ok=True)

files_to_download = [
    "ltx-2-19b-dev-fp8.safetensors",
    "ltx-2-19b-distilled-lora-384.safetensors",
    "ltx-2-spatial-upscaler-x2-1.0.safetensors"
]

for f in files_to_download:
    print(f"Downloading {f}...")
    try:
        path = hf_hub_download(repo_id=repo_id, filename=f, local_dir=local_dir)
        print(f"Successfully downloaded to: {path}")
    except Exception as e:
        print(f"Failed to download {f}:")
        traceback.print_exc()
        sys.exit(1)

print("All files downloaded successfully.")
