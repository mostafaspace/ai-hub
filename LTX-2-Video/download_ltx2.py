import os
import sys
import traceback
from huggingface_hub import hf_hub_download

repo_configs = [
    {
        "repo_id": "Lightricks/LTX-2.3-fp8",
        "local_dir": r"D:\hf_models\Lightricks\LTX-2.3",
        "files": ["ltx-2.3-22b-dev-fp8.safetensors"]
    },
    {
        "repo_id": "Lightricks/LTX-2.3",
        "local_dir": r"D:\hf_models\Lightricks\LTX-2.3",
        "files": [
            "ltx-2.3-22b-distilled-lora-384.safetensors",
            "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        ]
    }
]

for config in repo_configs:
    repo_id = config["repo_id"]
    local_dir = config["local_dir"]
    os.makedirs(local_dir, exist_ok=True)
    
    for f in config["files"]:
        print(f"Downloading {f} from {repo_id}...")
        try:
            path = hf_hub_download(repo_id=repo_id, filename=f, local_dir=local_dir)
            print(f"Successfully downloaded to: {path}")
        except Exception as e:
            print(f"Failed to download {f} from {repo_id}:")
            traceback.print_exc()
            sys.exit(1)

print("All files downloaded successfully.")
