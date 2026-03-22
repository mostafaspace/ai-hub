import safetensors
import json

path = r"D:\hf_models\Lightricks\LTX-2.3\ltx-2.3-22b-dev-fp8.safetensors"
with safetensors.safe_open(path, framework="pt", device="cpu") as f:
    keys = f.keys()
    voc_keys = [k for k in keys if "vocoder.vocoder" in k]
    print("VOCODER.VOCODER KEYS (first 10):")
    for k in voc_keys[:10]:
        print(f"  {k}")
