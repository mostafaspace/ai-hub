
import time
import sys

def test_import(module_name):
    print(f"Importing {module_name}...", flush=True)
    start = time.time()
    try:
        __import__(module_name)
        print(f"  Success in {time.time() - start:.2f}s", flush=True)
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)

test_import("json")
test_import("os")
test_import("torch")
test_import("huggingface_hub")
test_import("librosa")
test_import("transformers")
test_import("transformers.integrations")

print("Importing specific components from transformers...", flush=True)
try:
    from transformers.integrations import use_kernel_forward_from_hub
    print("  use_kernel_forward_from_hub SUCCESS", flush=True)
except ImportError:
    print("  use_kernel_forward_from_hub MISSING", flush=True)

try:
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
    print("  FlashAttentionKwargs SUCCESS", flush=True)
except ImportError:
    print("  FlashAttentionKwargs MISSING", flush=True)
