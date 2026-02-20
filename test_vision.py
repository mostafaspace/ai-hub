"""
Antigravity AI - Vision Service (GLM-Image) Test Script

Tests the GLM-Image Vision Service endpoints:
  - Health check
  - Text-to-image generation
  - Image-to-image editing

Usage:
    python test_vision.py
"""

import requests
import time
import base64
import io
import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(__file__))
import config

DEVICE_IP = config.TEST_DEVICE_IP
BASE_URL = f"http://{DEVICE_IP}:{config.VISION_PORT}"


def test_health():
    """Test the health endpoint."""
    print(f"\n[TEST] Health check at {BASE_URL}/health ...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  [OK] Status: {data['status']}, Model: {data['model']}, Loaded: {data['model_loaded']}")
            return True
        else:
            print(f"  [FAILED] Status {response.status_code}")
            return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def test_models():
    """Test the models listing endpoint."""
    print(f"\n[TEST] Model listing at {BASE_URL}/v1/models ...")
    try:
        response = requests.get(f"{BASE_URL}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            for m in models:
                print(f"  [OK] Model: {m['id']} — Capabilities: {m.get('capabilities', [])}")
            return True
        else:
            print(f"  [FAILED] Status {response.status_code}")
            return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def test_text_to_image():
    """Test text-to-image generation."""
    print(f"\n[TEST] Text-to-image generation ...")
    payload = {
        "prompt": "A futuristic city with flying cars and neon lights, highly detailed, photorealistic",
        "n": 1,
        "size": "1024x1024",
        "response_format": "url",
        "num_inference_steps": 50,
        "guidance_scale": 1.5,
    }

    try:
        start_time = time.time()
        print("  Sending generation request (GLM-Image can take several minutes)...")
        response = requests.post(f"{BASE_URL}/v1/images/generations", json=payload, timeout=600)

        if response.status_code == 200:
            duration = time.time() - start_time
            data = response.json()
            print(f"  [OK] Generation took {duration:.2f}s")

            if "data" in data and len(data["data"]) > 0:
                item = data["data"][0]
                if "url" in item:
                    print(f"  Image URL: {item['url']}")
                elif "b64_json" in item:
                    print(f"  Got base64 image ({len(item['b64_json'])} chars)")
            return True
        else:
            print(f"  [FAILED] Status {response.status_code}: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("  [TIMEOUT] Request timed out after 600s")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def test_text_to_image_b64():
    """Test text-to-image with base64 response."""
    print(f"\n[TEST] Text-to-image (base64 format) ...")
    payload = {
        "prompt": "A cute cat sitting on a windowsill, watercolor painting",
        "size": "512x512",
        "response_format": "b64_json",
        "num_inference_steps": 30,
    }

    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/v1/images/generations", json=payload, timeout=600)

        if response.status_code == 200:
            duration = time.time() - start_time
            data = response.json()
            b64 = data["data"][0].get("b64_json", "")
            if b64:
                # Verify valid image data
                img_bytes = base64.b64decode(b64)
                print(f"  [OK] Got {len(img_bytes)} bytes of image data in {duration:.2f}s")
                return True
            else:
                print("  [FAILED] No b64_json in response")
                return False
        else:
            print(f"  [FAILED] Status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def test_image_to_image():
    """Test image-to-image editing."""
    print(f"\n[TEST] Image-to-image editing ...")

    # Create a simple test image (solid color)
    try:
        from PIL import Image as PILImage
    except ImportError:
        print("  [SKIP] Pillow not installed, skipping image-to-image test")
        return None

    img = PILImage.new("RGB", (512, 512), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    try:
        start_time = time.time()
        files = {"image": ("test_input.png", buf, "image/png")}
        data = {
            "prompt": "Transform this into a beautiful sunset landscape",
            "size": "512x512",
            "response_format": "url",
            "num_inference_steps": 30,
        }

        response = requests.post(f"{BASE_URL}/v1/images/edits", files=files, data=data, timeout=600)

        if response.status_code == 200:
            duration = time.time() - start_time
            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                item = result["data"][0]
                print(f"  [OK] Image edit completed in {duration:.2f}s")
                if "url" in item:
                    print(f"  Result URL: {item['url']}")
                return True
        else:
            print(f"  [FAILED] Status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def main():
    print("=" * 60)
    print("      ANTIGRAVITY AI - Vision Service Test")
    print(f"      Server: {BASE_URL}")
    print("=" * 60)

    health_ok = test_health()
    models_ok = test_models()

    if not health_ok:
        print("\n[ABORT] Server is not reachable. Start it first:")
        print("  cd Vision-Service && python vision_server.py")
        return False

    t2i_ok = test_text_to_image()
    # Skip b64 and i2i tests if t2i fails (likely model loading issue)
    if t2i_ok:
        b64_ok = test_text_to_image_b64()
        i2i_ok = test_image_to_image()
    else:
        b64_ok = False
        i2i_ok = False

    print("\n" + "=" * 60)
    print("      Test Summary")
    print("=" * 60)
    print(f"  Health:           {'[OK]' if health_ok else '[FAILED]'}")
    print(f"  Models:           {'[OK]' if models_ok else '[FAILED]'}")
    print(f"  Text-to-Image:    {'[OK]' if t2i_ok else '[FAILED]'}")
    print(f"  T2I (base64):     {'[OK]' if b64_ok else '[FAILED/SKIPPED]'}")
    print(f"  Image-to-Image:   {'[OK]' if i2i_ok else '[FAILED/SKIPPED]'}")
    print("=" * 60)

    return health_ok and t2i_ok


if __name__ == "__main__":
    main()
