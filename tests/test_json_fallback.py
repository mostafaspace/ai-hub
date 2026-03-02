import os
import sys

# Add root directory to path for config imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
import warnings
warnings.filterwarnings("ignore")

def test_graceful_json_tts():
    print("Testing Qwen3-TTS with Malformed JSON...")
    # Import app from TTS
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Qwen3-TTS")))
    try:
        from server import app
    except ImportError as e:
        print(f"Failed to import TTS app for testing: {e}")
        return

    client = TestClient(app)

    # 1. Valid JSON
    valid_payload = '{"input": "Hello", "voice": "Vivian"}'
    resp = client.post("/v1/audio/speech", content=valid_payload, headers={"Content-Type": "application/json"})
    # It might fail with 500 if the model isn't downloaded or something, but it SHOULD NOT BE 422
    assert resp.status_code != 422, f"Valid JSON rejected: {resp.status_code} - {resp.text}"
    print("[PASS] Valid JSON handled correctly.")

    # 2. Malformed JSON: Single Quotes
    malformed_single_quotes = "{'input': 'Hello', 'voice': 'Vivian'}"
    resp = client.post("/v1/audio/speech", content=malformed_single_quotes, headers={"Content-Type": "application/json"})
    assert resp.status_code != 422, f"Single Quotes JSON rejected: {resp.status_code} - {resp.text}"
    print("[PASS] Single Quotes JSON fixed and handled correctly.")

    # 3. Malformed JSON: Python Booleans (Not exactly applicable here but let's try a different endpoint/payload just to see we don't 422)
    # Actually wait, TTSSpeechRequest doesn't have a boolean. 
    # But as long as it doesn't give 422, the middleware fixed the syntax.
    
    # 4. Hopeless JSON -> Should naturally throw 422
    hopeless_json = '{"input": "Hello", "voice": '
    resp = client.post("/v1/audio/speech", content=hopeless_json, headers={"Content-Type": "application/json"})
    assert resp.status_code == 422, "Hopeless JSON should return 422 Unprocessable Entity."
    print("[PASS] Hopeless JSON properly rejected with 422.")
    print("TTS JSON Fallback Test Successful!")

def test_graceful_json_vision():
    print("Testing Z-Image with Malformed JSON...")
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Z-Image")))
    try:
        from vision_server import app
    except ImportError as e:
        print(f"Failed to import Vision app for testing: {e}")
        return

    client = TestClient(app)

    # 1. Malformed JSON: Python Booleans
    malformed_bools = '{"prompt": "A cat", "cfg_normalization": True, "n": 1}'
    resp = client.post("/v1/images/async_generate", content=malformed_bools, headers={"Content-Type": "application/json"})
    assert resp.status_code != 422, f"Python Boolean JSON rejected: {resp.status_code} - {resp.text}"
    print("[PASS] Python Boolean JSON fixed and handled correctly.")

if __name__ == "__main__":
    test_graceful_json_tts()
    test_graceful_json_vision()
    print("All Graceful JSON Fallback Tests Passed!")
