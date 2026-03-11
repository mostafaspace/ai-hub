import requests
import json
import time

def check_health():
    print("Checking Orchestrator Health...")
    try:
        r = requests.get("http://127.0.0.1:9000/health", timeout=5)
        print(f"Orchestrator: {r.status_code} - {r.json()}")
    except Exception as e:
        print(f"Orchestrator Health Check FAILED: {e}")

    print("\nChecking Backend Services via Hub...")
    try:
        r = requests.get("http://127.0.0.1:9000/v1/hub/services", timeout=10)
        print(f"Services: {json.dumps(r.json(), indent=2)}")
    except Exception as e:
        print(f"Services Health Check FAILED: {e}")

    print("\nTesting Director Workflow...")
    try:
        payload = {
            "image_prompt": "A futuristic city in the clouds, sunset, highly detailed.",
            "voiceover_text": "Welcome to the future of AI. The sky is no longer the limit."
        }
        print(f"Sending request to /v1/workflows/director...")
        r = requests.post("http://127.0.0.1:9000/v1/workflows/director", json=payload, timeout=1800)
        if r.status_code == 200:
            data = r.json()
            print(f"Director SUCCESS: {data}")
        else:
            print(f"Director FAILED: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"Director EXCEPTION: {e}")

if __name__ == "__main__":
    check_health()
