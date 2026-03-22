import requests
import time
import sys

URL_BASE = "http://127.0.0.1:8004"

# Wait for server to be up
print("Waiting for server to start...")
for _ in range(60):
    try:
        res = requests.get(f"{URL_BASE}/health")
        if res.status_code == 200:
            print("Server is up!")
            break
    except requests.exceptions.ConnectionError:
        time.sleep(2)
else:
    print("Server failed to start in time.")
    sys.exit(1)

# Start generation using our optimized defaults explicitly!
payload = {
    "prompt": "3D animation, a futuristic cute robot dancing in a neon-lit cyber city, highly detailed, masterpiece, vibrant colors, dynamic neon lighting",
    "num_inference_steps": 11,
    "cfg_scale_video": 3.5,
    "num_frames": 81,
    "seed": 42
}

print(f"Sending generation request: {payload['prompt']}")
res = requests.post(f"{URL_BASE}/v1/video/async_t2v", json=payload)
if res.status_code != 200:
    print(f"Failed to submit: {res.text}")
    sys.exit(1)

task_id = res.json().get("task_id")
print(f"Task submitted! ID: {task_id}")

while True:
    try:
        task_res = requests.get(f"{URL_BASE}/v1/video/tasks/{task_id}")
        data = task_res.json()
        status = data.get("status")
        if status == "completed":
            print(f"\nGeneration completed! URL: {data.get('url')}")
            break
        elif status == "failed":
            print(f"\nGeneration failed: {data.get('error')}")
            sys.exit(1)
        else:
            print(".", end="", flush=True)
            time.sleep(5)
    except Exception as e:
        print(f"Error polling: {e}")
        time.sleep(5)
