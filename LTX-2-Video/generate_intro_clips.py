import requests
import time
import sys
import os

URL_BASE = "http://127.0.0.1:8004"

def submit_task(prompt, task_id_label):
    payload = {
        "prompt": prompt,
        "num_inference_steps": 80, # Ultimate quality
        "height": 576,
        "width": 1024,
        "cfg_scale_video": 3.0,
        "num_frames": 121, # 5 seconds
        "seed": 42
    }
    print(f"[{task_id_label}] Submitting: {prompt[:50]}...")
    res = requests.post(f"{URL_BASE}/v1/video/async_t2v", json=payload)
    if res.status_code != 200:
        print(f"[{task_id_label}] Failed: {res.text}")
        return None
    return res.json().get("task_id")

prompts = [
    "Wide cinematic shot of a rain-slicked cyberpunk megacity at night, massive neon holographic advertisements, volumetric fog, extreme detail, 8k resolution, sharp cinematography",
    "Extreme close-up of a glowing cybernetic eye, neon lights flashing in the reflection of the pupil, intricate mechanical iris, ultra high sharpness, macro lens",
    "Motion-blurred silhouette of a sleek futuristic motorcycle speeding through a glowing technological tunnel, streaks of incandescent orange light, dynamic low angle camera, pin sharp focus",
    "A hand reaching out into a digital void, disintegrating into glowing cyan data particles, cinematic depth of field, sharp focus on the fingertips, noir atmosphere"
]

task_ids = []
for i, p in enumerate(prompts):
    tid = submit_task(p, f"Scene {i+1}")
    if tid:
        task_ids.append((tid, f"Scene {i+1}"))

results = {}
while len(results) < len(task_ids):
    for tid, label in task_ids:
        if tid in results:
            continue
        try:
            res = requests.get(f"{URL_BASE}/v1/video/tasks/{tid}")
            data = res.json()
            status = data.get("status")
            if status == "completed":
                print(f"\n[{label}] Completed! URL: {data.get('url')}")
                results[tid] = data.get("url")
            elif status == "failed":
                print(f"\n[{label}] Failed: {data.get('error')}")
                results[tid] = "FAILED"
            else:
                pass
        except Exception as e:
            print(f"Error polling {label}: {e}")
    
    print(f"\rProgress: {len(results)}/{len(task_ids)} completed...", end="", flush=True)
    time.sleep(15)

print("\nAll tasks finished.")
with open("intro_results.txt", "w") as f:
    for tid, label in task_ids:
        url = results.get(tid, "N/A")
        f.write(f"{label}: {url}\n")
