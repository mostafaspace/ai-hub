import requests
import time
import os

URL_BASE = "http://127.0.0.1:8004"

task_ids = [
    ("e5fd02ca5d15406bb93d6f9a5bdb6d3b", "Scene 1"),
    ("b429b386e9b74a288080f82d34230cdf", "Scene 2"),
    ("23f54cf7ac25489995dcfb0e9d040267", "Scene 3"),
    ("2f389f15db61488abcac62ff2eb92915", "Scene 4")
]

results = {}
while len(results) < len(task_ids):
    all_down = True
    for tid, label in task_ids:
        if tid in results:
            continue
        try:
            res = requests.get(f"{URL_BASE}/v1/video/tasks/{tid}")
            if res.status_code == 200:
                all_down = False
                data = res.json()
                status = data.get("status")
                if status == "completed":
                    print(f"\n[{label}] Completed! URL: {data.get('url')}")
                    results[tid] = data.get("url")
                elif status == "failed":
                    print(f"\n[{label}] Failed: {data.get('error')}")
                    results[tid] = "FAILED"
            else:
                print(f"\n[{label}] Server error: {res.status_code}")
        except Exception as e:
            # print(f"Error polling {label}: {e}")
            pass
    
    if all_down:
        print("\r[WARNING] Server unreachable. Still waiting...", end="")
    else:
        print(f"\rProgress: {len(results)}/{len(task_ids)} completed...", end="", flush=True)
    
    time.sleep(30)

print("\nAll tasks finished.")
with open("intro_results.txt", "w") as f:
    for tid, label in task_ids:
        url = results.get(tid, "N/A")
        f.write(f"{label}: {url}\n")
