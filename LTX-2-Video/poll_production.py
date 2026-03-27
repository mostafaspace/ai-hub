import requests
import time
import os

URL_BASE = "http://127.0.0.1:8004"

# New IDs from server_intro_v4.log
task_ids = [
    ("3b47ca050d8a42758a67655cba35d8b5", "Scene 1"),
    ("a82178ec81894878a3538af36930ff4e", "Scene 2"),
    ("7aca06626941499b8c452f12946af2d3", "Scene 3"),
    ("62f07a69951b4843b628771d5550a33c", "Scene 4")
]

results = {}
print("Starting production monitor for 'Obsidian Veil'...")

while len(results) < len(task_ids):
    for tid, label in task_ids:
        if tid in results:
            continue
        try:
            res = requests.get(f"{URL_BASE}/v1/video/tasks/{tid}")
            if res.status_code == 200:
                data = res.json()
                status = data.get("status")
                if status == "completed":
                    print(f"\n[{label}] Completed! URL: {data.get('url')}")
                    results[tid] = data.get("url")
                elif status == "failed":
                    print(f"\n[{label}] Failed: {data.get('error')}")
                    results[tid] = "FAILED"
        except Exception:
            pass
    
    print(f"\rProduction Progress: {len(results)}/{len(task_ids)} renders complete...", end="", flush=True)
    time.sleep(60)

print("\nFinal Cut ready for assembly.")
with open("production_results.txt", "w") as f:
    for tid, label in task_ids:
        url = results.get(tid, "N/A")
        f.write(f"{label}: {url}\n")
