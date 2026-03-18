import requests
import time
import os

ORCHESTRATOR_URL = "http://127.0.0.1:9000"

def test_director_v2():
    print("--- Testing Production-Grade Director Workflow ---")
    payload = {
        "image_prompt": "A beautiful sunset over a cyberpunk city, high quality, neon lights",
        "voiceover_text": "In the heart of the neon metropolis, the sun sets not with a whimper, but with a vibrant explosion of color, echoing the synthetic pulse of the city.",
        "voice": "vivian",
        "music_prompt": "Cinematic cyberpunk ambient music, futuristic, slow tempo",
        "music_volume": 0.15
    }

    print(f"Triggering workflow via {ORCHESTRATOR_URL}/v1/workflows/director...")
    resp = requests.post(f"{ORCHESTRATOR_URL}/v1/workflows/director", json=payload)
    if resp.status_code != 202:
        print(f"Failed to start workflow: {resp.status_code} - {resp.text}")
        return

    task_id = resp.json().get("task_id")
    print(f"Task started: {task_id}. Polling for status...")

    start_time = time.time()
    last_status = None
    while True:
        status_resp = requests.get(f"{ORCHESTRATOR_URL}/v1/hub/tasks")
        tasks = status_resp.json().get("tasks", [])
        current_task = next((t for t in tasks if t["task_id"] == task_id), None)

        if current_task:
            status = current_task["status"]
            if status != last_status:
                print(f"[{int(time.time() - start_time)}s] Status: {status}")
                last_status = status
            
            if status == "COMPLETED":
                print("Workflow completed successfully!")
                break
            elif "FAILED" in status:
                print(f"Workflow failed: {status}")
                break
        
        if time.time() - start_time > 1800: # 30 min timeout
            print("Test timed out.")
            break
            
        time.sleep(5)

    # Verify output
    output_dir = f"d:/antigravity/orchestrator/director_outputs/{task_id}"
    final_cut = os.path.join(output_dir, "final_director_cut.mp4")
    
    if os.path.exists(final_cut):
        size_mb = os.path.getsize(final_cut) / (1024 * 1024)
        print(f"SUCCESS: Final cut found at {final_cut} ({size_mb:.2f} MB)")
    else:
        print(f"FAILURE: Final cut not found at {final_cut}")

if __name__ == "__main__":
    test_director_v2()
