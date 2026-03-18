
import requests
import time
import os
import sys

# Local imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
try:
    import config
    BASE_URL = f"http://localhost:{config.ORCHESTRATOR_PORT}"
except:
    BASE_URL = "http://localhost:9000"

def poll_task(task_id, endpoint="/v1/hub/tasks/"):
    url = f"{BASE_URL}{endpoint}{task_id}"
    print(f"Polling {url}...")
    while True:
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"Error polling: {resp.text}")
            return None
        
        data = resp.json()
        status = data.get("status")
        print(f"  Status: {status}")
        
        if status == "COMPLETED":
            return data
        if "FAILED" in status:
            print(f"Task failed: {data}")
            return data
        
        time.sleep(5)

def test_reflexive_loop():
    print("=== Testing Reflexive Loop (Director -> Audit) ===")
    
    # 1. Trigger Director
    director_url = f"{BASE_URL}/v1/workflows/director"
    payload = {
        "image_prompt": "A futuristic city with flying cars, neon lights, 8k, cinematic",
        "voiceover_text": "Welcome to the city of the future. Where technology and humanity coexist in harmony.",
        "voice": "vivian"
    }
    
    print(f"Triggering Director: {director_url}")
    resp = requests.post(director_url, json=payload)
    if resp.status_code != 202:
        print(f"Director failed to start: {resp.text}")
        return
    
    director_task_id = resp.json().get("task_id")
    print(f"Director Task ID: {director_task_id}")
    
    # 2. Wait for Director
    director_result = poll_task(director_task_id)
    if not director_result or director_result["status"] != "COMPLETED":
        print("Director workflow did not complete successfully.")
        return
    
    video_url = director_result.get("output_url")
    if video_url.startswith("/"):
        video_url = f"{BASE_URL}{video_url}"
    print(f"Director Video URL: {video_url}")
    
    # 3. Trigger Audit
    audit_url = f"{BASE_URL}/v1/workflows/audit"
    audit_payload = {
        "media_url": video_url,
        "media_type": "video"
    }
    
    print(f"Triggering Audit: {audit_url}")
    resp = requests.post(audit_url, json=audit_payload)
    if resp.status_code != 202:
        print(f"Audit failed to start: {resp.text}")
        return
    
    audit_task_id = resp.json().get("task_id")
    print(f"Audit Task ID: {audit_task_id}")
    
    # 4. Wait for Audit
    audit_result = poll_task(audit_task_id)
    if not audit_result or audit_result["status"] != "COMPLETED":
        print("Audit workflow did not complete successfully.")
        return
    
    print("=== Reflexive Loop Test PASSED ===")
    print(f"Audit Result: {audit_result.get('analysis')}")

if __name__ == "__main__":
    # Note: This requires the servers to be running.
    # Since I cannot run the servers persistent in the background with run_command easily 
    # and expect them to stay alive for long tasks, I will perform a DRY RUN or check
    # if the orchestrator server code handles the logic correctly via unit tests.
    
    # Actually, I already ran unit tests for Audit.
    # I should try to run the orchestrator in the background and test it.
    
    test_reflexive_loop()
