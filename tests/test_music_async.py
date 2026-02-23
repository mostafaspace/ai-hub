import requests
import time
import json
import sys

# Testing the new OpenAI-compatible music generation endpoints

BASE_URL = "http://127.0.0.1:8001"

print("1. Submitting asynchronous music generation task...")
payload = {
    "prompt": "An upbeat lo-fi hip hop track with a chill vibe, perfect for studying.",
    "lyrics": "",
    "audio_duration": 15,
    "thinking": True,
}

response = requests.post(f"{BASE_URL}/v1/audio/async_generations", json=payload)
print(f"Status Code: {response.status_code}")
try:
    data = response.json()
    print("Response:", json.dumps(data, indent=2))
except Exception as e:
    print("Failed to decode JSON:", response.text)
    sys.exit(1)

if response.status_code != 200 or "task_id" not in data:
    print("Failed to start task or missing task_id.")
    sys.exit(1)

task_id = data["task_id"]
print(f"\n2. Task started with ID: {task_id}")
print("Polling for completion (this may take 1-2 minutes)...")

max_polls = 100
poll_interval = 4

for i in range(max_polls):
    poll_response = requests.get(f"{BASE_URL}/v1/audio/tasks/{task_id}")
    if poll_response.status_code != 200:
        print(f"Error polling: HTTP {poll_response.status_code} - {poll_response.text}")
        time.sleep(poll_interval)
        continue
        
    try:
        poll_data = poll_response.json()
    except Exception as e:
        print("Failed to decode polling JSON:", poll_response.text)
        time.sleep(poll_interval)
        continue

    status = poll_data.get("status")
    print(f"Poll {i+1}/{max_polls} - Status: {status}")

    if status == "completed":
        print("\nSUCCESS! Generation complete.")
        print("Result Data:", json.dumps(poll_data, indent=2))
        url = poll_data["data"][0]["url"]
        
        # Download the audio file
        print(f"\n3. Downloading generated audio from: {url}")
        audio_response = requests.get(url)
        if audio_response.status_code == 200:
            with open("test_music_output.mp3", "wb") as f:
                f.write(audio_response.content)
            print("Successfully saved to test_music_output.mp3!")
            sys.exit(0)
        else:
            print(f"Failed to download audio. Status: {audio_response.status_code}")
            sys.exit(1)
            
    elif status == "failed":
        print("\nTask failed!")
        print("Error:", poll_data.get("error", "Unknown error"))
        sys.exit(1)

    time.sleep(poll_interval)

print("\nTimeout reached. Task did not complete in time.")
sys.exit(1)
