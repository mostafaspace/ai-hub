import time
import requests
import sys

PORT = 8004
URL = f"http://127.0.0.1:{PORT}/v1/video/t2v"

payload = {
    "prompt": "A beautiful sunset over the digital ocean, vivid colors, sweeping camera shot.",
    "height": 512,
    "width": 768,
    "num_frames": 49,  # Shorter clip for testing
    "frame_rate": 24.0,
    "num_inference_steps": 20, # Reduced steps for speed
    "seed": 42
}

print(f"Sending T2V request to {URL}...")
print(f"Prompt: {payload['prompt']}")
start = time.time()

try:
    # We use stream=True since the server returns a StreamingResponse
    response = requests.post(URL, json=payload, stream=True)
    
    if response.status_code != 200:
        print(f"[ERROR] Server returned error {response.status_code}: {response.text}")
        sys.exit(1)
        
    output_filename = "test_ltx2_output.mp4"
    with open(output_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
            
    elapsed = time.time() - start
    print(f"[SUCCESS] Video saved to {output_filename}")
    print(f"Time taken: {elapsed:.2f} seconds")
except requests.exceptions.ConnectionError:
    print(f"[ERROR] Failed to connect to port {PORT}. Is the LTX-2 Server running?")
except Exception as e:
    print(f"[ERROR] An error occurred: {e}")
