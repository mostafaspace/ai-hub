"""
LTX-2 Image-to-Video Test Client
Sends a test image to /v1/video/i2v and saves the generated video.
"""
import time
import sys
import requests
import shutil

PORT = 8004
URL = f"http://127.0.0.1:{PORT}/v1/video/i2v"
IMAGE_PATH = r"C:\Users\mmagd\.gemini\antigravity\brain\b6c84422-7776-463d-a62d-8ee0c8d90451\i2v_test_image_1772055066745.png"

PROMPT = (
    "A serene coastal beach scene. The camera slowly pushes in from a wide establishing shot "
    "towards the wooden pier extending into the ocean. Gentle waves roll in rhythmically, wet sand "
    "reflects the warm golden light of the setting sun. The sky transitions from deep orange near the "
    "horizon to soft pink and purple above. A light ocean breeze creates subtle movement in the water surface. "
    "Cinematic, shallow depth of field, handheld camera feel."
)

print(f"Sending I2V request to {URL}...")
print(f"Image: {IMAGE_PATH}")
print(f"Prompt: {PROMPT[:80]}...")
start = time.time()

try:
    with open(IMAGE_PATH, "rb") as img_f:
        response = requests.post(
            URL,
            data={
                "prompt": PROMPT,
                "height": 512,
                "width": 768,
                "num_frames": 49,
                "frame_rate": 24.0,
                "num_inference_steps": 20,
                "seed": 42,
            },
            files={"image": ("test_image.png", img_f, "image/png")},
            stream=True,
        )

    if response.status_code != 200:
        print(f"[ERROR] Server returned error {response.status_code}: {response.text}")
        sys.exit(1)

    output_filename = "test_ltx2_i2v_output.mp4"
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
    raise
