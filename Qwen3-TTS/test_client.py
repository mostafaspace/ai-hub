import requests
import time
import os

BASE_URL = "http://127.0.0.1:8000"

def save_audio(response, filename):
    if response.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"Saved: {filename}")
    else:
        print(f"Error {response.status_code}: {response.text}")

def test_custom_voice():
    print("\n--- Testing CustomVoice (Standard TTS) ---")
    url = f"{BASE_URL}/v1/audio/speech"
    payload = {
        "input": "This is a test of the Qwen3 Custom Voice model. It should sound natural.",
        "voice": "Vivian",
        "language": "Auto"
    }
    start = time.time()
    try:
        response = requests.post(url, json=payload, stream=True)
        print(f"Time: {time.time() - start:.2f}s")
        save_audio(response, "output_custom.mp3")
    except Exception as e:
        print(f"Failed: {e}")

def test_voice_design():
    print("\n--- Testing Voice Design (Prompted Voice) ---")
    url = f"{BASE_URL}/v1/audio/voice_design"
    payload = {
        "input": "I am a designed voice. I was created from a text description!",
        "instruct": "A deep, robotic male voice with a slow and echoing tone.",
        "language": "English"
    }
    start = time.time()
    try:
        response = requests.post(url, json=payload, stream=True)
        print(f"Time: {time.time() - start:.2f}s")
        save_audio(response, "output_design.mp3")
    except Exception as e:
        print(f"Failed: {e}")

def test_voice_clone():
    print("\n--- Testing Voice Cloning ---")
    url = f"{BASE_URL}/v1/audio/voice_clone"
    
    # We need a reference file. Let's use the custom voice output if it exists, or dummy.
    ref_file = "output_custom.mp3"
    if not os.path.exists(ref_file):
        print("Skipping clone test: No reference audio found (run custom voice test first).")
        return

    data = {
        "text": "This voice is cloned from the previous sample. How does it match?",
        "ref_text": "This is a test of the Qwen3 Custom Voice model. It should sound natural.", # Matches content of ref_file usually improves quality
        "language": "English"
    }
    
    files = {
        "ref_audio": (ref_file, open(ref_file, "rb"), "audio/wav")
    }
    
    start = time.time()
    try:
        response = requests.post(url, data=data, files=files, stream=True)
        print(f"Time: {time.time() - start:.2f}s")
        save_audio(response, "output_clone.mp3")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    # Ensure server is up
    try:
        requests.get(f"{BASE_URL}/health")
    except:
        print("Server not running. Start it with 'python server.py'")
        exit(1)

    test_custom_voice()
    test_voice_design()
    test_voice_clone() 
