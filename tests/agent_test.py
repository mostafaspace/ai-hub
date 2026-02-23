import requests
import time
import json
import base64
import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Configuration based on SKILL.md files
DEVICE_IP = "127.0.0.1" # Testing locally, but using standard IPs in actual deployment
TTS_URL = f"http://{DEVICE_IP}:8000"
MUSIC_URL = f"http://{DEVICE_IP}:8001"
ASR_URL = f"http://{DEVICE_IP}:8002"
VISION_URL = f"http://{DEVICE_IP}:8003"

def print_header(title):
    print(f"\n{'='*50}\n> AGENT TASK: {title}\n{'='*50}")

def simulate_tts_agent():
    print_header("Generate Speech (Qwen3 TTS)")
    print("Agent is constructing TTS payload...")
    payload = {
        "input": "Hello! I am an AI agent calling the Text to Speech API to generate this voice.",
        "voice": "Vivian",
        "response_format": "wav"
    }
    
    try:
        print(f"Agent Action: POST {TTS_URL}/v1/audio/speech")
        response = requests.post(f"{TTS_URL}/v1/audio/speech", json=payload, timeout=600)
        
        if response.status_code == 200:
            filename = "agent_output_speech.wav"
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"Success! Agent downloaded audio to {filename}")
            return filename
        else:
            print(f"Agent encountered error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Agent failed to reach TTS API: {e}")
        return None

def simulate_asr_agent(audio_path):
    print_header("Transcribe Audio (Qwen3 ASR)")
    if not os.path.exists(audio_path):
        print("Agent could not find audio file to transcribe.")
        return
        
    print(f"Agent is uploading {audio_path} for transcription...")
    
    try:
        print(f"Agent Action: POST {ASR_URL}/v1/audio/transcriptions")
        with open(audio_path, "rb") as f:
            files = {"file": f}
            data = {"prompt": "Transcribe accurately."}
            response = requests.post(f"{ASR_URL}/v1/audio/transcriptions", files=files, data=data, timeout=600)
            
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Agent received transcription:")
            print(f"  \"{result.get('text')}\"")
        else:
            print(f"Agent encountered error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Agent failed to reach ASR API: {e}")

def simulate_music_agent():
    print_header("Generate Music (ACE-Step)")
    print("Agent is constructing Music generation payload...")
    
    payload = {
        "prompt": "An upbeat electronic track introducing an AI agent, energetic and clean.",
        "audio_duration": 15,
        "thinking": True,
    }
    
    try:
        print(f"Agent Action: POST {MUSIC_URL}/release_task")
        response = requests.post(f"{MUSIC_URL}/release_task", json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            # Standardizing task_id extraction as API might wrap it natively
            task_id = data.get("data", {}).get("task_id") or data.get("task_id")
            print(f"Task released. Agent received Task ID: {task_id}")
            
            # Polling Loop
            print("Agent is now entering polling loop...")
            poll_payload = {"task_id_list": [task_id]}
            
            for _ in range(60): # 3 minutes max
                time.sleep(3)
                poll_res = requests.post(f"{MUSIC_URL}/query_result", json=poll_payload, timeout=10)
                if poll_res.status_code == 200:
                    poll_data = poll_res.json()
                    status_obj = poll_data.get("data", [{}])[0]
                    status = status_obj.get("status")
                    
                    if status == 1:
                        print("Agent detected completed task!")
                        result_str = status_obj.get("result", "[]")
                        try:
                            result_json = json.loads(result_str)
                            file_path = result_json[0].get("file")
                            print(f"Agent fetched generated file path: {file_path}")
                            print(f"Agent would now GET: {MUSIC_URL}{file_path} to download.")
                            break
                        except:
                            print("Agent failed to parse result JSON string.")
                            break
                    elif status == 2:
                        print("Agent detected Task FAILED.")
                        break
                    else:
                        print("  Agent checking status... still Processing (0).")
                else:
                    print(f"Polling error: {poll_res.status_code}")
                    break
        else:
            print(f"Agent encountered error: {response.status_code} - {response.text}")
    except Exception as e:
         print(f"Agent failed to reach Music API: {e}")

def simulate_vision_agent():
    print("\n" + "="*50)
    print("> AGENT TASK: Generate Image (Z-Image Vision)")
    print("="*50)
    
    payload = {
        "prompt": "A serene futuristic city at night, photorealistic"
    }
    
    try:
        print("Agent is constructing Z-Image Vision async payload...")
        print(f"Agent Action: POST {VISION_URL}/v1/images/async_generate")
        response = requests.post(f"{VISION_URL}/v1/images/async_generate", json=payload, timeout=5)
        
        if response.status_code == 200:
            task_data = response.json()
            task_id = task_data.get("task_id")
            print(f"Agent received async Task ID: {task_id}")
            
            print("Agent is now entering polling loop for Vision task...")
            for _ in range(60): # 60 * 5s = 5 minutes max
                status_res = requests.get(f"{VISION_URL}/v1/images/tasks/{task_id}", timeout=5)
                if status_res.status_code == 200:
                    status_data = status_res.json()
                    if status_data.get("status") == "completed":
                        print("Agent detected completed Vision task!")
                        url = status_data["data"][0]["url"]
                        print(f"Success! Agent received image URL: {url}")
                        return url
                    elif status_data.get("status") == "failed":
                        print(f"Agent detected Vision task failed: {status_data.get('error')}")
                        return False
                    else:
                        print(f"  Agent checking status... still {status_data.get('status')}.")
                time.sleep(5)
            print("Agent timeout waiting for Vision image.")
        else:
            print(f"Agent encountered error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Agent failed to reach Vision API: {e}")
    return False

def simulate_vision_edit_agent(source_url: str):
    print("\n" + "="*50)
    print("> AGENT TASK: Edit Image (Z-Image-Edit)")
    print("="*50)
    
    try:
        print(f"Agent downloading source image from loop 1: {source_url}...")
        img_response = requests.get(source_url)
        img_response.raise_for_status()
        
        # Save temp file for multipart upload
        temp_img_path = "agent_temp_edit_source.png"
        with open(temp_img_path, "wb") as f:
            f.write(img_response.content)
            
        print("Agent is constructing Z-Image-Edit multipart payload...")
        print(f"Agent Action: POST {VISION_URL}/v1/images/async_edit")
        
        with open(temp_img_path, "rb") as img_file:
            files = {"image": ("source.png", img_file, "image/png")}
            data = {"prompt": "Make it raining heavily"}
            
            response = requests.post(
                f"{VISION_URL}/v1/images/async_edit", 
                files=files, 
                data=data, 
                timeout=10
            )
            
        if os.path.exists(temp_img_path):
             os.remove(temp_img_path)
        
        if response.status_code == 200:
            task_data = response.json()
            task_id = task_data.get("task_id")
            print(f"Agent received async Edit Task ID: {task_id}")
            
            print("Agent is now entering polling loop for Vision Edit task...")
            for _ in range(60): # 60 * 5s = 5 minutes max
                status_res = requests.get(f"{VISION_URL}/v1/images/tasks/{task_id}", timeout=5)
                if status_res.status_code == 200:
                    status_data = status_res.json()
                    if status_data.get("status") == "completed":
                        print("Agent detected completed Vision Edit task!")
                        url = status_data["data"][0]["url"]
                        print(f"Success! Agent received edited image URL: {url}")
                        return True
                    elif status_data.get("status") == "failed":
                        print(f"Agent detected Vision Edit task failed: {status_data.get('error')}")
                        return False
                    else:
                        print(f"  Agent checking status... still {status_data.get('status')}.")
                time.sleep(5)
            print("Agent timeout waiting for Vision Edit image.")
        else:
            print(f"Agent encountered error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Agent failed to reach Vision Edit API: {e}")
    return False

def run_agent_workflow():
    print("==================================================")
    print("🤖 INITIATING OPENCLAW AGENT API CONSUMPTION TEST")
    print("==================================================")
    
    # Check health simply
    print("Agent performing Pre-flight checks...")
    for name, url in [("TTS", TTS_URL), ("Music", MUSIC_URL), ("ASR", ASR_URL), ("Vision", VISION_URL)]:
        try:
            res = requests.get(f"{url}/health", timeout=5)
            status = "Ready" if res.status_code == 200 else f"Error {res.status_code}"
            print(f"  - {name} ({url}/health): {status}")
        except:
             print(f"  - {name} ({url}/health): UNREACHABLE")

    print("\nStarting OpenClaw Action Pipelines...")
    # NOTE: Run sequentially to simulate a single agent's execution path
    # and prevent VRAM OOM on the local machine testing all 4 simultaneously.
    
    # 1. Agent generates speech
    generated_audio = simulate_tts_agent()
    
    # 2. Agent immediately transcribes the speech it just generated
    if generated_audio:
        simulate_asr_agent(generated_audio)
        
    # 3. Agent generates a background track
    simulate_music_agent()
    
    # 4. Agent generates an avatar
    vision_url = simulate_vision_agent()
    
    # 5. Agent edits the avatar
    if vision_url:
        simulate_vision_edit_agent(vision_url)

    print("\n==================================================")
    print("✅ AGENT API CONSUMPTION TEST COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    run_agent_workflow()
