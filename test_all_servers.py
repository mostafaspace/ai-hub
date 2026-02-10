"""
Antigravity AI - Unified Server Test Script

Tests both Qwen3 TTS and ACE-Step Music servers to verify they're working.

Usage:
    python test_all_servers.py
"""

import requests
import sys
import time

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# IP Configuration - Use device IP instead of localhost
DEVICE_IP = "192.168.1.26"

SERVERS = [
    {"name": "Qwen3 TTS", "port": 8000, "health": "/health", "url": f"http://{DEVICE_IP}:8000"},
    {"name": "ACE-Step Music", "port": 8001, "health": "/health", "url": f"http://{DEVICE_IP}:8001"},
]


def check_health(server):
    """Check if a server is healthy."""
    try:
        url = f"{server['url']}{server['health']}"
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except:
        return False


def test_tts_server():
    """Test the TTS server with a simple request."""
    print(f"\n[TEST] Testing Qwen3 TTS Server at {DEVICE_IP}...")
    try:
        response = requests.post(
            f"http://{DEVICE_IP}:8000/v1/audio/speech",
            json={
                "input": "Hello, this is a test.",
                "voice": "Vivian",
                "response_format": "mp3"
            },
            timeout=60
        )
        if response.status_code == 200:
            print("  [OK] TTS generation successful!")
            filename = "test_tts_output.mp3"
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"  [SAVED] Audio saved to: {filename} ({len(response.content)} bytes)")
            return True
        else:
            print(f"  [ERROR] TTS failed: {response.status_code}")
            print(f"  Response Content: {response.text}")
            return False
    except Exception as e:
        print(f"  [ERROR] TTS test failed: {e}")
        return False


def test_acestep_server():
    """Test ACE-Step with a quick generation and download the result."""
    print(f"\n[TEST] Testing ACE-Step Music Server at {DEVICE_IP}...")
    try:
        # 1. Create a short task
        response = requests.post(
            f"http://{DEVICE_IP}:8001/release_task",
            json={
                "prompt": "simple piano melody",
                "audio_duration": 10,
                "thinking": False,  # Faster for testing
                "batch_size": 1,
                "inference_steps": 4
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            task_id = data.get("data", {}).get("task_id")
            if task_id:
                print(f"  [OK] Task created: {task_id}")
                
                # 2. Poll for the result
                print("  [POLLING] Waiting for music generation to complete...")
                max_retries = 20
                for i in range(max_retries):
                    status_response = requests.post(
                        f"http://{DEVICE_IP}:8001/query_result",
                        json={"task_id_list": [task_id]},
                        timeout=10
                    )
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        task_info = status_data.get("data", [{}])[0]
                        status_code = task_info.get("status")
                        
                        if status_code == 1:  # Success
                            import json
                            results = json.loads(task_info.get("result", "[]"))
                            if results:
                                audio_url_path = results[0].get("file")
                                # 3. Download the audio
                                download_url = f"http://{DEVICE_IP}:8001{audio_url_path}"
                                audio_response = requests.get(download_url, timeout=30)
                                if audio_response.status_code == 200:
                                    filename = "test_acestep_output.mp3"
                                    with open(filename, "wb") as f:
                                        f.write(audio_response.content)
                                    print(f"  [SAVED] Music saved to: {filename} ({len(audio_response.content)} bytes)")
                                    return True
                        elif status_code == 2:  # Failed
                            print(f"  [ERROR] Generation failed on server")
                            return False
                    
                    time.sleep(3)
                print("  [ERROR] Polling timed out")
                return False
        
        print(f"  [ERROR] ACE-Step failed: {response.text}")
        return False
    except Exception as e:
        print(f"  [ERROR] ACE-Step test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("      ANTIGRAVITY AI - Server Health Check")
    print("=" * 60)
    
    all_ok = True
    
    # Check health of all servers
    print("\n[HEALTH] Checking server status...")
    for server in SERVERS:
        status = check_health(server)
        if status:
            print(f"  [ONLINE]  {server['name']} (Port {server['port']}) at {DEVICE_IP}")
        else:
            print(f"  [OFFLINE] {server['name']} (Port {server['port']}) at {DEVICE_IP}")
            all_ok = False
    
    if not all_ok:
        print("\n[WARNING] Some servers are offline or unreachable at this IP!")
        print(f"Make sure servers are running and DEVICE_IP in this script matches your machine.")
        print("Run 'run_server.bat' to start all servers locally.")
        return False
    
    print("\n" + "=" * 60)
    print("      Running API Tests")
    print("=" * 60)
    
    # Test TTS
    tts_ok = test_tts_server()
    
    # Test ACE-Step
    acestep_ok = test_acestep_server()
    
    # Summary
    print("\n" + "=" * 60)
    print("      Test Summary")
    print("=" * 60)
    print(f"  Qwen3 TTS:    {'[OK]' if tts_ok else '[FAILED]'}")
    print(f"  ACE-Step:     {'[OK]' if acestep_ok else '[FAILED]'}")
    print("=" * 60)
    
    return tts_ok and acestep_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
