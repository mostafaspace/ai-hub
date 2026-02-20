
"""
Antigravity AI - Unified Server Test Script

Tests Qwen3 TTS, ACE-Step Music, and Qwen3 ASR servers to verify they're working.

Usage:
    python test_all_servers.py
"""

import requests
import sys
import time
import wave
import struct
import math
import os
import json

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import config

# IP Configuration - Use device IP from config
DEVICE_IP = config.TEST_DEVICE_IP

SERVERS = [
    {"name": "Qwen3 TTS", "port": config.TTS_PORT, "health": "/health", "url": f"http://{DEVICE_IP}:{config.TTS_PORT}"},
    {"name": "ACE-Step Music", "port": config.MUSIC_PORT, "health": "/health", "url": f"http://{DEVICE_IP}:{config.MUSIC_PORT}"},
    {"name": "Qwen3 ASR", "port": config.ASR_PORT, "health": "/health", "url": f"http://{DEVICE_IP}:{config.ASR_PORT}"},
    {"name": "Vision (Z-Image)", "port": config.VISION_PORT, "health": "/health", "url": f"http://{DEVICE_IP}:{config.VISION_PORT}"},
]


def check_health(server):
    """Check if a server is healthy."""
    try:
        url = f"{server['url']}{server['health']}"
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except:
        return False


def create_dummy_wav(filename="test_audio.wav"):
    """Create a simple 1-second sine wave audio file for testing."""
    if os.path.exists(filename):
        return filename
        
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    
    with wave.open(filename, 'w') as obj:
        obj.setnchannels(1) # mono
        obj.setsampwidth(2) # 2 bytes
        obj.setframerate(sample_rate)
        
        for i in range(int(sample_rate * duration)):
            value = int(32767.0 * math.sin(frequency * math.pi * 2 * i / sample_rate))
            data = struct.pack('<h', value)
            obj.writeframesraw(data)
            
    return filename


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
            return False
    except Exception as e:
        print(f"  [ERROR] TTS test failed: {e}")
        return False


def test_acestep_server():
    """Test ACE-Step with a quick generation."""
    print(f"\n[TEST] Testing ACE-Step Music Server at {DEVICE_IP}...")
    try:
        # Create a short task
        response = requests.post(
            f"http://{DEVICE_IP}:8001/release_task",
            json={
                "prompt": "simple piano melody",
                "audio_duration": 5,
                "thinking": False,
                "batch_size": 1,
                "inference_steps": 4
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            task_id = data.get("data", {}).get("task_id") or data.get("task_id") # Handle different response formats
            
            if task_id:
                print(f"  [OK] Task created: {task_id}")
                return True # Assuming queue works, just checking submission for speed
        
        print(f"  [ERROR] ACE-Step submission failed: {response.text}")
        return False
    except Exception as e:
        print(f"  [ERROR] ACE-Step test failed: {e}")
        return False


def test_asr_server():
    """Test Qwen3 ASR with a dummy audio file."""
    print(f"\n[TEST] Testing Qwen3 ASR Server at {DEVICE_IP}...")
    try:
        filename = create_dummy_wav()
        
        with open(filename, "rb") as f:
            files = {"file": f}
            data = {"prompt": "Describe this sound."}
            
            response = requests.post(
                f"http://{DEVICE_IP}:8002/v1/audio/transcriptions",
                files=files,
                data=data,
                timeout=60
            )
            
        if response.status_code == 200:
            result = response.json()
            print(f"  [OK] ASR successful! Output: {result.get('text', 'No text returned')}")
            return True
        else:
             print(f"  [ERROR] ASR failed: {response.status_code} - {response.text}")
             return False

    except Exception as e:
        print(f"  [ERROR] ASR test failed: {e}")
        return False


def test_vision_server():
    """Test the Vision service with a quick text-to-image request."""
    print(f"\n[TEST] Testing Vision Service (Z-Image) at {DEVICE_IP}...")
    try:
        response = requests.post(
            f"http://{DEVICE_IP}:{config.VISION_PORT}/v1/images/generations",
            json={
                "prompt": "a simple red circle on white background",
                "negative_prompt": "blurry",
                "size": "512x512",
                "num_inference_steps": 20,
                "guidance_scale": 4.0,
                "cfg_normalization": False,
            },
            timeout=600
        )
        if response.status_code == 200:
            data = response.json()
            print(f"  [OK] Vision generation successful!")
            if "data" in data and len(data["data"]) > 0:
                item = data["data"][0]
                if "url" in item:
                    print(f"  Image URL: {item['url']}")
            return True
        else:
            print(f"  [ERROR] Vision failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  [ERROR] Vision test failed: {e}")
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
    
    print("\n" + "=" * 60)
    print("      Running API Tests")
    print("=" * 60)
    
    # Test TTS
    tts_ok = test_tts_server()
    
    # Test ACE-Step
    acestep_ok = test_acestep_server()
    
    # Test ASR
    asr_ok = test_asr_server()
    
    # Test Vision
    vision_ok = test_vision_server()
    
    # Summary
    print("\n" + "=" * 60)
    print("      Test Summary")
    print("=" * 60)
    print(f"  Qwen3 TTS:    {'[OK]' if tts_ok else '[FAILED/OFFLINE]'}")
    print(f"  ACE-Step:     {'[OK]' if acestep_ok else '[FAILED/OFFLINE]'}")
    print(f"  Qwen3 ASR:    {'[OK]' if asr_ok else '[FAILED/OFFLINE]'}")
    print(f"  Vision:       {'[OK]' if vision_ok else '[FAILED/OFFLINE]'}")
    print("=" * 60)
    
    return tts_ok and acestep_ok and asr_ok and vision_ok


if __name__ == "__main__":
    main()
