"""
ACE-Step 1.5 Music Generation API Test Client

This script demonstrates how to use the ACE-Step API to generate music.
It shows various generation modes and proper handling of the async task workflow.

Usage:
    1. Start the ACE-Step API server: python acestep/api_server.py (or run_acestep_server.bat)
    2. Run this script: python test_acestep_client.py

Requirements:
    - requests library: pip install requests
"""

import requests
import time
import json
import os
import sys
from urllib.parse import urlencode

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Server configuration
BASE_URL = os.environ.get("ACESTEP_API_URL", "http://192.168.1.26:8001")
OUTPUT_DIR = "generated_music"


def check_server_health():
    """Check if the ACE-Step server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("[OK] Server is healthy and running!")
            return True
        else:
            print(f"[ERROR] Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to server. Make sure ACE-Step API is running on port 8001")
        return False


def list_available_models():
    """List available DiT models on the server."""
    try:
        response = requests.get(f"{BASE_URL}/v1/models")
        if response.status_code == 200:
            data = response.json()
            print("\n[MODELS] Available Models:")
            for model in data.get("data", []):
                print(f"   - {model.get('id', 'unknown')}")
            return data
    except Exception as e:
        print(f"Error listing models: {e}")
    return None


def get_server_stats():
    """Get server statistics."""
    try:
        response = requests.get(f"{BASE_URL}/server_stats")
        if response.status_code == 200:
            data = response.json()
            stats = data.get("data", {})
            print("\n[STATS] Server Statistics:")
            print(f"   Queue length: {stats.get('queue_length', 0)}")
            print(f"   Active: {stats.get('is_active', False)}")
            return stats
    except Exception as e:
        print(f"Error getting stats: {e}")
    return None


def create_generation_task(
    prompt: str,
    lyrics: str = "",
    duration: float = 30.0,
    thinking: bool = True,
    audio_format: str = "mp3",
    batch_size: int = 1,
    bpm: int = None,
    key_scale: str = None,
    inference_steps: int = 8,
    **kwargs
):
    """
    Create a music generation task.
    
    Args:
        prompt: Music description (e.g., "upbeat pop song with catchy melody")
        lyrics: Song lyrics (optional)
        duration: Generation duration in seconds (10-600)
        thinking: Use LM for enhanced generation (recommended: True)
        audio_format: Output format (mp3, wav, flac)
        batch_size: Number of variations to generate (1-8)
        bpm: Tempo in BPM (30-300, optional)
        key_scale: Musical key (e.g., "C Major", "Am", optional)
        inference_steps: Number of diffusion steps (default 8 for turbo model)
        
    Returns:
        task_id if successful, None otherwise
    """
    payload = {
        "prompt": prompt,
        "lyrics": lyrics,
        "audio_duration": duration,
        "thinking": thinking,
        "audio_format": audio_format,
        "batch_size": batch_size,
        "inference_steps": inference_steps,
        "use_random_seed": True,
    }
    
    # Add optional parameters
    if bpm is not None:
        payload["bpm"] = bpm
    if key_scale is not None:
        payload["key_scale"] = key_scale
    
    # Add any additional kwargs
    payload.update(kwargs)
    
    print(f"\n[GENERATE] Creating generation task...")
    print(f"   Prompt: {prompt}")
    print(f"   Duration: {duration}s")
    print(f"   Thinking mode: {thinking}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/release_task",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            task_id = data.get("data", {}).get("task_id")
            queue_pos = data.get("data", {}).get("queue_position", 0)
            print(f"   [OK] Task created! ID: {task_id}")
            print(f"   Queue position: {queue_pos}")
            return task_id
        else:
            print(f"   [ERROR] Failed to create task: {response.text}")
            return None
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return None


def create_sample_task(description: str, duration: float = 30.0):
    """
    Create a generation task using natural language description.
    The LM will automatically generate lyrics, caption, and metadata.
    
    Args:
        description: Natural language description (e.g., "a soft Bengali love song")
        duration: Generation duration in seconds
        
    Returns:
        task_id if successful, None otherwise
    """
    payload = {
        "sample_query": description,
        "audio_duration": duration,
        "thinking": True,
        "batch_size": 1,
    }
    
    print(f"\n[GENERATE] Creating sample generation task...")
    print(f"   Description: {description}")
    print(f"   Duration: {duration}s")
    
    try:
        response = requests.post(
            f"{BASE_URL}/release_task",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            task_id = data.get("data", {}).get("task_id")
            print(f"   [OK] Task created! ID: {task_id}")
            return task_id
        else:
            print(f"   [ERROR] Failed: {response.text}")
            return None
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return None


def poll_task_result(task_id: str, poll_interval: float = 3.0, max_wait: float = 600.0):
    """
    Poll for task completion and return the result.
    
    Args:
        task_id: The task ID to poll
        poll_interval: Seconds between polls
        max_wait: Maximum seconds to wait
        
    Returns:
        Result data if successful, None otherwise
    """
    print(f"\n[WAIT] Waiting for task {task_id} to complete...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.post(
                f"{BASE_URL}/query_result",
                headers={"Content-Type": "application/json"},
                json={"task_id_list": [task_id]}
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("data", [])
                
                if results:
                    result = results[0]
                    status = result.get("status", 0)
                    
                    # Status: 0 = in progress, 1 = success, 2 = failed
                    if status == 1:
                        print(f"\n   [OK] Task completed!")
                        # Parse the nested result string
                        result_str = result.get("result", "[]")
                        try:
                            parsed_result = json.loads(result_str)
                            return parsed_result
                        except json.JSONDecodeError:
                            return result_str
                    elif status == 2:
                        print(f"\n   [ERROR] Task failed!")
                        return None
                    else:
                        elapsed = int(time.time() - start_time)
                        print(f"   [WAIT] Still processing... ({elapsed}s elapsed)", end="\r")
            
            time.sleep(poll_interval)
            
        except Exception as e:
            print(f"   Error polling: {e}")
            time.sleep(poll_interval)
    
    print(f"\n   [ERROR] Timeout waiting for task completion")
    return None


def download_audio(file_url: str, output_path: str) -> bool:
    """
    Download a generated audio file.
    
    Args:
        file_url: The file URL from the task result (e.g., "/v1/audio?path=...")
        output_path: Local path to save the file
        
    Returns:
        True if successful, False otherwise
    """
    # Construct full URL
    if file_url.startswith("/"):
        full_url = f"{BASE_URL}{file_url}"
    else:
        full_url = file_url
    
    print(f"\n[DOWNLOAD] Downloading audio...")
    print(f"   URL: {full_url}")
    print(f"   Saving to: {output_path}")
    
    try:
        response = requests.get(full_url, stream=True)
        if response.status_code == 200:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = os.path.getsize(output_path)
            print(f"   [OK] Downloaded! Size: {file_size / 1024:.1f} KB")
            return True
        else:
            print(f"   [ERROR] Download failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return False


def generate_music_simple(prompt: str, lyrics: str = "", duration: float = 30.0, output_name: str = "output"):
    """
    High-level function to generate music with a single call.
    Handles task creation, polling, and downloading.
    
    Args:
        prompt: Music description
        lyrics: Song lyrics (optional)
        duration: Duration in seconds
        output_name: Base name for output file (without extension)
        
    Returns:
        Path to downloaded file if successful, None otherwise
    """
    # Create task
    task_id = create_generation_task(prompt=prompt, lyrics=lyrics, duration=duration)
    if not task_id:
        return None
    
    # Poll for completion
    results = poll_task_result(task_id)
    if not results:
        return None
    
    # Download all generated files
    downloaded_files = []
    for i, result in enumerate(results):
        if isinstance(result, dict):
            file_url = result.get("file")
            if file_url:
                # Determine extension from URL or default to mp3
                ext = ".mp3"
                if "wav" in file_url:
                    ext = ".wav"
                elif "flac" in file_url:
                    ext = ".flac"
                
                suffix = f"_{i+1}" if len(results) > 1 else ""
                output_path = os.path.join(OUTPUT_DIR, f"{output_name}{suffix}{ext}")
                
                if download_audio(file_url, output_path):
                    downloaded_files.append(output_path)
    
    return downloaded_files if downloaded_files else None


def main():
    """Main demonstration of the ACE-Step API."""
    print("=" * 60)
    print("   ACE-Step 1.5 Music Generation API - Test Client")
    print("=" * 60)
    
    # Check server health
    if not check_server_health():
        print("\nPlease start the ACE-Step server first!")
        print("Run: python acestep/api_server.py")
        print("Or:  uv run acestep-api")
        return
    
    # List available models
    list_available_models()
    
    # Get server stats
    get_server_stats()
    
    # Example 1: Basic generation with prompt and lyrics
    print("\n" + "=" * 60)
    print("   Example 1: Basic Music Generation")
    print("=" * 60)
    
    files = generate_music_simple(
        prompt="upbeat electronic pop song with energetic synths and catchy melody",
        lyrics="""[Verse 1]
Dancing through the night
Stars are shining bright
Feel the rhythm flow
Let the music go

[Chorus]
We're alive tonight
Everything feels right
Moving to the beat
This moment is so sweet""",
        duration=30.0,
        output_name="demo_pop_song"
    )
    
    if files:
        print(f"\n[SUCCESS] Successfully generated music!")
        for f in files:
            print(f"   File: {f}")
    else:
        print("\n[WARNING] Generation failed or no files downloaded")
    
    print("\n" + "=" * 60)
    print("   Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
