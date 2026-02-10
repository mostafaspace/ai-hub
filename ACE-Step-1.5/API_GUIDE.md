# ACE-Step 1.5 API Guide for AI Agents

Complete guide for using the ACE-Step music generation API.

## Quick Start

**Base URL:** `http://192.168.1.26:8001`

```python
import requests

# Generate music
response = requests.post("http://192.168.1.26:8001/release_task", json={
    "prompt": "upbeat pop song with catchy melody",
    "lyrics": "[Verse]\nHello world\n[Chorus]\nLa la la",
    "audio_duration": 30,
    "thinking": True
})
task_id = response.json()["data"]["task_id"]
```

---

## Endpoints

### 1. Generate Music - `POST /release_task`

Creates a music generation task.

#### Essential Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | `""` | Music style description |
| `lyrics` | string | `""` | Song lyrics with `[Verse]`, `[Chorus]` tags |
| `audio_duration` | float | 30 | Duration in seconds (10-600) |
| `thinking` | bool | `false` | **Recommended: `true`** - Uses LM for better quality |
| `audio_format` | string | `"mp3"` | Output format: `mp3`, `wav`, `flac` |
| `batch_size` | int | 2 | Number of variations (1-8) |

#### Optional Music Control

| Parameter | Type | Description |
|-----------|------|-------------|
| `bpm` | int | Tempo (30-300) |
| `key_scale` | string | Key: `"C Major"`, `"Am"`, etc. |
| `time_signature` | string | `"4"` for 4/4, `"3"` for 3/4 |
| `inference_steps` | int | Diffusion steps (default 8 for turbo) |

#### Sample Mode (Auto-Generate Everything)

| Parameter | Type | Description |
|-----------|------|-------------|
| `sample_query` | string | Natural description: `"a soft jazz song"` |
| `use_format` | bool | LM enhances your prompt/lyrics |

#### Response

```json
{
  "data": {
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "queued",
    "queue_position": 1
  },
  "code": 200
}
```

---

### 2. Check Task Status - `POST /query_result`

#### Request

```json
{"task_id_list": ["550e8400-e29b-41d4-a716-446655440000"]}
```

#### Response

```json
{
  "data": [{
    "task_id": "...",
    "status": 1,
    "result": "[{\"file\": \"/v1/audio?path=...\", \"status\": 1, ...}]"
  }],
  "code": 200
}
```

**Status codes:** `0` = processing, `1` = success, `2` = failed

---

### 3. Download Audio - `GET /v1/audio?path={encoded_path}`

Use the `file` URL from the task result to download the generated audio.

---

### 4. List Models - `GET /v1/models`

---

### 5. Server Stats - `GET /server_stats`

---

### 6. Health Check - `GET /health`

---

## Complete Python Example

```python
import requests
import time
import json

BASE_URL = "http://192.168.1.26:8001"

def generate_music(prompt, lyrics="", duration=30):
    """Generate music and return file paths."""
    
    # 1. Create task
    resp = requests.post(f"{BASE_URL}/release_task", json={
        "prompt": prompt,
        "lyrics": lyrics,
        "audio_duration": duration,
        "thinking": True,
        "batch_size": 1
    })
    task_id = resp.json()["data"]["task_id"]
    
    # 2. Poll until complete
    while True:
        resp = requests.post(f"{BASE_URL}/query_result", 
            json={"task_id_list": [task_id]})
        result = resp.json()["data"][0]
        
        if result["status"] == 1:  # Success
            files = json.loads(result["result"])
            return [f["file"] for f in files]
        elif result["status"] == 2:  # Failed
            raise Exception("Generation failed")
        
        time.sleep(3)
    
    # 3. Download: GET {BASE_URL}{file_path}

# Usage
files = generate_music(
    prompt="energetic rock song with electric guitar",
    lyrics="[Verse]\nRock and roll tonight\n[Chorus]\nYeah!",
    duration=30
)
```

---

## cURL Examples

**Basic Generation:**
```bash
curl -X POST http://192.168.1.26:8001/release_task \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "jazz piano trio", "thinking": true, "audio_duration": 30}'
```

**Check Status:**
```bash
curl -X POST http://192.168.1.26:8001/query_result \
  -H 'Content-Type: application/json' \
  -d '{"task_id_list": ["YOUR_TASK_ID"]}'
```

**Sample Mode (LM generates everything):**
```bash
curl -X POST http://192.168.1.26:8001/release_task \
  -H 'Content-Type: application/json' \
  -d '{"sample_query": "a romantic acoustic ballad", "thinking": true}'
```

---

## Best Practices

1. **Always use `thinking: true`** for best quality
2. **Poll every 3-5 seconds** - generation takes 10-60s depending on duration
3. **Use `sample_query`** if you want fully automated generation
4. **Add `[Verse]`, `[Chorus]` tags** in lyrics for better structure
5. **Duration 30-60s** is optimal for most use cases
