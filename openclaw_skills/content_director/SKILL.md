---
name: content_director
description: Content Director Macro-Skill - Generate a fully narrated short video from scratch using an image prompt and voiceover text.
---

# Content Director Macro-Skill

The **Content Director** is a heavy compute "Macro-Skill" exposed by the AI-Hub Orchestrator. 
Instead of forcing you (the agent) to manually orchestrate Z-Image, Qwen3-TTS, LTX-2-Video, and FFmpeg one by one, the Orchestrator handles the entire complex pipeline server-side.

This is a **Synchronous Endpoint**, meaning the HTTP request will remain open and block until the final `.mp4` is finished rendering. Because it triggers 3 individual foundation models in sequence, **this request will take 10-15 minutes to complete on a consumer GPU.**

You **MUST** set your HTTP client timeout to at least `timeout=1200` (20 minutes). If your HTTP client drops the connection early, the Orchestrator will continue rendering in the background but you will lose the JSON response URL.

## Server Details

-   **Base URL**: `http://192.168.1.26:9000`

## Agent Workflow

### Generate a Narrated Video
Send a direct POST request to the Orchestrator. It will generate the image, synthesize the voiceover, animate the image, and stitch them together using FFmpeg automatically.

**POST** `/v1/workflows/director`

**JSON Body Requirements:**
- `image_prompt` (str): Highly detailed, photorealistic visual description for the base frame sequence.
- `voiceover_text` (str): The script the narrator will read.
- `voice` (str, optional): The voice profile. Default is `"Vivian"`.

**Example Python Request:**
```python
import httpx

url = "http://192.168.1.26:9000/v1/workflows/director"
payload = {
    "image_prompt": "A cyberpunk city at night with neon lights and flying cars, cinematic lighting, 4k.",
    "voiceover_text": "Welcome to the future. A city that never sleeps, driven by the neon heartbeat of tomorrow.",
    "voice": "Vivian"
}

# CRITICAL: timeout=None or very high number is required.
response = httpx.post(url, json=payload, timeout=None)
print(response.json())
```

**Final Response:**
```json
{
  "status": "COMPLETED",
  "task_id": "8b9cad0e1f20",
  "message": "Content Director successfully assembled the video.",
  "output_url": "http://127.0.0.1:9000/outputs/8b9cad0e1f20/final_director_cut.mp4"
}
```

Once the response returns, you can provide the `output_url` to the user or download it if needed.
