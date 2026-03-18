---
name: content_director
description: Content Director Macro-Skill - Generate a fully narrated short video from scratch using an image prompt and voiceover text.
---

# Content Director Macro-Skill

The **Content Director** is a heavy compute "Macro-Skill" exposed by the AI-Hub Orchestrator. 
Instead of forcing you (the agent) to manually orchestrate Z-Image, Qwen3-TTS, LTX-2-Video, ACE-Step-Music, and FFmpeg one by one, the Orchestrator handles the entire complex pipeline server-side.

This is an **Asynchronous Endpoint**. It triggers multiple foundation models in sequence, taking **5-15 minutes** to complete.

## Server Details

- **Base URL**: `http://{{HUB_IP}}:9000`

## Agent Workflow

### 1. Start a Task
Send a POST request to initiate the generation.

**POST** `/v1/workflows/director`

**JSON Body:**
- `image_prompt`: Detailed visual description.
- `voiceover_text`: Script for the narrator.
- `voice`: (Optional) Preset voice ID (default: "vivian").
- `music_prompt`: (Optional) Style description for background music (e.g., "fast heavy metal").
- `music_volume`: (Optional) Float 0.0 to 1.0 (default: 0.2).

**Example:**
```json
{
  "image_prompt": "A cyberpunk city at night with neon lights.",
  "voiceover_text": "Welcome to the future.",
  "music_prompt": "lofi hip hop track",
  "music_volume": 0.15
}
```

**Response:**
```json
{
  "status": "processing",
  "task_id": "task_123",
  "message": "Director task started."
}
```

### 2. Poll for Progress
Wait 30 seconds, then poll every 10 seconds.

**GET** `/v1/hub/tasks/{task_id}`

**Response while processing:**
```json
{
  "task_id": "task_123",
  "status": "processing",
  "step": "music"
}
```

**Response when finished:**
```json
{
  "task_id": "task_123",
  "status": "completed",
  "output_url": "http://{{HUB_IP}}:9000/outputs/task_123/final_director_cut.mp4"
}
```

Once the response returns, provide the `output_url` to the user.
