---
name: studio_tools
description: API documentation for the AI-Hub Practical Studio endpoints, including workspaces, character packs, captions, thumbnails, transcodes, voice auditions, and prompt comparison.
---

# Practical Studio API Skill

The **Practical Studio** endpoints extend the Orchestrator with non-destructive user-facing tools for organizing media projects and producing derivative assets without changing the existing production routes.

## Server Details

- **Base URL**: `http://192.168.1.26:9000`
- **OpenAPI Spec**: `openapi.yaml`

## Core Capabilities

1. **Project Workspaces**: Create reusable project containers that track assets, notes, and a timeline manifest.
2. **Character Consistency Packs**: Save prompt prefixes/suffixes, default voice settings, and negative prompts for reuse.
3. **Timeline Editor Lite**: Store clip/audio/subtitle arrangement data and generate a render plan without mutating source media.
4. **Auto-Captions & Burn-In**: Generate approximate SRT captions from ASR and optionally produce a subtitle-burned video.
5. **Thumbnails & Contact Sheets**: Build preview images for videos.
6. **Output Profiles**: Apply reusable delivery presets like `youtube_short`, `discord_clip`, `podcast_mp3`, and `whatsapp_voice`.
7. **Voice Audition Mode**: Generate the same text in multiple voices and receive downloadable sample URLs.
8. **Prompt Compare Mode**: Run multiple prompt variants through the Vision backend and collect the outputs in one task.

## Async Task Pattern

Long-running media operations follow the hub polling contract:

1. Submit the request to a `POST /v1/studio/...` endpoint.
2. Receive `{ "task_id": "...", "status": "processing" }` immediately.
3. Poll `GET /v1/studio/tasks/{task_id}` until the status becomes `completed` or `failed`.
4. Download the returned URLs when complete.

## Important Endpoints

### Create a Workspace
**POST** `/v1/studio/projects`

```json
{
  "name": "Launch Kit",
  "description": "Assets for a product launch",
  "default_profile": "youtube_short",
  "tags": ["launch", "video"]
}
```

### Save a Character Pack
**POST** `/v1/studio/character-packs`

```json
{
  "name": "Hero Pack",
  "voice": "Vivian",
  "prompt_prefix": "cinematic hero portrait",
  "prompt_suffix": "high detail, dramatic lighting",
  "negative_prompt": "blurry"
}
```

### Generate Captions
**POST** `/v1/studio/captions`

```json
{
  "media_url": "http://192.168.1.26:9000/outputs/practical-demo/source.mp4",
  "burn_in": true,
  "output_profile": "youtube_short"
}
```

### Generate Thumbnails or Contact Sheets
**POST** `/v1/studio/thumbnails`

```json
{
  "media_url": "http://192.168.1.26:9000/outputs/practical-demo/source.mp4",
  "timestamp_sec": 3.5
}
```

**POST** `/v1/studio/contact-sheets`

```json
{
  "media_url": "http://192.168.1.26:9000/outputs/practical-demo/source.mp4",
  "columns": 4,
  "rows": 2,
  "thumb_width": 320
}
```

### Voice Audition Mode
**POST** `/v1/studio/voice-auditions`

```json
{
  "text": "Welcome to the launch event.",
  "voices": ["Vivian", "Ethan", "Serena"],
  "response_format": "mp3"
}
```

### Prompt Compare Mode
**POST** `/v1/studio/prompt-compare`

```json
{
  "prompt_variants": [
    "a clean studio product shot",
    "a dramatic cinematic product shot",
    "a bright social-media product shot"
  ],
  "shared_options": {
    "size": "1024x1024",
    "cfg_normalization": true
  }
}
```

### Poll a Task
**GET** `/v1/studio/tasks/{task_id}`

Completed tasks return URLs inside the `result` field, for example:

```json
{
  "task_id": "abc123",
  "status": "completed",
  "result": {
    "samples": [
      {"voice": "Vivian", "url": "http://192.168.1.26:9000/outputs/practical/abc123/01-vivian.mp3"}
    ]
  }
}
```
