---
name: studio_tools
description: API documentation for the AI-Hub Practical Studio endpoints, including workspaces, uploads, character packs, timeline render, captions, thumbnails, transcodes, voice auditions, prompt comparison, project export/import, and async task controls.
---

# Practical Studio API Skill

The **Practical Studio** endpoints extend the Orchestrator with non-destructive user-facing tools for organizing media projects and producing derivative assets without changing the existing production routes.

## Server Details

- **Base URL**: `http://192.168.1.26:9000`
- **OpenAPI Spec**: `openapi.yaml`

## Core Capabilities

1. **Project Workspaces**: Create reusable project containers that track assets, notes, a timeline manifest, assets, and related tasks.
2. **Character Consistency Packs**: Save prompt prefixes/suffixes, default voice settings, negative prompts, and an optional reference photo.
3. **Upload-First Reuse**: Upload media once to Studio and reuse the returned hosted URL across caption, thumbnail, transcode, and timeline jobs.
4. **Safe Timeline Render**: Render a new output from stored clips/audio/subtitles without mutating originals.
5. **Auto-Captions & Burn-In**: Generate approximate SRT captions and optionally produce a subtitle-burned video.
6. **Thumbnails & Contact Sheets**: Build preview images for videos.
7. **Output Profiles**: Apply reusable delivery presets like `youtube_short`, `discord_clip`, `podcast_mp3`, and `whatsapp_voice`.
8. **Voice Audition Mode**: Generate the same text in multiple voices and receive downloadable sample URLs.
9. **Prompt Compare Mode**: Run multiple prompt variants through the Vision backend and collect the outputs in one async task.
10. **Director-to-Project Runs**: Run the existing Director workflow behind an async Studio task and automatically attach the final video to a project.
11. **Project Export / Import**: Package a workspace and bundled local assets into a zip and import it elsewhere.
12. **Task Durability**: Persist Studio tasks to disk, expose task events, and support cancel / resume / retry controls.
13. **Webhooks & Observability**: Send optional task callbacks and query Studio metrics for counts, durations, failures, and storage usage.

## Request Guidance

- For Windows `curl`, write JSON bodies to a temporary `payload.json` file and use `curl -d @payload.json` instead of inline JSON.
- All long-running jobs return a `task_id` immediately and must be polled.
- Use Studio uploads first when you have local files and need a reusable URL for later tasks.

## Async Task Pattern

1. Submit the request to a `POST /v1/studio/...` endpoint.
2. Receive `{ "task_id": "...", "status": "processing" }` immediately.
3. Poll `GET /v1/studio/tasks/{task_id}` until the status becomes `completed`, `failed`, `cancelled`, or `interrupted`.
4. Optional controls:
   - `POST /v1/studio/tasks/{task_id}/cancel`
   - `POST /v1/studio/tasks/{task_id}/resume`
   - `POST /v1/studio/tasks/{task_id}/retry`
5. Download URLs from the `result` field when complete.

## Important Endpoints

### Upload Media First
**POST** `/v1/studio/uploads` (multipart form)

Form fields:
- `file`: binary file
- `project_id` (optional)
- `pack_id` (optional)
- `kind` (optional)
- `label` (optional)
- `use_as_pack_photo` (optional boolean)

### Create a Workspace
**POST** `/v1/studio/projects`

`payload.json`
```json
{
  "name": "Launch Kit",
  "description": "Assets for a product launch",
  "default_profile": "youtube_short",
  "tags": ["launch", "video"]
}
```

Example:
```bash
curl -X POST http://192.168.1.26:9000/v1/studio/projects ^
  -H "Content-Type: application/json" ^
  -d @payload.json
```

### Save a Character Pack
**POST** `/v1/studio/character-packs`

`payload.json`
```json
{
  "name": "Hero Pack",
  "voice": "Vivian",
  "prompt_prefix": "cinematic hero portrait",
  "prompt_suffix": "high detail, dramatic lighting",
  "negative_prompt": "blurry",
  "photo_url": "http://192.168.1.26:9000/outputs/practical/uploads/generic/.../headshot.jpg"
}
```

### Attach a Character Photo Later
**POST** `/v1/studio/character-packs/{pack_id}/photo`

Multipart form with `file` and optional `label`.

### Generate Captions
**POST** `/v1/studio/captions`

`payload.json`
```json
{
  "media_url": "http://192.168.1.26:9000/outputs/practical/uploads/generic/.../source.mp4",
  "burn_in": true,
  "output_profile": "youtube_short",
  "project_id": "proj-launch-kit-ab12cd34"
}
```

### Render a Safe Timeline Output
**POST** `/v1/studio/projects/{project_id}/timeline/render`

`payload.json`
```json
{
  "format_profile": "youtube_short",
  "include_audio_bed": true,
  "burn_subtitles": true
}
```

### Director to Project
**POST** `/v1/studio/projects/{project_id}/director-runs`

`payload.json`
```json
{
  "image_prompt": "cinematic skyline at dusk",
  "voiceover_text": "Tonight on the show",
  "voice": "Vivian"
}
```

### Voice Audition Mode
**POST** `/v1/studio/voice-auditions`

`payload.json`
```json
{
  "text": "Welcome to the launch event.",
  "voices": ["Vivian", "Ethan", "Serena"],
  "response_format": "mp3"
}
```

### Prompt Compare Mode
**POST** `/v1/studio/prompt-compare`

`payload.json`
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

### Export and Import a Project
- **POST** `/v1/studio/projects/{project_id}/export`
- **POST** `/v1/studio/projects/import` (multipart `file` upload)

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