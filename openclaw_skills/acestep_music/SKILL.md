---
name: acestep_music_api
description: API documentation for the ACE-Step Music server, enabling music generation with lyrics and style control.
---

# ACE-Step Music API Skill

This skill provides the API specification for the ACE-Step Music server. Agents can use this information to construct HTTP requests to generate music tracks.

## Server Details

-   **Base URL**: `http://192.168.1.26:8001`
-   **OpenAPI Spec**: `openapi.yaml` (in this directory)

> [!IMPORTANT]
> **Polling Time**: Music generation takes **20-60 seconds**. You must poll the status until it completes.
> **First Request Slow**: The first request may take longer as the model loads into memory.
> **Network Access**: Ensure the agent can reach the server IP (default: `192.168.1.26`).
> **Cold Start**: The service uses dynamic model loading. The first request after interactivity may take **extra time** (up to 20s) to initialize.

> [!CAUTION]
> **SHELL QUOTING (CRITICAL)**: If you use `curl` or `Invoke-RestMethod` from a Windows shell, your JSON body will be truncated at the first space unless properly quoted.
> **DO NOT DO THIS:** `curl -d '{"prompt": "An upbeat..."}'` (Quotes will fail, payload truncates to `{"prompt": "An`)
> **DO THIS INSTEAD:** Save your JSON to a file first: `echo {"prompt": "An upbeat track"} > payload.json` then `curl -d @payload.json`

## Capabilities

1.  **Music Generation**: Create tracks from text prompts.
2.  **Lyrics Support**: Include verse/chorus lyrics in the song.
3.  **Musical Control**: Specify BPM, Key, and Time Signature.
4.  **Sample Mode**: Auto-generate a track from a simple description.

## API Endpoints

### 1. Start Generation Task
**POST** `/v1/audio/async_generations`

Initiates a background task to generate music.

**JSON Body (Standard):**
```json
{
  "prompt": "An upbeat pop song about summer.",
  "lyrics": "[Verse]\nSun is shining...\n[Chorus]\nIt's a beautiful day!",
  "audio_duration": 30,
  "thinking": true,
  "bpm": 120,
  "key_scale": "C Major",
  "time_signature": "4"
}
```
-   `thinking`: **Always set to true** for best quality (uses LM reasoning).
-   `audio_duration`: In seconds (10-600).

**JSON Body (Sample Mode):**
```json
{
  "sample_query": "A sad violin solo",
  "thinking": true
}
```
-   `sample_query`: Let the AI infer all musical parameters from this description.

**Response:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "message": "Task queued. Poll GET /v1/audio/tasks/{task_id}"
}
```

### 2. Check Status (Poll)
**GET** `/v1/audio/tasks/{task_id}`

Check if the generation is complete by querying the `task_id`. You must poll this endpoint every 3-5 seconds.

**Response (Processing):**
```json
{
  "status": "processing"
}
```

**Response (Success):**
```json
{
  "status": "completed",
  "data": [
    {
      "url": "http://192.168.1.26:8001/v1/audio?path=..."
    }
  ]
}
```

### 3. Download Audio
**GET** `{url}`

Using the `url` from the polling result (e.g., `http://192.168.1.26:8001/v1/audio?path=...`), download the MP3.

## Example Workflow (Agent)

1.  **Start Task**: Send **POST** to `/v1/audio/async_generations` with your `prompt` and `lyrics`.
2.  **Get ID**: Extract `task_id` from the JSON response.
3.  **Poll**: Loop every 3 seconds:
    -   Send **GET** to `/v1/audio/tasks/{task_id}`.
    -   Check `status`.
    -   If `"completed"`, break loop. If `"failed"`, stop (error).
4.  **Get URL**: Retrieve the `url` from `data[0].url`.
5.  **Download**: Send **GET** to the `url` and save the file.

### Generation Stuck on "Processing"
-   **Wait Longer**: Complex prompts or high loads can take up to 2 minutes.
