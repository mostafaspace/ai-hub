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
> **Network Access**: Ensure the agent can reach `192.168.1.26` (or `127.0.0.1` if local).

## Capabilities

1.  **Music Generation**: Create tracks from text prompts.
2.  **Lyrics Support**: Include verse/chorus lyrics in the song.
3.  **Musical Control**: Specify BPM, Key, and Time Signature.
4.  **Sample Mode**: Auto-generate a track from a simple description.

## API Endpoints

### 1. Start Generation Task
**POST** `/release_task`

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
  "data": { "task_id": "550e8400-e29b-41d4-a716-446655440000", ... },
  "code": 200
}
```

### 2. Check Status (Poll)
**POST** `/query_result`

Check if the generation is complete. You must poll this endpoint every 3-5 seconds.

**JSON Body:**
```json
{
  "task_id_list": ["550e8400-e29b-41d4-a716-446655440000"]
}
```

**Response:**
```json
{
  "data": [
    {
      "task_id": "...",
      "status": 1, 
      "result": "[{\"file\": \"/v1/audio?path=...\"}]"
    }
  ],
  "code": 200
}
```
-   `status`: `0` (Processing), `1` (Success), `2` (Failed).
-   `result`: A JSON string containing the file path. You must parse this string to get the `file` URL.

### 3. Download Audio
**GET** `{Base URL}{file_path}`

Using the `file` path from the polling result (e.g., `/v1/audio?path=...`), construct the full URL to download the MP3.

## Example Workflow (Agent)

1.  **Start Task**: Send **POST** to `/release_task` with your `prompt` and `lyrics`.
2.  **Get ID**: Extract `task_id` from the response.
3.  **Poll**: Loop every 3 seconds:
    -   Send **POST** to `/query_result` with the `task_id`.
    -   Check `data[0].status`.
    -   If `1`, break loop. If `2`, stop (error).
4.  **Get URL**: Parse `data[0].result` (JSON string) to find the `file` path.
5.  **Download**: Send **GET** to `http://192.168.1.26:8001{file_path}` and save the file.

## Troubleshooting

### Generation Stuck on "Processing"
-   **Wait Longer**: Complex prompts or high loads can take up to 2 minutes.
-   **Check Server Logs**: If it stays stuck forever, the server might have run out of VRAM.

### Connection Refused
-   **Check IP**: Ensure `192.168.1.26` is correct and reachable.
-   **Firewall**: Ensure port `8001` is open.
