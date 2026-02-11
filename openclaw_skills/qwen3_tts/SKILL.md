---
name: qwen3_tts
description: API documentation for the Qwen3 TTS server, enabling speech generation, voice design, and voice cloning.
---

# Qwen3 TTS API Skill

This skill provides the API specification for the Qwen3 TTS server. Agents can use this information to construct HTTP requests to generate speech.

## Server Details

-   **Base URL**: `http://192.168.1.26:8000`
-   **OpenAPI Spec**: `openapi.yaml` (in this directory)

> [!IMPORTANT]
> **Timeout Setting**: The first request may take up to **60-120 seconds** to load the model. Ensure your HTTP client has a long timeout.
> **Network Access**: Ensure the agent can reach `192.168.1.26`. If running locally on the same machine, you can use `http://127.0.0.1:8000`.

## Capabilities

1.  **Standard TTS**: Generate speech from text using preset voices.
2.  **Voice Design**: Generate speech using a voice created from a text description.
3.  **Voice Cloning**: Generate speech using a voice cloned from a reference audio file.

## API Endpoints

### 1. Generate Speech (Standard)
**POST** `/v1/audio/speech`

Generates audio from text using a specific voice ID.

**JSON Body:**
```json
{
  "input": "Hello world, this is a test.",
  "voice": "Vivian", 
  "language": "Auto",
  "instruct": "Speak clearly and slowly",
  "response_format": "mp3"
}
```
-   `voice`: Default is "Vivian". Use `GET /v1/audio/voices` to list others.
-   `response_format`: `mp3` (default), `wav`, `pcm`, `ogg`.
    -   `ogg`: Best for WhatsApp compatibility (Opus codec).

### 2. Voice Design (Text-to-Voice)
**POST** `/v1/audio/voice_design`

Creates a voice on-the-fly based on a description.

**JSON Body:**
```json
{
  "input": "This voice was designed by an AI agent.",
  "instruct": "A deep, resonant male voice with a slight British accent, sounding like a narrator.",
  "language": "Auto",
  "response_format": "mp3"
}
```
-   `instruct`: The description of the voice you want to create.
-   `response_format`: `mp3` (default), `wav`, `pcm`, `ogg`. (Use `ogg` for WhatsApp).

### 3. Voice Cloning (Audio-to-Voice)
**POST** `/v1/audio/voice_clone`

**Content-Type**: `multipart/form-data`

Clones a voice from a reference audio file.

**Fields:**
-   `text`: The text to speak (String).
-   `ref_audio`: The reference audio file (File).
-   `ref_text`: The text spoken in the reference audio (String).
-   `language`: Language code (String, default "Auto").

### 4. List Voices
**GET** `/v1/audio/voices`

Returns a list of available preset voices.

## Example Workflow (Agent)

To generate speech with a custom designed voice:

1.  Construct the JSON payload with `input` (text to speak) and `instruct` (voice description).
2.  Send a **POST** request to `http://192.168.1.26:8000/v1/audio/voice_design`.
3.  Receive the binary audio stream (content-type: `audio/mpeg` or `audio/wav`).
4.  Save the stream to a file.

## Troubleshooting

### Connection Refused / Timeout
-   **Check IP**: Ensure `192.168.1.26` is reachable from the agent's machine. Try `ping 192.168.1.26`.
-   **Firewall**: Ensure port `8000` is open on the server machine.
-   **First Request Slow**: The server loads models on demand. The first request can take 1-2 minutes.
