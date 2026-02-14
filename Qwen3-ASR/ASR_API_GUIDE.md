# Qwen3 ASR API Guide for AI Agents

Complete guide for using the Qwen3-ASR (Qwen2-Audio) API for speech-to-text transcription and audio analysis.

## Quick Start

**Base URL:** `http://192.168.1.26:8002`

```python
import requests

# Transcribe an audio file
url = "http://192.168.1.26:8002/v1/audio/transcriptions"
files = {"file": open("recording.wav", "rb")}
response = requests.post(url, files=files)
print(response.json()["text"])
```

---

## Endpoints

### 1. Transcribe Audio - `POST /v1/audio/transcriptions`

OpenAI-compatible transcription endpoint. Accepts an audio file and returns the transcribed text.

**Content-Type**: `multipart/form-data`

#### Request Parameters (Form Fields)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | **Required** | Audio file to transcribe (WAV, MP3, WebM, OGG, etc.) |
| `prompt` | string | `null` | Context or instruction to guide transcription |
| `language` | string | `null` | Target language hint |
| `model` | string | `null` | Model name (ignored, single model) |
| `response_format` | string | `"json"` | Response format |
| `temperature` | float | `0.0` | Sampling temperature |

#### Response

```json
{"text": "The transcribed text from the audio file."}
```

---

### 2. Audio Chat / Analysis - `POST /v1/chat/completions`

OpenAI-compatible chat endpoint with audio support. Use this for audio Q&A, emotion detection, sound classification, or any audio understanding task.

#### Request Body (JSON)

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "audio", "audio_url": "https://example.com/audio.wav"},
        {"type": "text", "text": "What is being said in this audio?"}
      ]
    }
  ],
  "max_tokens": 256,
  "temperature": 0.7
}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | array | **Required** | Chat messages (see audio format below) |
| `max_tokens` | int | `256` | Maximum response length |
| `temperature` | float | `0.7` | Sampling temperature |
| `top_p` | float | `0.9` | Nucleus sampling threshold |

#### Audio Sources

Audio can be provided in the `audio_url` field using any of these formats:

| Format | Example |
|--------|---------|
| **HTTP URL** | `"https://example.com/audio.wav"` |
| **Local file path** | `"C:/path/to/audio.wav"` |
| **File URI** | `"file:///C:/path/to/audio.wav"` |
| **Base64 data URI** | `"data:audio/wav;base64,UklGR..."` |

#### Response

```json
{
  "id": "chatcmpl-1707900000",
  "object": "chat.completion",
  "model": "Qwen2-Audio-7B-Instruct",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "The speaker says: Hello world."},
    "finish_reason": "stop"
  }]
}
```

---

### 3. Health Check - `GET /health`

```json
{
  "status": "running",
  "device": "cuda",
  "device_map": "auto",
  "model_loaded": true
}
```

---

## Model Status & Performance

- **Lazy Loading**: The model loads on first request (~5-10s cold start).
- **Idle Timeout**: Unloads after **60 seconds** of inactivity to free VRAM.
- **Supported Formats**: WAV, MP3, WebM, OGG, FLAC — anything FFmpeg can decode.

---

## Complete Python Examples

### Simple Transcription

```python
import requests

url = "http://192.168.1.26:8002/v1/audio/transcriptions"
files = {"file": open("meeting.wav", "rb")}
data = {"prompt": "This is a business meeting recording."}

response = requests.post(url, files=files, data=data)
print(response.json()["text"])
```

### Audio Analysis (Emotion Detection)

```python
import requests

url = "http://192.168.1.26:8002/v1/chat/completions"
payload = {
    "messages": [{
        "role": "user",
        "content": [
            {"type": "audio", "audio_url": "https://example.com/speech.wav"},
            {"type": "text", "text": "What emotion is the speaker expressing?"}
        ]
    }],
    "max_tokens": 100
}

response = requests.post(url, json=payload)
print(response.json()["choices"][0]["message"]["content"])
```

### Multi-Turn Audio Chat

```python
import requests

url = "http://192.168.1.26:8002/v1/chat/completions"
payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "C:/recordings/sample.wav"},
                {"type": "text", "text": "Transcribe this audio and summarize the key points."}
            ]
        }
    ],
    "max_tokens": 512
}

response = requests.post(url, json=payload)
print(response.json()["choices"][0]["message"]["content"])
```

---

## cURL Examples

**Transcribe:**
```bash
curl -X POST http://192.168.1.26:8002/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "prompt=Transcribe this clearly."
```

**Audio Chat:**
```bash
curl -X POST http://192.168.1.26:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": [{"type": "audio", "audio_url": "https://example.com/audio.wav"}, {"type": "text", "text": "What is being said?"}]}]}'
```

---

## Best Practices

1. **Use transcription endpoint** for simple speech-to-text — it's simpler and optimized for ASR.
2. **Use chat endpoint** for analysis tasks (emotion, classification, Q&A about audio content).
3. **Provide context** via the `prompt` field for better transcription accuracy.
4. **Resample not needed** — the server handles resampling automatically.
5. **Timeout**: Set HTTP client timeout to **60+ seconds** for cold start scenarios.
