# Qwen3 TTS API Guide for AI Agents

Complete guide for using the Qwen3 TTS (Text-to-Speech) API.

## Quick Start

**Base URL:** `http://192.168.1.26:8000`

```python
import requests

# Generate speech
response = requests.post("http://192.168.1.26:8000/v1/audio/speech", json={
    "input": "Hello, world! This is a test of the Qwen3 TTS API.",
    "voice": "Vivian",
    "response_format": "mp3"
})

if response.status_code == 200:
    with open("output.mp3", "wb") as f:
        f.write(response.content)
```

---

## Endpoints

### 1. Standard TTS - `POST /v1/audio/speech`

Generates high-quality speech using preset speakers.

#### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | **Required** | The text to convert to speech |
| `voice` | string | `"Vivian"` | Speaker name (e.g., `"Vivian"`, `"Ryan"`, `"Chelsie"`) |
| `language` | string | `"Auto"` | Content language (e.g., `"en"`, `"zh"`, `"ja"`) |
| `response_format` | string | `"mp3"` | Output format: `"wav"`, `"mp3"`, `"pcm"` |

---

### 2. Voice Design - `POST /v1/audio/voice_design`

Creates a unique voice from a natural language description.

#### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | **Required** | The text to speak |
| `instruct` | string | **Required** | Voice description (e.g., `"A deep male voice with a calm tone"`) |
| `language` | string | `"Auto"` | Content language |
| `response_format` | string | `"mp3"` | Output format |

---

### 3. Voice Cloning - `POST /v1/audio/voice_clone`

Clones a voice from a reference audio file.

> [!NOTE]
> This endpoint uses `multipart/form-data`.

#### Request Parameters (Form Fields)

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | string | The text to speak |
| `ref_text` | string | Transcribed text of the reference audio |
| `ref_audio` | file | The reference audio file to clone from |
| `language` | string | Language (default: `"Auto"`) |
| `response_format` | string | Format (default: `"mp3"`) |

---

### 4. Health Check - `GET /health`

Returns server status and active device.

```json
{
  "status": "running",
  "device": "cuda"
}

---

### 5. List Models - `GET /v1/models`

Returns a list of available models.

### 6. List Voices - `GET /v1/audio/voices`

Returns a list of available voices and speakers.

### 7. TTS Alias - `POST /v1/audio/tts`

Alias for `POST /v1/audio/speech` to support clients expecting this standard endpoint.
```

---

## Best Practices

1. **Voice Selection**: `"Vivian"` is the most reliable default for English.
2. **Batching**: For long texts, split by sentences to avoid timeouts and high VRAM usage.
3. **Format**: Use `"mp3"` for smaller file sizes, or `"wav"` for lossless quality.
4. **VRAM Management**: High-tier GPUs (RTX 3090/4090/5090) perform best with these models.
