---
name: Use Qwen3-ASR for Audio Transcription and Analysis
description: Transcribe audio files or perform audio analysis/chat using the Qwen3-ASR (Qwen2-Audio) model.
---

# Qwen3-ASR Skill

This skill allows you to interact with the Qwen3-ASR server (backed by Qwen2-Audio-7B-Instruct) for Speech-to-Text (ASR) and general audio analysis.

## Server Details
- **URL**: `http://192.168.1.26:8002` (Default)
- **Model**: `Qwen/Qwen2-Audio-7B-Instruct`
- **Cold Start**: Service sleeps after inactivity. Expect 5-10s delay on first request.

## Endpoints

### 1. Transcribe Audio (ASR)
**Endpoint**: `POST /v1/audio/transcriptions`

**Parameters**:
- `file`: The audio file to transcribe (multipart/form-data).
- `prompt`: (Optional) Context or specific instruction (guided ASR).
- `language`: (Optional) Target language hint.

**Example (Python)**:
```python
import requests

url = "http://192.168.1.26:8002/v1/audio/transcriptions"
files = {"file": open("path/to/audio.wav", "rb")}
data = {"prompt": "Transcribe this meeting recording."}

response = requests.post(url, files=files, data=data)
print(response.json())
# Output: {"text": "The transcribed text..."}
```

### 2. Audio Chat / Analysis
**Endpoint**: `POST /v1/chat/completions`

**Payload**: OpenAI-compatible chat format, with audio content support.
Audio can be provided as a URL or base64 data URI.

**Example (Python)**:
```python
import requests

url = "http://192.168.1.26:8002/v1/chat/completions"
messages = [
    {
        "role": "user", 
        "content": [
            {"type": "audio", "audio_url": "https://example.com/audio.wav"},
            {"type": "text", "text": "What is the emotion of the speaker?"}
        ]
    }
]

response = requests.post(url, json={"messages": messages})
print(response.json())
```
