# AI-Hub 🚀

Welcome to **AI-Hub**, the unified gateway for open source AI's suite of powerful generative models and services. This repository serves as a central orchestrator, providing a single point of entry to state-of-the-art Text-to-Speech (TTS), Music Generation, Speech Recognition, and Image Generation capabilities.

## 🌟 Overview

AI-Hub is designed to be a modular, scalable, and easy-to-use ecosystem for AI-powered creativity. Whether you're looking to generate expressive speech or compose unique musical tracks, AI-Hub brings these services together under one roof with a unified management system.

## ⚡ Current Services

### 🌐 AI-Hub Orchestrator
The central Gateway/Reverse-Proxy routing API requests to the appropriate backend model engines. Use `GET /health` on port `9000` for a machine-readable hub health check.
- **Path**: [`/orchestrator`](./orchestrator)
- **Primary Endpoint**: `http://<device-ip>:9000/v1/...`
- **Features**: Transparent proxy routing (tts, music, vision, video), GPU crash prevention loops, and custom macro-skills like the `/v1/workflows/director` that stitches image, audio, and video together automatically using local FFmpeg.

### 🎙️ Qwen3 TTS
High-fidelity text-to-speech generation powered by the latest Qwen models.
- **Path**: [`/Qwen3-TTS`](./Qwen3-TTS)
- **Primary Endpoint**: `http://<device-ip>:8000/v1/audio/speech`
- **Features**: Multi-voice support, high-quality MP3 output, and OpenAI-compatible API.

### 🎵 ACE-Step Music
Advanced music generation service utilizing the ACE-Step-1.5 model.
- **Path**: [`/ACE-Step-1.5`](./ACE-Step-1.5)
- **Primary Endpoint**: `http://<device-ip>:8001/v1/audio/async_generations`
- **Features**: Prompt-based music generation, flexible duration settings, and asynchronous task processing with OpenAI-compatible endpoint aliases.

### 🎧 Qwen3 ASR
Automatic Speech Recognition and Audio Intelligence.
- **Path**: [`/Qwen3-ASR`](./Qwen3-ASR)
- **Primary Endpoint**: `http://<device-ip>:8002/v1/audio/transcriptions`
- **Features**: Speech-to-Text transcription and Audio Analysis/Chat using Qwen2-Audio-7B.

### 🖼️ Vision Service (Z-Image)
High-fidelity photorealistic and bilingual image generation powered by the Tongyi-MAI Z-Image 6B diffusion transformer.
- **Path**: [`/Z-Image`](./Z-Image)
- **Primary Endpoint**: `http://<device-ip>:8003/v1/images/generations`
- **Features**: Text-to-image generation, image-to-image editing, OpenAI-compatible API, base64 and URL response formats.

### 🎬 LTX-2 Video
High-fidelity, synchronized audio and video generation powered by Lightricks' DiT-based foundation model.
- **Path**: [`/LTX-2-Video`](./LTX-2-Video)
- **Primary Endpoint**: `http://<device-ip>:8004/v1/video/t2v`
- **Features**: Text-to-Video, Image-to-Video generation, multimodality sync, automatic VRAM unloading.

---

## 💻 System Requirements

AI-Hub runs powerful generative models locally. Below are the estimated VRAM requirements for each service:

| Service | Model | Precision | Estimated VRAM |
| :--- | :--- | :--- | :--- |
| **Qwen3 TTS** | 1.7B Custom/Design/Base | float16 | ~4-6 GB |
| **ACE-Step Music** | 1.5 Turbo (DiT + 1.7B LM) | float16 | ~8-12 GB* |
| **Qwen3 ASR** | Qwen2-Audio-7B | float16 | ~16-18 GB |
| **Vision (Z-Image)** | 6B DiT | bfloat16 | ~16 GB** |
| **LTX-2 Video** | 19B DiT (FP8) | fp8 | ~16-24 GB |

> [!NOTE]
> **ACE-Step Music** scales with available VRAM. It can run in "DiT-only" mode on as little as **4GB VRAM**, or use larger Language Models (up to 4B) if 16GB+ is available.
> **Z-Image** uses CPU offload by default, allowing it to comfortably fit on 16GB-32GB consumer GPUs with high quality `bfloat16` generation.

### 🧠 Smart VRAM Management (Auto-Unload)
To support running multiple heavy models on consumer hardware (e.g., RTX 3090/4090), AI-Hub implements **Automatic Model Unloading**:
- Models are **lazy-loaded** only when an API request is received.
- If a model remains idle (default: 60 seconds), it is **automatically unloaded** from VRAM to make room for other services.
- This allows you to host all services even if your total VRAM is less than the sum of all models.

### 📊 Dashboard UI
The Orchestrator includes a built-in real-time dashboard at `http://<device-ip>:9000/dashboard/`:
- **Live service status** — see which services are Online, Idle (model unloaded), or Offline
- **VRAM monitoring** — per-service GPU memory usage
- **Quick Tools** — Health Check All, Unload All, Start All Servers buttons
- **Service Actions** — Unload individual services directly from the UI

### 🔧 Management API
The Orchestrator exposes management endpoints for programmatic control:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/v1/hub/services` | GET | Aggregated health status of all services |
| `/v1/hub/unload/{service}` | POST | Unload a specific service's models from VRAM |
| `/v1/hub/launch-all` | POST | Start all backend servers via `unified_server.py` |

All backend services also expose consistent endpoints:
- `GET /health` — Returns `{"status": "running", "model_loaded": true/false}`
- `POST /v1/internal/unload` — Manually unload model from VRAM

### Overall Recommendation
- **Minimum**: 8GB VRAM (Supports TTS and Music comfortably).
- **Recommended**: 24GB VRAM (Supports TTS, Music, and ASR simultaneously).
- **Vision**: ~16GB VRAM for Z-Image with CPU offload enabled.
- **Storage**: ~100GB+ for model weights (stored in `D:\hf_models` by default).

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- **uv** (Recommended for fast dependency management)
- **Git**

### Installation
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/mostafaspace/ai-hub.git
    cd ai-hub
    ```

2.  **Initialize Submodules** (If applicable):
    ```bash
    git submodule update --init --recursive
    ```

### Running the Hub
The easiest way to start all services is using the unified launcher:

**Windows**:
```cmd
run_server.bat
```
- **Option [U]**: Highly recommended! Starts all servers (TTS, Music, ASR, Vision, Video) in a **single window** with unified, color-coded logging.
- **Option [A]**: Starts all servers in separate popup windows with auto-restart.

### Verifying Installation
Once the servers are up, you can run the unified health check script:
```bash
python tests/test_all_servers.py
```

---

## Practical Studio Tools

The Orchestrator now exposes a **Practical Studio** toolkit at `http://<device-ip>:9000/v1/studio/...` for additive, user-facing media workflows that do **not** modify the existing generation APIs:

- **Project Workspaces** - Save project notes, assets, and a lightweight timeline manifest.
- **Character Consistency Packs** - Reuse prompt prefixes/suffixes, negative prompts, default voice settings, and an optional attached reference photo.
- **Upload-First Asset Flow** - Upload media once, attach it to a project or character pack, and reuse the hosted URL across Studio jobs.
- **Timeline Editor Lite + Safe Render** - Store clip/audio/subtitle layout, build a render plan, and create a new rendered output without mutating source media.
- **Auto-Captions & Burn-In** - Generate approximate `.srt` subtitles via ASR and optionally burn them into a copied video output.
- **Thumbnail / Contact Sheet Generation** - Produce preview assets for videos with FFmpeg.
- **Output Format Profiles** - Apply reusable presets such as `youtube_short`, `discord_clip`, `podcast_mp3`, and `whatsapp_voice`.
- **Voice Audition Mode** - Generate the same script in multiple voices and compare the outputs side-by-side.
- **Prompt Compare Mode** - Run multiple prompt variants through the Vision backend and collect all outputs in one task.
- **Director-to-Project Runs** - Launch the existing Director workflow behind an async Studio task and automatically attach the final video to a project.
- **Project Export / Import** - Package a workspace plus bundled local assets into a zip, then re-import it on another machine.
- **Task Durability** - Studio tasks persist to disk, survive restarts as `interrupted`, and can be cancelled, resumed, or retried.
- **Webhooks & Observability** - Optional per-task webhook delivery plus Studio metrics for counts, durations, failures, and storage usage.

### Studio Task Pattern

Long-running studio operations follow the hub's async polling pattern:

1. `POST /v1/studio/...` to submit a job.
2. Receive a `task_id` immediately.
3. Poll `GET /v1/studio/tasks/{task_id}` until status becomes `completed`, `failed`, `cancelled`, or `interrupted`.
4. Use `POST /v1/studio/tasks/{task_id}/cancel` or `POST /v1/studio/tasks/{task_id}/resume` when needed.
5. Download any returned URLs from `/outputs/practical/...`.

### Agent Request Guidance

For remote agents and Windows shells, prefer writing JSON bodies to a temporary `payload.json` file and using `curl -d @payload.json` instead of inline JSON strings. This avoids quote-escaping issues and matches the development protocol guidance.

---

## 📅 Roadmap & Future Integrations

AI-Hub is a living project with significant expansions planned. We are committed to making this the ultimate hub for open source AI services. 🛸

| Feature | Description | Status |
| :--- | :--- | :--- |
| **Vision API** | Image generation with Z-Image (text-to-image). | ✅ Done |
| **Dashboard UI** | Real-time dashboard to monitor and manage all services. | ✅ Done |
| **Omni-Chat** | Unified chat interface for LLM interactions. | 📅 Planned |
| **Agent Workspace** | Infrastructure for autonomous AI agents. | 💡 Researching |

---

## 🏗️ Repository Structure

```text
ai-hub/
├── ACE-Step-1.5/      # Music generation service (Port 8001)
├── Qwen3-TTS/         # Text-to-speech service (Port 8000)
├── Qwen3-ASR/         # Speech-to-text service (Port 8002)
├── Z-Image/           # Image generation service (Port 8003)
├── LTX-2-Video/       # Video generation service (Port 8004)
├── orchestrator/      # Orchestrator + Dashboard (Port 9000)
├── openclaw_skills/   # API skills for remote AI agents
├── tests/             # All test scripts
├── run_server.bat     # Centralized launcher menu
└── unified_server.py  # Unified All-in-One process manager
```

## 🤝 Contributing

We're looking forward to growing the AI-Hub ecosystem! Feel free to open issues or submit pull requests as we continue to build the future of open source AI.

---

*Made with ❤️ by the AI-Hub Team.*
