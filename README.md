# AI-Hub 🚀

Welcome to **AI-Hub**, the unified gateway for open source AI's suite of powerful generative models and services. This repository serves as a central orchestrator, providing a single point of entry to state-of-the-art Text-to-Speech (TTS) and Music Generation capabilities.

## 🌟 Overview

AI-Hub is designed to be a modular, scalable, and easy-to-use ecosystem for AI-powered creativity. Whether you're looking to generate expressive speech or compose unique musical tracks, AI-Hub brings these services together under one roof with a unified management system.

## 📜 Development Protocol
Please refer to [DEVELOPMENT_PROTOCOL.md](DEVELOPMENT_PROTOCOL.md) for mandatory rules on adding features and updating the launcher.

## 🛠️ Current Services

### 🎙️ Qwen3 TTS
High-fidelity text-to-speech generation powered by the latest Qwen models.
- **Path**: [`/Qwen3-TTS`](./Qwen3-TTS)
- **Primary Endpoint**: `http://<device-ip>:8000/v1/audio/speech`
- **Features**: Multi-voice support, high-quality MP3 output, and OpenAI-compatible API.

### 🎵 ACE-Step Music
Advanced music generation service utilizing the ACE-Step-1.5 model.
- **Path**: [`/ACE-Step-1.5`](./ACE-Step-1.5)
- **Primary Endpoint**: `http://<device-ip>:8001/release_task`
- **Features**: Prompt-based music generation, flexible duration settings, and asynchronous task processing.

### 🎧 Qwen3 ASR
Automatic Speech Recognition and Audio Intelligence.
- **Path**: [`/Qwen3-ASR`](./Qwen3-ASR)
- **Primary Endpoint**: `http://<device-ip>:8002/v1/audio/transcriptions`
- **Features**: Speech-to-Text transcription and Audio Analysis/Chat using Qwen2-Audio-7B.

---

## 💻 System Requirements

AI-Hub runs powerful generative models locally. Below are the estimated VRAM requirements for each service:

| Service | Model | Precision | Estimated VRAM |
| :--- | :--- | :--- | :--- |
| **Qwen3 TTS** | 1.7B Custom/Design/Base | float16 | ~4-6 GB |
| **ACE-Step Music** | 1.5 Turbo (DiT + 1.7B LM) | float16 | ~8-12 GB* |
| **Qwen3 ASR** | Qwen2-Audio-7B | float16 | ~16-18 GB |

> [!NOTE]
> **ACE-Step Music** scales with available VRAM. It can run in "DiT-only" mode on as little as **4GB VRAM**, or use larger Language Models (up to 4B) if 16GB+ is available.

### 🧠 Smart VRAM Management (Auto-Unload)
To support running multiple heavy models on consumer hardware (e.g., RTX 3090/4090), AI-Hub implements **Automatic Model Unloading**:
- Models are **lazy-loaded** only when an API request is received.
- If a model remains idle (default: 60 seconds), it is **automatically unloaded** from VRAM to make room for other services.
- This allows you to host all services even if your total VRAM is less than the sum of all models.

### Overall Recommendation
- **Minimum**: 8GB VRAM (Supports TTS and Music comfortably).
- **Recommended**: 24GB VRAM (Supports all models including 7B ASR simultaneously).
- **Storage**: ~50GB+ for model weights (stored in `D:\hf_models` by default).

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- **uv** (Recommended for fast dependency management)
- **Git**

### Installation
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/[your-username]/ai-hub.git
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
- **Option [U]**: Highly recommended! Starts all servers (TTS, Music, ASR) in a **single window** with unified, color-coded logging.
- **Option [A]**: Starts all servers in separate popup windows with auto-restart.

### Verifying Installation
Once the servers are up, you can run the unified health check script:
```bash
python test_all_servers.py
```

---

## 📅 Roadmap & Future Integrations

AI-Hub is a living project with significant expansions planned. We are committed to making this the ultimate hub for open source AI services. 🛸

| Feature | Description | Status |
| :--- | :--- | :--- |
| **Vision API** | Image analysis and generation integration. | 🔜 Coming Soon |
| **Omni-Chat** | Unified chat interface for LLM interactions. | 📅 Planned |
| **Agent Workspace** | Infrastructure for autonomous AI agents. | 💡 Researching |
| **Unified UI** | A single dashboard to monitor and interact with all hub services. | 🛤️ On Roadmap |

---

## 🏗️ Repository Structure

```text
ai-hub/
├── ACE-Step-1.5/      # Music generation service (Port 8001)
├── Qwen3-TTS/         # Text-to-speech service (Port 8000)
├── Qwen3-ASR/         # Speech-to-text service (Port 8002)
├── openclaw_skills/   # Extensible skills for AI agents
├── run_server.bat     # Centralized launcher menu
├── unified_server.py  # Unified All-in-One process manager
└── test_all_servers.py # Unified health check script
```

## 🤝 Contributing

We're looking forward to growing the AI-Hub ecosystem! Feel free to open issues or submit pull requests as we continue to build the future of open source AI.

---

*Made with ❤️ by the AI-Hub Team.*
