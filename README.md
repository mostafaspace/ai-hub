# AI-Hub 🚀

Welcome to **AI-Hub**, the unified gateway for Antigravity AI's suite of powerful generative models and services. This repository serves as a central orchestrator, providing a single point of entry to state-of-the-art Text-to-Speech (TTS) and Music Generation capabilities.

## 🌟 Overview

AI-Hub is designed to be a modular, scalable, and easy-to-use ecosystem for AI-powered creativity. Whether you're looking to generate expressive speech or compose unique musical tracks, AI-Hub brings these services together under one roof with a unified management system.

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
*(This script will handle process management, port allocation, and starting the separate service windows.)*

### Verifying Installation
Once the servers are up, you can run the unified health check script:
```bash
python test_all_servers.py
```

---

## 📅 Roadmap & Future Integrations

AI-Hub is a living project with significant expansions planned. We are committed to making this the ultimate hub for Antigravity AI services. 🛸

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
├── ACE-Step-1.5/     # Music generation service
├── Qwen3-TTS/        # Text-to-speech service
├── openclaw_skills/   # Extensible skills for AI agents
├── run_server.bat    # Unified server launcher
└── test_all_servers.py # Unified health check script
```

## 🤝 Contributing

We're looking forward to growing the AI-Hub ecosystem! Feel free to open issues or submit pull requests as we continue to build the future of Antigravity AI.

---

*Made with ❤️ by the Antigravity AI Team.*
