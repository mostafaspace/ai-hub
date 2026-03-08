import os
import sys
import yaml
import httpx
import uuid
import asyncio
import time
from datetime import datetime, timezone
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

# Add root directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from orchestrator.media_utils import mux_video_and_audio
from orchestrator.utils_api import utils_router
from orchestrator.studio_api import studio_router

try:
    from api_utils import GracefulJSONRoute
except ImportError:
    GracefulJSONRoute = None

# --- Load Registry ---
REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "models.yaml")
BACKENDS = {}

def load_registry():
    global BACKENDS
    try:
        with open(REGISTRY_PATH, "r") as f:
            data = yaml.safe_load(f)
            BACKENDS = data.get("backends", {})
            print(f"Orchestrator Registry Loaded: {BACKENDS}")
    except Exception as e:
        print(f"Failed to load models.yaml: {e}")

# HTTPX Client for connection pooling
http_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    load_registry()
    # High timeout since generation takes time (None = Infinite because LTX-2 takes >5mins)
    http_client = httpx.AsyncClient(timeout=None)
    yield
    await http_client.aclose()

app = FastAPI(title="AI-Hub Orchestrator", version="1.0", lifespan=lifespan)

if GracefulJSONRoute:
    app.router.route_class = GracefulJSONRoute

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Utility APIs
app.include_router(utils_router)
app.include_router(studio_router)

# --- Mount Dashboard Static Files ---
DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "dashboard")
if os.path.isdir(DASHBOARD_DIR):
    app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")
    print(f"Dashboard mounted from {DASHBOARD_DIR}")

# --- Mount Outputs Directory ---
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "director_outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

# --- Task Tracking (in-memory) ---
recent_tasks = []  # List of dicts: {task_id, workflow, status, started_at, finished_at}
MAX_TRACKED_TASKS = 50

def record_task(task_id, workflow, status, started_at=None):
    """Record or update a task in the recent tasks list."""
    for t in recent_tasks:
        if t["task_id"] == task_id:
            t["status"] = status
            if status in ("COMPLETED", "FAILED"):
                t["finished_at"] = time.time()
            return
    recent_tasks.insert(0, {
        "task_id": task_id,
        "workflow": workflow,
        "status": status,
        "started_at": started_at or time.time(),
        "finished_at": None,
    })
    if len(recent_tasks) > MAX_TRACKED_TASKS:
        recent_tasks.pop()


# --- MACRO SKILLS: The Content Director ---

class DirectorRequest(BaseModel):
    image_prompt: str
    voiceover_text: str
    voice: str = "Vivian"

@app.post("/v1/workflows/director")
async def content_director(req: DirectorRequest, request: Request):
    """
    The Director Macro-Skill:
    1. Generates Image via Z-Image
    2. Generates Audio via Qwen3-TTS
    3. Transforms Image to Video via LTX-2-Video
    4. Stitches the video and audio together with FFmpeg.
    """
    if "vision" not in BACKENDS or "tts" not in BACKENDS or "video" not in BACKENDS:
        raise HTTPException(status_code=503, detail="Required backends (vision, tts, video) are not all configured in the registry.")

    # Create a unique working directory in the orchestrator folder for this task
    task_id = str(uuid.uuid4())
    work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "director_outputs", task_id))
    os.makedirs(work_dir, exist_ok=True)
    
    # Fast polling interval and maximum deadline to prevent infinite hangs
    poll_interval = 2.0
    POLL_DEADLINE = 30 * 60  # 30 minutes max per generation step
    
    print(f"[Director] Started Workflow {task_id}")
    record_task(task_id, "content_director", "RUNNING")

    try:
        # NOTE: TTS is done first (sequential) because it's a cold-boot and fast relative to video.
        print("[Director] Requesting TTS and Vision generation...")
        
        vision_url = f"{BACKENDS['vision']}/v1/images/async_generate"
        vision_payload = {
            "prompt": req.image_prompt,
            "cfg_normalization": True
        }
        
        tts_url = f"{BACKENDS['tts']}/v1/audio/speech"
        tts_payload = {
            "input": req.voiceover_text,
            "voice": req.voice
        }

        # --- STEP 1: Generate Audio ---
        record_task(task_id, "content_director", "RUNNING (1/4: Generating Text-to-Speech)")
        print(f"[Director] Sending TTS Request -> {tts_url}")
        # Qwen3-TTS takes a long time to load its 1.7B parameter model on cold boot
        tts_resp = await http_client.post(tts_url, json=tts_payload, timeout=None)
        
        if tts_resp.status_code != 200:
            raise Exception(f"TTS backend failed: {tts_resp.text}")
            
        audio_path = os.path.join(work_dir, "voiceover.wav")
        with open(audio_path, 'wb') as f:
            f.write(tts_resp.content)
        print(f"[Director] Audio saved -> {audio_path}")

        # --- STEP 2: Generate Image ---
        record_task(task_id, "content_director", "RUNNING (2/4: Generating Base Image)")
        print(f"[Director] Sending Vision Request -> {vision_url}")
        vision_resp = await http_client.post(vision_url, json=vision_payload)
        
        # Check Vision result (Async typically, need to poll)
        if vision_resp.status_code != 200:
            raise Exception(f"Vision backend initialization failed: {vision_resp.text}")
            
        vision_task_id = vision_resp.json().get("task_id")
        print(f"[Director] Vision task started: {vision_task_id}. Polling...")
        
        # Poll Vision with a deadline to avoid infinite hang
        image_url = None
        deadline = asyncio.get_event_loop().time() + POLL_DEADLINE
        while True:
            if asyncio.get_event_loop().time() > deadline:
                raise Exception(f"Vision task {vision_task_id} timed out after {POLL_DEADLINE}s")
            v_poll = await http_client.get(f"{BACKENDS['vision']}/v1/images/tasks/{vision_task_id}")
            v_data = v_poll.json()
            if v_data["status"] == "COMPLETED" or v_data["status"] == "completed":
                # Z-Image returns `data` array containing dicts with `url`
                image_url = v_data["data"][0]["url"]
                # Convert relative URL to full if needed
                if image_url.startswith("/"):
                    image_url = f"{BACKENDS['vision']}{image_url}"
                break
            elif v_data["status"] == "FAILED" or v_data["status"] == "failed":
                raise Exception(f"Vision task failed: {v_data.get('error')}")
            await asyncio.sleep(poll_interval)
            
        # Download the generated image to our work dir
        img_download = await http_client.get(image_url)
        image_path = os.path.join(work_dir, "base_image.jpg")
        with open(image_path, 'wb') as f:
            f.write(img_download.content)
        print(f"[Director] Image completed and saved to {image_path}.")
        
        # --- STEP 3: Image-to-Video ---
        record_task(task_id, "content_director", "RUNNING (3/4: Animating Image to Video)")
        print("[Director] Requesting Video generation from Image...")
        video_url = f"{BACKENDS['video']}/v1/video/async_i2v"
        
        # LTX-2-Video async_i2v explicitly requires a multipart/form boundary containing an `UploadFile` and `Form` fields.
        with open(image_path, "rb") as img_file:
            vid_files = {
                "image": ("base_image.jpg", img_file, "image/jpeg")
            }
            vid_data = {
                "prompt": req.image_prompt,
                "seed": "-1"
            }
            vid_init = await http_client.post(video_url, data=vid_data, files=vid_files)
        if vid_init.status_code != 200:
            raise Exception(f"Video backend initialization failed: {vid_init.text}")
            
        vid_task_id = vid_init.json().get("task_id")
        print(f"[Director] Video task started: {vid_task_id}. Polling...")
        
        # Poll Video with a deadline — LTX-2 can take many minutes
        # LTX video tasks return a flat dict: {"status": "completed", "url": "http://..."}
        raw_video_path = None
        deadline = asyncio.get_event_loop().time() + POLL_DEADLINE
        while True:
            if asyncio.get_event_loop().time() > deadline:
                raise Exception(f"Video task {vid_task_id} timed out after {POLL_DEADLINE}s")
            vid_poll = await http_client.get(f"{BACKENDS['video']}/v1/video/tasks/{vid_task_id}")
            vid_data = vid_poll.json()
            if vid_data["status"] == "completed":
                # LTX-2 Video returns a flat dict with a "url" key (served via /outputs/{filename}).
                # There is no nested "output" key and no "local_path" key in the current server impl.
                video_result_url = vid_data.get("url")
                if not video_result_url:
                    raise Exception(f"Video task completed but returned no URL: {vid_data}")
                # Build a fully-qualified URL if relative
                target_url = video_result_url if video_result_url.startswith("http") else f"{BACKENDS['video']}{video_result_url}"
                vid_download = await http_client.get(target_url)
                raw_video_path = os.path.join(work_dir, "raw_video.mp4")
                with open(raw_video_path, 'wb') as f:
                    f.write(vid_download.content)
                break
            elif vid_data["status"] == "failed":
                raise Exception(f"Video task failed: {vid_data.get('error')}")
            await asyncio.sleep(poll_interval)
            
        print(f"[Director] Video completed at {raw_video_path}.")

        # --- STEP 4: Async FFmpeg Muxing ---
        record_task(task_id, "content_director", "RUNNING (4/4: Stitching Final Render)")
        print("[Director] Stitching Audio and Video...")
        loop = asyncio.get_running_loop()
        final_video_path = os.path.join(work_dir, "final_director_cut.mp4")
        
        # Run the blocking subprocess in an executor
        success = await loop.run_in_executor(None, mux_video_and_audio, raw_video_path, audio_path, final_video_path)
        
        if not success:
            raise Exception("FFmpeg failed to mux the final video.")
            
        print(f"[Director] Workflow complete! {final_video_path}")
        
        record_task(task_id, "content_director", "COMPLETED")
        
        host = getattr(config, "HOST", "127.0.0.1")
        # default to 0.0.0.0 mapping to localhost for URLs if needed, but best effort
        if host == "0.0.0.0":
            host = "127.0.0.1"
            
        port = getattr(config, "ORCHESTRATOR_PORT", 9000)
        output_url = f"http://{host}:{port}/outputs/{task_id}/final_director_cut.mp4"
        
        return JSONResponse(status_code=200, content={
            "task_id": task_id,
            "status": "COMPLETED",
            "message": "Content Director successfully assembled the video.",
            "output_url": output_url
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        record_task(task_id, "content_director", "FAILED")
        return JSONResponse(status_code=500, content={"error": str(e), "task_id": task_id})


# --- Proxy Logic ---
async def proxy_request(request: Request, backend_url: str):
    url = f"{backend_url}{request.url.path}"
    if request.url.query:
        url += f"?{request.url.query}"
        
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)
    
    try:
        req = http_client.build_request(
            method=request.method,
            url=url,
            headers=headers,
            content=body
        )
        proxy_resp = await http_client.send(req, stream=True)
        
        async def stream_generator():
            async for chunk in proxy_resp.aiter_bytes():
                yield chunk
                
        return StreamingResponse(
            stream_generator(),
            status_code=proxy_resp.status_code,
            headers={k: v for k, v in proxy_resp.headers.items() if k.lower() not in ('content-encoding', 'content-length', 'transfer-encoding')}
        )

    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Bad Gateway: Error communicating with backend model service: {e}")

# --- Routing ---

# Audio (TTS)
@app.post("/v1/audio/speech")
@app.get("/v1/audio/speech")
@app.post("/v1/audio/speech/stream")
async def route_tts(request: Request):
    if "tts" not in BACKENDS:
        raise HTTPException(status_code=503, detail="TTS Backend not configured in registry.")
    return await proxy_request(request, BACKENDS["tts"])

# Audio (Voice Clone/Design specific extensions)
@app.post("/v1/audio/voice_design")
@app.post("/v1/audio/voice_clone")
async def route_tts_extensions(request: Request):
    if "tts" not in BACKENDS:
        raise HTTPException(status_code=503, detail="TTS Backend not configured.")
    return await proxy_request(request, BACKENDS["tts"])

# Audio (ASR)
@app.post("/v1/audio/transcriptions")
async def route_asr(request: Request):
    if "asr" not in BACKENDS:
        raise HTTPException(status_code=503, detail="ASR Backend not configured.")
    return await proxy_request(request, BACKENDS["asr"])

# Vision (Images)
@app.post("/v1/images/generations")
@app.post("/v1/images/async_generate")
@app.post("/v1/images/async_edit")
@app.get("/v1/images/tasks/{task_id}")
async def route_vision(request: Request, task_id: str = None):
    if "vision" not in BACKENDS:
        raise HTTPException(status_code=503, detail="Vision Backend not configured.")
    return await proxy_request(request, BACKENDS["vision"])

# Video
@app.post("/v1/video/generations")
@app.post("/v1/video/async_t2v")
@app.post("/v1/video/async_i2v")
@app.get("/v1/video/tasks/{task_id}")
async def route_video(request: Request, task_id: str = None):
    if "video" not in BACKENDS:
        raise HTTPException(status_code=503, detail="Video Backend not configured.")
    return await proxy_request(request, BACKENDS["video"])

# Chat Completions (Multiplexing)
# The Chat Completions endpoint is tricky because multiple backends might use it (ASR uses it for Qwen audio chat, potentially future text LLMs).
# For now, default route it to ASR as it's the primary provider of chat completions currently in the hub.
@app.post("/v1/chat/completions")
async def route_chat(request: Request):
    # Could dynamically inspect request JSON here later to route based on "model" parameter.
    if "asr" not in BACKENDS:
        raise HTTPException(status_code=503, detail="ASR Backend not configured.")
    return await proxy_request(request, BACKENDS["asr"])

# --- DASHBOARD API ENDPOINTS ---

def _time_ago(ts):
    """Convert a unix timestamp to a human-readable 'X ago' string."""
    if ts is None:
        return ""
    delta = int(time.time() - ts)
    if delta < 60:
        return f"{delta}s ago"
    elif delta < 3600:
        return f"{delta // 60}m ago"
    elif delta < 86400:
        return f"{delta // 3600}h ago"
    return f"{delta // 86400}d ago"

@app.get("/v1/hub/services")
async def hub_services():
    """
    Aggregated health check: polls all backends in parallel
    and returns a unified status object for the dashboard.
    """
    results = {}

    async def check_backend(key, url):
        try:
            resp = await http_client.get(f"{url}/health", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                # Determine if model is loaded (varies by backend)
                model_loaded = data.get("model_loaded",
                    data.get("vram_loaded",
                    data.get("t2i_loaded", True)))
                results[key] = {
                    "status": "online",
                    "model_loaded": model_loaded,
                    "health_data": data,
                }
            else:
                results[key] = {"status": "offline"}
        except Exception:
            results[key] = {"status": "offline"}

        # Try to get VRAM info from /v1/internal/status if available
        if results[key]["status"] == "online":
            try:
                sr = await http_client.get(f"{url}/v1/internal/status", timeout=2.0)
                if sr.status_code == 200:
                    sdata = sr.json()
                    if "vram_used_gb" in sdata:
                        results[key]["vram_used_gb"] = sdata["vram_used_gb"]
                    if "vram_total_gb" in sdata:
                        results[key]["vram_total_gb"] = sdata["vram_total_gb"]
            except Exception:
                pass

    tasks = [check_backend(key, url) for key, url in BACKENDS.items()]
    await asyncio.gather(*tasks)

    return {"services": results}


@app.get("/v1/hub/tasks")
async def hub_tasks():
    """
    Returns recent workflow tasks for the dashboard task feed.
    """
    formatted = []
    for t in recent_tasks:
        duration = ""
        if t["finished_at"] and t["started_at"]:
            secs = int(t["finished_at"] - t["started_at"])
            mins = secs // 60
            duration = f"{mins}m {secs % 60}s" if mins > 0 else f"{secs}s"
        formatted.append({
            "task_id": t["task_id"],
            "workflow": t["workflow"],
            "status": t["status"],
            "duration": duration,
            "time_ago": _time_ago(t["started_at"]),
        })
    return {"tasks": formatted}


@app.post("/v1/hub/unload/{service}")
async def hub_unload(service: str):
    """
    Send an unload command to a backend service.
    Tries POST /v1/internal/unload, falls back to message.
    """
    if service not in BACKENDS:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service}")

    url = f"{BACKENDS[service]}/v1/internal/unload"
    try:
        resp = await http_client.post(url, timeout=5.0)
        if resp.status_code == 200:
            return {"message": f"{service} model unloaded successfully."}
        else:
            return {"message": f"{service} responded with status {resp.status_code}. Endpoint may not be supported yet."}
    except Exception as e:
        return {"message": f"Could not reach {service} unload endpoint: {str(e)}"}


# --- Track launched subprocess ---
_launched_process = None

@app.post("/v1/hub/launch-all")
async def hub_launch_all():
    """
    Launch all backend servers via unified_server.py as a subprocess.
    The orchestrator itself is already running, so this only starts the backends.
    """
    global _launched_process
    import subprocess as sp

    # Check if already launched
    if _launched_process is not None and _launched_process.poll() is None:
        return {"message": "Servers are already launching / running."}

    unified_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "unified_server.py"))
    if not os.path.exists(unified_script):
        raise HTTPException(status_code=404, detail="unified_server.py not found.")

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        _launched_process = sp.Popen(
            [sys.executable, unified_script],
            cwd=os.path.dirname(unified_script),
            env=env,
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
        )
        return {"message": f"unified_server.py launched (PID {_launched_process.pid}). Services will come online shortly."}
    except Exception as e:
        return {"message": f"Failed to launch: {str(e)}"}


# Generic pass-through for anything else
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(request: Request, path: str):
    # Root: redirect to dashboard
    if path == "" or path == "/":
        return RedirectResponse(url="/dashboard/", status_code=302)

    if path == "health":
        return {"status": "ok", "message": "AI-Hub Orchestrator is running.", "registry": BACKENDS}

    raise HTTPException(status_code=404, detail=f"Orchestrator route not defined for path: {path}")

if __name__ == "__main__":
    port = getattr(config, "ORCHESTRATOR_PORT", 9000)
    print(f"Starting AI-Hub Orchestrator on {config.HOST}:{port}")
    uvicorn.run(app, host=config.HOST, port=port)
