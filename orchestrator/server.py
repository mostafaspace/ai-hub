import os
import sys
import yaml
import httpx
import uuid
import asyncio
import time
import json
from datetime import datetime, timezone
from urllib.parse import urlparse
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.background import BackgroundTask
import uvicorn
from contextlib import asynccontextmanager

# Add root directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from typing import Optional, List
from orchestrator.media_utils import mux_video_and_audio, get_media_duration, loop_video_to_duration, mix_audio_files
from orchestrator.studio_media import extract_audio_for_transcription, extract_thumbnail, detect_duration
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
    async with httpx.AsyncClient(timeout=None) as client:
        http_client = client
        yield
    http_client = None

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

def _public_host() -> str:
    host = getattr(config, "PUBLIC_HOST", None) or getattr(config, "HOST", "127.0.0.1")
    if host == "0.0.0.0":
        return "127.0.0.1"
    return host
def _orchestrator_base_url() -> str:
    return f"http://{_public_host()}:{getattr(config, 'ORCHESTRATOR_PORT', 9000)}"
def _rewrite_backend_url_to_hub(url: str, backend_name: str) -> str:
    if not url:
        return url
    backend_url = BACKENDS.get(backend_name)
    if not backend_url:
        return url
    if url.startswith("/"):
        return f"{_orchestrator_base_url()}{url}"
    parsed_url = urlparse(url)
    parsed_backend = urlparse(backend_url)
    if parsed_url.scheme not in {"http", "https"}:
        return url
    if parsed_url.netloc != parsed_backend.netloc:
        return url
    rewritten = f"{_orchestrator_base_url()}{parsed_url.path}"
    if parsed_url.query:
        rewritten += f"?{parsed_url.query}"
    return rewritten

def _rewrite_backend_output_url(url: str, backend_name: str, hub_output_prefix: str) -> str:
    if not url:
        return url

    backend_url = BACKENDS.get(backend_name)
    if not backend_url:
        return url

    if url.startswith("/outputs/"):
        rel_path = url[len("/outputs/"):].lstrip("/")
    else:
        parsed_url = urlparse(url)
        parsed_backend = urlparse(backend_url)
        if parsed_url.scheme not in {"http", "https"}:
            return url
        if parsed_url.netloc != parsed_backend.netloc or not parsed_url.path.startswith("/outputs/"):
            return url
        rel_path = parsed_url.path[len("/outputs/"):].lstrip("/")

    return f"{_orchestrator_base_url()}{hub_output_prefix}/{rel_path}"


async def _proxy_backend_output(request: Request, backend_name: str, output_path: str):
    if backend_name not in BACKENDS:
        raise HTTPException(status_code=503, detail=f"{backend_name.title()} Backend not configured.")

    url = f"{BACKENDS[backend_name]}/outputs/{output_path.lstrip('/')}"
    headers = dict(request.headers)
    headers.pop("host", None)

    try:
        req = http_client.build_request(method="GET", url=url, headers=headers)
        proxy_resp = await http_client.send(req, stream=True)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Bad Gateway: Error communicating with backend model service: {e}")

    async def stream_generator():
        async for chunk in proxy_resp.aiter_bytes():
            yield chunk

    filtered_headers = {
        k: v for k, v in proxy_resp.headers.items()
        if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")
    }

    return StreamingResponse(
        stream_generator(),
        status_code=proxy_resp.status_code,
        headers=filtered_headers,
        background=BackgroundTask(proxy_resp.aclose),
    )

# --- MACRO SKILLS: The Content Director ---

class DirectorRequest(BaseModel):
    image_prompt: str
    voiceover_text: str
    voice: str = "Vivian"
    music_prompt: Optional[str] = None
    music_volume: float = 0.2

class AuditRequest(BaseModel):
    media_url: str
    prompt_context: str = ""
    check_audio: bool = True
    check_visual: bool = True

@app.post("/v1/workflows/director")
async def content_director(req: DirectorRequest, request: Request):
    """
    The Director Macro-Skill (Async):
    1. Returns a task_id immediately.
    2. Runs the generation pipeline in the background.
    """
    if "vision" not in BACKENDS or "tts" not in BACKENDS or "video" not in BACKENDS:
        raise HTTPException(status_code=503, detail="Required backends (vision, tts, video) are not all configured in the registry.")

    task_id = str(uuid.uuid4())
    record_task(task_id, "content_director", "QUEUED", started_at=time.time())
    
    background_task = BackgroundTask(run_director_task, task_id, req)
    return JSONResponse(status_code=202, content={"task_id": task_id, "status": "QUEUED"}, background=background_task)


async def run_director_task(task_id: str, req: DirectorRequest):
    """Background task for the Director workflow."""
    work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "director_outputs", task_id))
    os.makedirs(work_dir, exist_ok=True)
    
    # Fast polling interval and maximum deadline
    poll_interval = 2.0
    POLL_DEADLINE = 30 * 60
    
    print(f"[Director] Started Background Workflow {task_id}")
    record_task(task_id, "content_director", "RUNNING")

    try:
        # --- STEP 1: Generate Audio (Voiceover) ---
        record_task(task_id, "content_director", "RUNNING (1/5: Generating Voiceover)")
        tts_url = f"{BACKENDS['tts']}/v1/audio/speech"
        tts_payload = {"input": req.voiceover_text, "voice": req.voice}
        
        print(f"[Director] Sending TTS Request -> {tts_url}")
        tts_resp = await http_client.post(tts_url, json=tts_payload, timeout=httpx.Timeout(1800.0, connect=30.0))
        if tts_resp.status_code != 200:
            raise Exception(f"TTS backend failed: {tts_resp.text}")
            
        audio_path = os.path.join(work_dir, "voiceover.wav")
        with open(audio_path, 'wb') as f:
            f.write(tts_resp.content)
        
        voice_duration = await asyncio.to_thread(get_media_duration, audio_path)
        print(f"[Director] Voiceover duration: {voice_duration}s")

        # --- STEP 2: Generate Music (Optional) ---
        music_path = None
        if req.music_prompt and "music" in BACKENDS:
            record_task(task_id, "content_director", "RUNNING (2/5: Generating Soundtrack)")
            music_url = f"{BACKENDS['music']}/v1/audio/async_generations"
            music_payload = {
                "prompt": req.music_prompt,
                "audio_duration": voice_duration + 2.0, # Add a small buffer
                "audio_format": "mp3"
            }
            print(f"[Director] Sending Music Request -> {music_url}")
            music_init = await http_client.post(music_url, json=music_payload, timeout=httpx.Timeout(60.0, connect=10.0))
            if music_init.status_code == 200:
                music_task_id = music_init.json().get("task_id")
                deadline = asyncio.get_event_loop().time() + POLL_DEADLINE
                while True:
                    if asyncio.get_event_loop().time() > deadline:
                        print("[Director] Music task timed out, proceeding without music.")
                        break
                    m_poll = await http_client.get(f"{BACKENDS['music']}/v1/audio/tasks/{music_task_id}")
                    m_data = m_poll.json()
                    if m_data["status"] == "completed":
                        music_result_url = m_data.get("data", [{}])[0].get("url")
                        if music_result_url:
                            target_url = music_result_url if music_result_url.startswith("http") else f"{BACKENDS['music']}{music_result_url}"
                            m_download = await http_client.get(target_url)
                            music_path = os.path.join(work_dir, "music_raw.mp3")
                            with open(music_path, 'wb') as f:
                                f.write(m_download.content)
                        break
                    elif m_data["status"] == "failed":
                        print(f"[Director] Music failed: {m_data.get('error')}, skip music.")
                        break
                    await asyncio.sleep(poll_interval)
            else:
                print(f"[Director] Music init failed ({music_init.status_code}), skip music.")

        # --- STEP 3: Generate Visuals ---
        record_task(task_id, "content_director", "RUNNING (3/5: Generating Visuals)")
        vision_url = f"{BACKENDS['vision']}/v1/images/async_generate"
        vision_payload = {"prompt": req.image_prompt}
        
        print(f"[Director] Sending Vision Request -> {vision_url}")
        vision_resp = await http_client.post(vision_url, json=vision_payload, timeout=httpx.Timeout(1800.0, connect=30.0))
        if vision_resp.status_code != 200:
            raise Exception(f"Vision backend failed: {vision_resp.text}")
            
        vision_task_id = vision_resp.json().get("task_id")
        image_url = None
        deadline = asyncio.get_event_loop().time() + POLL_DEADLINE
        while True:
            if asyncio.get_event_loop().time() > deadline:
                raise Exception(f"Vision task {vision_task_id} timed out")
            v_poll = await http_client.get(f"{BACKENDS['vision']}/v1/images/tasks/{vision_task_id}", timeout=httpx.Timeout(30.0))
            v_data = v_poll.json()
            if v_data["status"] in ("COMPLETED", "completed"):
                image_url = v_data["data"][0]["url"]
                break
            elif v_data["status"] in ("FAILED", "failed"):
                raise Exception(f"Vision task failed: {v_data.get('error')}")
            await asyncio.sleep(poll_interval)
            
        img_download = await http_client.get(image_url, timeout=httpx.Timeout(60.0))
        image_path = os.path.join(work_dir, "base_image.jpg")
        with open(image_path, 'wb') as f:
            f.write(img_download.content)

        # Video (I2V)
        video_url = f"{BACKENDS['video']}/v1/video/async_i2v"
        vid_task_id = None
        with open(image_path, "rb") as img_file:
            vid_files = {"image": ("base_image.jpg", img_file, "image/jpeg")}
            vid_data = {"prompt": req.image_prompt, "seed": "-1"}
            vid_init = await http_client.post(video_url, data=vid_data, files=vid_files, timeout=httpx.Timeout(1800.0, connect=30.0))
            
        if vid_init.status_code != 200:
            raise Exception(f"Video backend initialization failed: {vid_init.text}")
        vid_task_id = vid_init.json().get("task_id")
        
        raw_video_path = None
        deadline = asyncio.get_event_loop().time() + POLL_DEADLINE
        while True:
            if asyncio.get_event_loop().time() > deadline:
                raise Exception(f"Video task {vid_task_id} timed out")
            vid_poll = await http_client.get(f"{BACKENDS['video']}/v1/video/tasks/{vid_task_id}")
            vid_data = vid_poll.json()
            if vid_data["status"] == "completed":
                video_result_url = vid_data.get("url")
                target_url = video_result_url if video_result_url.startswith("http") else f"{BACKENDS['video']}{video_result_url}"
                vid_download = await http_client.get(target_url)
                raw_video_path = os.path.join(work_dir, "raw_video.mp4")
                with open(raw_video_path, 'wb') as f:
                    f.write(vid_download.content)
                break
            elif vid_data["status"] == "failed":
                raise Exception(f"Video task failed: {vid_data.get('error')}")
            await asyncio.sleep(poll_interval)

        # --- STEP 4: Media Post-Processing ---
        record_task(task_id, "content_director", "RUNNING (4/5: Syncing & Mixing)")
        
        video_duration = await asyncio.to_thread(get_media_duration, raw_video_path)
        final_video_source = raw_video_path

        # Loop video if voiceover is longer
        if voice_duration > video_duration + 0.5:
            print(f"[Director] Looping video {video_duration}s to match voiceover {voice_duration}s")
            looped_path = os.path.join(work_dir, "looped_video.mp4")
            success = await asyncio.to_thread(loop_video_to_duration, raw_video_path, voice_duration, looped_path)
            if success:
                final_video_source = looped_path

        # Mix audio if music is present
        final_audio_source = audio_path
        if music_path:
            print("[Director] Mixing voiceover and background music")
            mixed_audio_path = os.path.join(work_dir, "mixed_audio.wav")
            success = await asyncio.to_thread(mix_audio_files, audio_path, music_path, mixed_audio_path, req.music_volume)
            if success:
                final_audio_source = mixed_audio_path

        # --- STEP 5: Final Mux ---
        record_task(task_id, "content_director", "RUNNING (5/5: Final Muxing)")
        final_output_path = os.path.join(work_dir, "final_director_cut.mp4")
        
        success = await asyncio.to_thread(mux_video_and_audio, final_video_source, final_audio_source, final_output_path)
        if not success:
            raise Exception("FFmpeg failed to mux the final director cut.")
            
        record_task(task_id, "content_director", "COMPLETED")
        print(f"[Director] Completed Production-Grade Workflow {task_id}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        record_task(task_id, "content_director", f"FAILED: {str(e)}")


@app.post("/v1/workflows/audit")
async def content_auditor(req: AuditRequest, request: Request):
    """
    The Auditor Macro-Skill (Async):
    1. Returns a task_id immediately.
    2. Runs the multi-modal audit in the background.
    """
    if req.check_audio and "asr" not in BACKENDS:
        raise HTTPException(status_code=503, detail="ASR backend not configured for audio audit.")
    if req.check_visual and "vision" not in BACKENDS:
        raise HTTPException(status_code=503, detail="Vision backend not configured for visual audit.")

    task_id = str(uuid.uuid4())
    record_task(task_id, "content_auditor", "STARTING", started_at=time.time())
    
    # We store the final report in a results dict or in the task record
    # For now, we'll update the 'recent_tasks' status with a JSON string of the report when done
    # or rely on the directory.
    
    background_task = BackgroundTask(run_audit_task, task_id, req)
    return JSONResponse(status_code=202, content={"task_id": task_id, "status": "STARTING"}, background=background_task)


async def run_audit_task(task_id: str, req: AuditRequest):
    import shutil
    import base64
    work_dir = os.path.abspath(os.path.join(OUTPUTS_DIR, "audit", task_id))
    os.makedirs(work_dir, exist_ok=True)

    print(f"[Auditor] Started Background Audit {task_id} for {req.media_url}")
    record_task(task_id, "content_auditor", "RUNNING")

    try:
        # Step 1: Materialize Media
        record_task(task_id, "content_auditor", "RUNNING (1/3: Materializing Media)")
        
        filename = os.path.basename(urlparse(req.media_url).path) or "input_media.mp4"
        local_media_path = os.path.join(work_dir, filename)
        
        if req.media_url.startswith("http"):
            resp = await http_client.get(req.media_url, timeout=300.0)
            resp.raise_for_status()
            with open(local_media_path, "wb") as f:
                f.write(resp.content)
        else:
            # Handle relative /outputs/ or direct relative paths
            resolve_path = req.media_url
            if resolve_path.startswith("/outputs/"):
                resolve_path = resolve_path.replace("/outputs/", "", 1)
                resolve_path = os.path.join(OUTPUTS_DIR, resolve_path)
            
            if os.path.exists(resolve_path):
                shutil.copy2(resolve_path, local_media_path)
            else:
                # Last resort fallback to basename in OUTPUTS_DIR
                alt_path = os.path.join(OUTPUTS_DIR, os.path.basename(req.media_url))
                if os.path.exists(alt_path):
                    shutil.copy2(alt_path, local_media_path)
                else:
                    raise Exception(f"Media path not found: {req.media_url} (Resolved: {resolve_path})")

        audit_report: dict = {
            "task_id": task_id,
            "media_url": req.media_url,
            "audio_audit": None,
            "visual_audit": None,
            "overall_status": "COMPLETED"
        }

        # Step 2: Audio Audit (ASR)
        if req.check_audio:
            record_task(task_id, "content_auditor", "RUNNING (2/3: Auditing Audio)")
            audio_path = os.path.join(work_dir, "extracted_audio.wav")
            loop = asyncio.get_running_loop()
            success, msg = await loop.run_in_executor(None, extract_audio_for_transcription, local_media_path, audio_path)
            
            if success:
                print(f"[Auditor] Sending to ASR at {BACKENDS['asr']}...")
                with open(audio_path, "rb") as f:
                    asr_resp = await http_client.post(
                        f"{BACKENDS['asr']}/v1/audio/transcriptions",
                        files={"file": (os.path.basename(audio_path), f, "audio/wav")},
                        data={"prompt": req.prompt_context} if req.prompt_context else {},
                        timeout=None
                    )
                if asr_resp.status_code == 200:
                    audit_report["audio_audit"] = asr_resp.json()
                else:
                    audit_report["audio_audit"] = {"error": f"ASR Backend Failed: {asr_resp.text}"}
            else:
                audit_report["audio_audit"] = {"error": f"Audio Extraction Failed: {msg}"}

        # Step 3: Visual Audit (Vision)
        if req.check_visual:
            record_task(task_id, "content_auditor", "RUNNING (3/3: Auditing Visuals)")
            duration = await asyncio.to_thread(detect_duration, local_media_path)
            timestamps = [duration * 0.1, duration * 0.5, duration * 0.9] if duration > 0 else [0]
            visual_results = []
            loop = asyncio.get_running_loop()
            
            for i, ts in enumerate(timestamps):
                frame_path = os.path.join(work_dir, f"frame_{i}.jpg")
                success, msg = await loop.run_in_executor(None, extract_thumbnail, local_media_path, frame_path, ts)
                if success:
                    try:
                        with open(frame_path, "rb") as f:
                            b64_img = base64.b64encode(f.read()).decode('utf-8')
                        
                        vision_payload = {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": "Describe this image in detail. Focus on consistency if it's from a video."},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                                    ]
                                }
                            ],
                            "max_tokens": 300
                        }
                        v_resp = await http_client.post(f"{BACKENDS['vision']}/v1/chat/completions", json=vision_payload, timeout=None)
                        if v_resp.status_code == 200:
                            v_data = v_resp.json()
                            desc = v_data["choices"][0]["message"]["content"]
                            visual_results.append({"timestamp": ts, "description": desc})
                        else:
                            visual_results.append({"timestamp": ts, "error": f"Vision backend error: {v_resp.text}"})
                    except Exception as ve:
                        visual_results.append({"timestamp": ts, "error": f"Vision analysis exception: {str(ve)}"})
            
            audit_report["visual_audit"] = visual_results

        # Write result to task record (as a JSON string in status or a separate field)
        # For simplicity, we'll write a report.json in the work_dir.
        with open(os.path.join(work_dir, "report.json"), "w") as f:
            json.dump(audit_report, f, indent=2)

        record_task(task_id, "content_auditor", "COMPLETED")
        print(f"[Auditor] Completed Audit {task_id}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        record_task(task_id, "content_auditor", f"FAILED: {str(e)}")


# --- Proxy Logic ---
async def proxy_request(request: Request, backend_url: str):
    if http_client is None:
        raise HTTPException(status_code=503, detail="Orchestrator client not initialized.")
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

@app.get("/health")
async def orchestrator_health():
    return {
        "status": "ok",
        "message": "AI-Hub Orchestrator is running.",
        "registry": BACKENDS,
    }


@app.get("/v1/models")
async def route_models():
    if http_client is None:
        return {"object": "list", "data": [], "unavailable": [{"backend": "all", "error": "Orchestrator client not initialized"}]}
    models = []
    unavailable = []

    for backend_name, backend_url in BACKENDS.items():
        try:
            resp = await http_client.get(f"{backend_url}/v1/models")
        except httpx.RequestError as e:
            unavailable.append({"backend": backend_name, "error": str(e)})
            continue

        content_type = resp.headers.get("content-type", "")
        if resp.status_code == 404 or "application/json" not in content_type.lower():
            continue
        if resp.status_code >= 400:
            unavailable.append({"backend": backend_name, "status_code": resp.status_code})
            continue

        payload = resp.json()
        for item in payload.get("data", []):
            if isinstance(item, dict):
                enriched_item = dict(item)
                enriched_item.setdefault("source_backend", backend_name)
                models.append(enriched_item)

    return {
        "object": "list",
        "data": models,
        "unavailable": unavailable,
    }
# Audio (TTS)
@app.get("/v1/audio/voices")
@app.get("/v1/audio/voices/list")
@app.get("/v1/audio/speakers")
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

# Audio (Music)
@app.get("/v1/stats")
@app.post("/v1/audio/async_generations")
@app.get("/v1/audio")
async def route_music(request: Request):
    if "music" not in BACKENDS:
        raise HTTPException(status_code=503, detail="Music Backend not configured.")
    return await proxy_request(request, BACKENDS["music"])
@app.get("/v1/audio/tasks/{task_id}")
async def route_music_tasks(request: Request, task_id: str):
    if "music" not in BACKENDS:
        raise HTTPException(status_code=503, detail="Music Backend not configured.")
    url = f"{BACKENDS['music']}{request.url.path}"
    if request.url.query:
        url += f"?{request.url.query}"
    headers = dict(request.headers)
    headers.pop("host", None)
    try:
        resp = await http_client.get(url, headers=headers)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Bad Gateway: Error communicating with backend model service: {e}")
    filtered_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")
    }
    content_type = resp.headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        return Response(content=resp.content, status_code=resp.status_code, headers=filtered_headers)
    data = resp.json()
    if data.get("status") == "completed":
        for item in data.get("data", []):
            if isinstance(item, dict) and item.get("url"):
                item["url"] = _rewrite_backend_url_to_hub(item["url"], "music")
    return JSONResponse(status_code=resp.status_code, content=data, headers=filtered_headers)
# Vision (Images)
@app.post("/v1/images/generations")
@app.post("/v1/images/async_generate")
@app.post("/v1/images/async_edit")
async def route_vision(request: Request):
    if "vision" not in BACKENDS:
        raise HTTPException(status_code=503, detail="Vision Backend not configured.")
    return await proxy_request(request, BACKENDS["vision"])


@app.get("/v1/images/tasks/{task_id}")
async def route_vision_tasks(task_id: str):
    if "vision" not in BACKENDS:
        raise HTTPException(status_code=503, detail="Vision Backend not configured.")
    try:
        resp = await http_client.get(f"{BACKENDS['vision']}/v1/images/tasks/{task_id}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Bad Gateway: Error communicating with backend model service: {e}")
    filtered_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")
    }
    content_type = resp.headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        return Response(content=resp.content, status_code=resp.status_code, headers=filtered_headers)
    data = resp.json()
    for item in data.get("data", []):
        if isinstance(item, dict) and item.get("url"):
            item["url"] = _rewrite_backend_output_url(item["url"], "vision", "/v1/images/outputs")
    return JSONResponse(status_code=resp.status_code, content=data, headers=filtered_headers)


@app.get("/v1/images/outputs/{filename:path}")
async def route_vision_output(request: Request, filename: str):
    return await _proxy_backend_output(request, "vision", filename)

# Video
@app.post("/v1/video/generations")
@app.post("/v1/video/async_t2v")
@app.post("/v1/video/async_i2v")
async def route_video(request: Request):
    if "video" not in BACKENDS:
        raise HTTPException(status_code=503, detail="Video Backend not configured.")
    return await proxy_request(request, BACKENDS["video"])


@app.get("/v1/video/tasks/{task_id}")
async def route_video_tasks(task_id: str):
    if "video" not in BACKENDS:
        raise HTTPException(status_code=503, detail="Video Backend not configured.")
    try:
        resp = await http_client.get(f"{BACKENDS['video']}/v1/video/tasks/{task_id}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Bad Gateway: Error communicating with backend model service: {e}")
    filtered_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")
    }
    content_type = resp.headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        return Response(content=resp.content, status_code=resp.status_code, headers=filtered_headers)
    data = resp.json()
    if isinstance(data, dict) and data.get("url"):
        data["url"] = _rewrite_backend_output_url(data["url"], "video", "/v1/video/outputs")
    return JSONResponse(status_code=resp.status_code, content=data, headers=filtered_headers)


@app.get("/v1/video/outputs/{filename:path}")
async def route_video_output(request: Request, filename: str):
    return await _proxy_backend_output(request, "video", filename)

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
            resp = await http_client.get(f"{url}/health", timeout=10.0)
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


@app.get("/v1/hub/tasks/{task_id}")
async def hub_task_status(task_id: str):
    """
    Retrieve status for a specific task.
    """
    for t in recent_tasks:
        if t["task_id"] == task_id:
            # Add output_url for Director tasks if completed
            data = dict(t)
            if t["status"] == "COMPLETED" and t["workflow"] == "content_director":
                data["output_url"] = f"/outputs/{task_id}/final_director_cut.mp4"
            return data
            
    # Also check audit directory
    audit_report_path = os.path.join(OUTPUTS_DIR, "audit", task_id, "report.json")
    if os.path.exists(audit_report_path):
        with open(audit_report_path, "r") as f:
            report = json.load(f)
        return {"task_id": task_id, "workflow": "content_auditor", "status": "COMPLETED", "analysis": report}

    raise HTTPException(status_code=404, detail=f"Task {task_id} not found")


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
