import os
import sys
import yaml
import httpx
import uuid
import asyncio
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

# Add root directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from orchestrator.media_utils import mux_video_and_audio

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

app = FastAPI(title="AI-Hub Orchestrator (Jarvis)", version="1.0", lifespan=lifespan)

if GracefulJSONRoute:
    app.router.route_class = GracefulJSONRoute

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    
    # Fast polling interval
    poll_interval = 2.0
    
    print(f"[Director] Started Workflow {task_id}")

    try:
        # --- STEP 1 & 2: Generate Image and Audio concurrently ---
        print("[Director] Requesting Vision and Audio generation concurrently...")
        
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
        print(f"[Director] Sending Vision Request -> {vision_url}")
        vision_resp = await http_client.post(vision_url, json=vision_payload)
        
        # Check Vision result (Async typically, need to poll)
        if vision_resp.status_code != 200:
            raise Exception(f"Vision backend initialization failed: {vision_resp.text}")
            
        vision_task_id = vision_resp.json().get("task_id")
        print(f"[Director] Vision task started: {vision_task_id}. Polling...")
        
        # Poll Vision
        image_url = None
        while True:
            v_poll = await http_client.get(f"{BACKENDS['vision']}/v1/images/tasks/{vision_task_id}")
            v_data = v_poll.json()
            if v_data["status"] == "COMPLETED" or v_data["status"] == "completed":
                # Ensure we have the local path or data. Vision currently returns URLs.
                # Assuming Vision and Orchestrator are on the same machine, we can grab the relative or absolute path 
                # or download the URL result. Let's download it to be safe.
                # Z-Image returns `data` array containing dicts with `url`
                image_url = v_data["data"][0]["url"]
                # Convert relative URL to full if needed
                if image_url.startswith("/"):
                    # Extract base from BACKENDS
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
        
        # Poll Video
        raw_video_url = None
        while True:
            vid_poll = await http_client.get(f"{BACKENDS['video']}/v1/video/tasks/{vid_task_id}")
            vid_data = vid_poll.json()
            if vid_data["status"] == "completed":
                # LTX-2 returns local absolute paths usually, or URLs.
                local_path = vid_data["output"].get("local_path")
                raw_video_url = vid_data["output"].get("url")
                
                # If local_path exists, use it direct. Else fallback to downloading from URL.
                if local_path and os.path.exists(local_path):
                    raw_video_path = local_path
                else:
                    target_url = raw_video_url if raw_video_url.startswith("http") else f"{BACKENDS['video']}{raw_video_url}"
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
        print("[Director] Stitching Audio and Video...")
        loop = asyncio.get_running_loop()
        final_video_path = os.path.join(work_dir, "final_director_cut.mp4")
        
        # Run the blocking subprocess in an executor
        success = await loop.run_in_executor(None, mux_video_and_audio, raw_video_path, audio_path, final_video_path)
        
        if not success:
            raise Exception("FFmpeg failed to mux the final video.")
            
        print(f"[Director] Workflow complete! {final_video_path}")
        
        return JSONResponse(status_code=200, content={
            "task_id": task_id,
            "status": "COMPLETED",
            "message": "Content Director successfully assembled the video.",
            "final_video_path": final_video_path,
            "working_directory": work_dir
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
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

# Generic pass-through for anything else (optional, maybe too open, but good for outputs)
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(request: Request, path: str):
    # Route /outputs/ to specific servers based on file prefixes if desired, 
    # but currently all servers use their own /outputs/. We can just pick vision as default for now 
    # or return a generic hub status.
    if path == "health" or path == "":
        return {"status": "ok", "message": "AI-Hub Swappable Jarvis Orchestrator is running.", "registry": BACKENDS}
        
    # If it's an /outputs/ request, we might need a mapping. Usually the URL given to the client 
    # comes from the server itself. To support the hub, servers should ideally return full URLs, 
    # or we mount a shared network drive. For now, proxying outputs is complex without a shared directory.
    # We will just raise 404 for unmapped endpoints.
    raise HTTPException(status_code=404, detail=f"Orchestrator route not defined for path: {path}")

if __name__ == "__main__":
    port = getattr(config, "ORCHESTRATOR_PORT", 9000)
    print(f"Starting Swappable Jarvis Orchestrator on {config.HOST}:{port}")
    uvicorn.run(app, host=config.HOST, port=port)
