"""
Antigravity AI - Vision Service (Z-Image)

Serves Z-Image (Tongyi-MAI/Z-Image) as an OpenAI-compatible image generation API.
Supports text-to-image (/v1/images/generations).
"""

import os
# Force memory strategy
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Ensure CUDA is visible
if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]

import sys
import threading
import uvicorn
import logging
import asyncio
import torch
import traceback
import time
import io
import base64
from uuid import uuid4
from typing import Optional, List
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from diffusers.utils import load_image

# --- NUCLEAR OPTION: Default Device ---
try:
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        print("Global default device set to CUDA.")
except Exception as e:
    print(f"Warning: Could not set default device to CUDA: {e}")

# Add parent dir for config import. Insert at 0 to prevent pip shadowing.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
print(f"DEBUG: Loaded config from {config.__file__}")

# --- Configuration ---
HOST = config.HOST
PORT = config.VISION_PORT
T2I_MODEL_ID = config.VISION_MODEL
EDIT_MODEL_ID = config.VISION_EDIT_MODEL
CACHE_DIR = config.HF_HOME
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated_images")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Z-Image")

# --- Global State ---
t2i_pipeline = None
edit_pipeline = None
last_activity_time = time.time()
server_shutdown_event = asyncio.Event()
is_generating = False

# --- Model Management ---
def apply_device_monkeypatch(pipe):
    """Applies the universal context patch for sequence offloading"""
    original_call = pipe.__call__
    def patched_call(*args, **kwargs):
        if "generator" in kwargs and kwargs["generator"] is not None:
            try:
                kwargs["generator"] = kwargs["generator"].to("cuda")
            except: pass
        with torch.device("cuda:0"):
            return original_call(*args, **kwargs)
    pipe.__call__ = patched_call
    return pipe

def _purge_vram():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def unload_model():
    """Unload all models to free VRAM completely."""
    global t2i_pipeline, edit_pipeline
    purged = False
    if t2i_pipeline is not None:
        logger.info("Unloading Z-Image T2I model...")
        del t2i_pipeline
        t2i_pipeline = None
        purged = True
    if edit_pipeline is not None:
        logger.info("Unloading Z-Image Edit model...")
        del edit_pipeline
        edit_pipeline = None
        purged = True
        
    if purged:
        _purge_vram()
        logger.info("VRAM purged.")

def load_t2i_model():
    """Load the T2I pipeline (Unloads Edit pipeline if active)."""
    global t2i_pipeline, edit_pipeline
    
    if edit_pipeline is not None:
        logger.info("Purging Edit pipeline to make room for T2I...")
        del edit_pipeline
        edit_pipeline = None
        _purge_vram()
        
    if t2i_pipeline is None:
        logger.info(f"Loading Z-Image T2I (Sequential Offload): {T2I_MODEL_ID}...")
        _purge_vram()

        try:
            from diffusers import ZImagePipeline

            load_args = {
                "cache_dir": CACHE_DIR,
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
            }

            try:
                t2i_pipeline = ZImagePipeline.from_pretrained(T2I_MODEL_ID, **load_args)
            except Exception as e:
                logger.warning(f"Initial T2I load failed ({e}), trying local only...")
                load_args["local_files_only"] = True
                t2i_pipeline = ZImagePipeline.from_pretrained(T2I_MODEL_ID, **load_args)

            try:
                t2i_pipeline.enable_model_cpu_offload()
                t2i_pipeline.vae.enable_tiling()
            except Exception as mem_err:
                logger.warning(f"Could not enable model offload: {mem_err}")

            t2i_pipeline = apply_device_monkeypatch(t2i_pipeline)
            logger.info("Z-Image T2I loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load Z-Image T2I: {e}")
            logger.error(traceback.format_exc())
            t2i_pipeline = None
            raise HTTPException(status_code=500, detail=f"T2I Load Error: {str(e)}")

def load_edit_model():
    """Load the Edit pipeline (Unloads T2I pipeline if active)."""
    global t2i_pipeline, edit_pipeline
    
    if t2i_pipeline is not None:
        logger.info("Purging T2I pipeline to make room for Edit...")
        del t2i_pipeline
        t2i_pipeline = None
        _purge_vram()
        
    if edit_pipeline is None:
        logger.info(f"Loading Z-Image Edit (Sequential Offload): {EDIT_MODEL_ID}...")
        _purge_vram()

        try:
            # Note: diffusers might import this differently depending on version. 
            # Assuming standard structural naming based on ZImagePipeline.
            from diffusers import ZImagePipeline
            # Using standard AutoPipelineForImage2Image as fallback if ZImageEditPipeline isn't explicitly exposed yet
            from diffusers import AutoPipelineForImage2Image

            load_args = {
                "cache_dir": CACHE_DIR,
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
            }

            try:
                # Many custom archs like Z-Image expose the base pipeline and override call, 
                # or have a dedicated pipeline.
                try: 
                    from diffusers import ZImageEditPipeline
                    edit_pipeline = ZImageEditPipeline.from_pretrained(EDIT_MODEL_ID, **load_args)
                except ImportError:
                    edit_pipeline = AutoPipelineForImage2Image.from_pretrained(EDIT_MODEL_ID, **load_args)
            except Exception as e:
                logger.warning(f"Initial Edit load failed ({e}), trying local only...")
                load_args["local_files_only"] = True
                try: 
                    from diffusers import ZImageEditPipeline
                    edit_pipeline = ZImageEditPipeline.from_pretrained(EDIT_MODEL_ID, **load_args)
                except ImportError:
                    edit_pipeline = AutoPipelineForImage2Image.from_pretrained(EDIT_MODEL_ID, **load_args)

            try:
                edit_pipeline.enable_model_cpu_offload()
                if hasattr(edit_pipeline, 'vae'):
                    edit_pipeline.vae.enable_tiling()
            except Exception as mem_err:
                logger.warning(f"Could not enable model offload: {mem_err}")

            edit_pipeline = apply_device_monkeypatch(edit_pipeline)
            logger.info("Z-Image Edit loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load Z-Image Edit: {e}")
            logger.error(traceback.format_exc())
            edit_pipeline = None
            raise HTTPException(status_code=500, detail=f"Edit Load Error: {str(e)}")

async def idle_check_loop():
    """Background task to unload model after inactivity."""
    global t2i_pipeline, edit_pipeline, last_activity_time
    unload_timeout = getattr(config, "VISION_IDLE_TIMEOUT", 600)
    
    logger.info(f"Idle check loop started (timeout={unload_timeout}s)")
    while not server_shutdown_event.is_set():
        if t2i_pipeline is not None or edit_pipeline is not None:
            idle_time = time.time() - last_activity_time
            if idle_time > unload_timeout:
                logger.info(f"Vision model idle for {int(idle_time)}s. Unloading...")
                unload_model()
        await asyncio.sleep(60)

# --- Helpers ---
def validate_and_round_size(width: int, height: int) -> tuple[int, int]:
    w = max(32, (width // 32) * 32)
    h = max(32, (height // 32) * 32)
    return w, h

def parse_size(size_str: str) -> tuple[int, int]:
    try:
        parts = size_str.lower().split("x")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return 1024, 1024

def image_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def save_image(img: Image.Image) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"gen_{int(time.time())}_{uuid4().hex[:8]}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    img.save(filepath)
    return filename

# --- API Models ---
class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "url"
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    cfg_normalization: bool = False

# --- FastAPI App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(idle_check_loop())
    yield
    server_shutdown_event.set()
    unload_model()

app = FastAPI(title="Antigravity Vision Service", lifespan=lifespan)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/outputs/{filename}")
async def get_output_file(filename: str):
    # Prevent directory traversal
    filepath = os.path.abspath(os.path.join(OUTPUT_DIR, filename))
    if not filepath.startswith(os.path.abspath(OUTPUT_DIR)):
        raise HTTPException(status_code=403, detail="Forbidden")
    # Gracefully catch invalid Windows characters (WinError 123)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath)

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "t2i_model": T2I_MODEL_ID, 
        "edit_model": EDIT_MODEL_ID,
        "t2i_loaded": t2i_pipeline is not None,
        "edit_loaded": edit_pipeline is not None
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": T2I_MODEL_ID, 
                "object": "model", 
                "created": int(time.time()), 
                "owned_by": "zai-org",
                "capabilities": ["text-to-image"]
            },
            {
                "id": EDIT_MODEL_ID, 
                "object": "model", 
                "created": int(time.time()), 
                "owned_by": "zai-org",
                "capabilities": ["image-to-image"]
            }
        ]
    }

@app.get("/v1/internal/status")
async def internal_status():
    return {
        "status": "generating" if is_generating else "idle",
        "t2i_loaded": t2i_pipeline is not None,
        "edit_loaded": edit_pipeline is not None
    }

generation_tasks = {}
generation_lock = threading.Lock()

def process_async_task(task_id: str, request: ImageGenerationRequest):
    global last_activity_time, is_generating
    last_activity_time = time.time()
    
    with generation_lock:
        is_generating = True
        try:
            load_t2i_model()
            width, height = parse_size(request.size)
            width, height = validate_and_round_size(width, height)
            
            logger.info(f"[Task {task_id}] Generating image (Sequential Offload): '{request.prompt[:50]}...'")
            with torch.inference_mode():
                images = t2i_pipeline(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    cfg_normalization=request.cfg_normalization,
                    num_images_per_prompt=request.n
                ).images

            img = images[0]
            
            if request.response_format == "b64_json":
                result_data = {"b64_json": image_to_base64(img)}
            else:
                filename = save_image(img)
                url = f"http://{config.PUBLIC_HOST}:{PORT}/outputs/{filename}"
                result_data = {"url": url}
                
            generation_tasks[task_id] = {"status": "completed", "data": [result_data]}
            logger.info(f"[Task {task_id}] Completed successfully.")
            
        except Exception as e:
            logger.error(f"[Task {task_id}] Generation error: {e}")
            logger.error(traceback.format_exc())
            generation_tasks[task_id] = {"status": "failed", "error": str(e)}
        finally:
            is_generating = False
            last_activity_time = time.time()

@app.post("/v1/images/async_generate")
def async_generate(request: ImageGenerationRequest, background_tasks: BackgroundTasks):
    """
    Agent-proof endpoint: Returns immediately, runs generation in the background.
    """
    task_id = str(uuid4().hex)
    generation_tasks[task_id] = {"status": "processing"}
    background_tasks.add_task(process_async_task, task_id, request)
    return {"task_id": task_id, "status": "processing", "message": "Task queued. Poll GET /v1/images/tasks/{task_id}"}


def process_async_edit_task(task_id: str, image_path: str, prompt: str):
    global last_activity_time, is_generating
    last_activity_time = time.time()
    
    with generation_lock:
        is_generating = True
        try:
            load_edit_model()
            
            logger.info(f"[Task {task_id}] Editing image (Sequential Offload): '{prompt[:50]}...'")
            
            # Load the source image into PIL format for Diffusers
            source_image = load_image(image_path)
            
            with torch.inference_mode():
                images = edit_pipeline(
                    prompt=prompt,
                    image=source_image,
                    num_inference_steps=50, # Z-Image-Edit recommends 50 steps
                    guidance_scale=4.0
                ).images

            img = images[0]
            filename = save_image(img)
            url = f"http://{config.PUBLIC_HOST}:{PORT}/outputs/{filename}"
            result_data = {"url": url}
                
            generation_tasks[task_id] = {"status": "completed", "data": [result_data]}
            logger.info(f"[Task {task_id}] Edit completed successfully.")
            
        except Exception as e:
            logger.error(f"[Task {task_id}] Edit error: {e}")
            logger.error(traceback.format_exc())
            generation_tasks[task_id] = {"status": "failed", "error": str(e)}
        finally:
            is_generating = False
            last_activity_time = time.time()
            # Clean up the temporary uploaded source file
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
            except:
                 pass

@app.post("/v1/images/async_edit")
async def async_edit(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    """
    Agent-proof editing endpoint: Returns immediately, edits in the background.
    Uses multipart/form-data to accept the source image.
    """
    task_id = str(uuid4().hex)
    
    # Save the uploaded file temporarily so the background thread can load it
    temp_dir = os.path.join(CACHE_DIR, "tmp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_path = os.path.join(temp_dir, f"{task_id}_{image.filename}")
    with open(temp_path, "wb") as buffer:
        buffer.write(await image.read())
        
    generation_tasks[task_id] = {"status": "processing"}
    background_tasks.add_task(process_async_edit_task, task_id, temp_path, prompt)
    
    return {"task_id": task_id, "status": "processing", "message": "Task queued. Poll GET /v1/images/tasks/{task_id}"}


@app.get("/v1/images/tasks/{task_id}")
def get_task_status(task_id: str):
    """
    Poll this endpoint for generation status.
    """
    if task_id not in generation_tasks:
        return {
            "status": "failed", 
            "error": "Task not found (server may have restarted or ID is invalid). Please submit a new request."
        }
    return generation_tasks[task_id]

# --- LEGACY OPENAI SYNC ENDPOINT ---
@app.get("/v1/images/generations")
async def get_generations_status():
    return {
        "error": "Method not allowed. Use POST to generate.",
        "hint": "If you are checking status, the server is currently " + ("GENERATING" if is_generating else "IDLE"),
        "is_generating": is_generating
    }

@app.post("/v1/images/generations")
def generate_image_sync(request: ImageGenerationRequest):
    global last_activity_time, is_generating
    last_activity_time = time.time()
    
    with generation_lock:
        is_generating = True
        try:
            load_t2i_model()
            width, height = parse_size(request.size)
            width, height = validate_and_round_size(width, height)
            
            logger.info(f"Generating image (Sync): '{request.prompt[:50]}...'")
            with torch.inference_mode():
                images = t2i_pipeline(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    cfg_normalization=request.cfg_normalization,
                    num_images_per_prompt=request.n
                ).images

            data = []
            for img in images:
                if request.response_format == "b64_json":
                    data.append({"b64_json": image_to_base64(img)})
                else:
                    filename = save_image(img)
                    url = f"http://{config.PUBLIC_HOST}:{PORT}/outputs/{filename}"
                    data.append({"url": url})
            
            return {"created": int(time.time()), "data": data}
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            is_generating = False
            last_activity_time = time.time()

if __name__ == "__main__":
    logger.info(f"Vision Service starting (Sequential Offload) on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
