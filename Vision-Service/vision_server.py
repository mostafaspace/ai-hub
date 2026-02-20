"""
Antigravity AI - Vision Service (GLM-Image)

Serves GLM-Image (zai-org/GLM-Image) as an OpenAI-compatible image generation API.
Supports text-to-image (/v1/images/generations).
"""

import os
# Force memory strategy
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Ensure CUDA is visible
if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]

import sys
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
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles

# --- NUCLEAR OPTION: Default Device ---
try:
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        print("Global default device set to CUDA.")
except Exception as e:
    print(f"Warning: Could not set default device to CUDA: {e}")

# Add parent dir for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# --- Configuration ---
HOST = config.HOST
PORT = config.VISION_PORT
MODEL_ID = config.VISION_MODEL
CACHE_DIR = config.HF_HOME
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated_images")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Vision-Service")

# --- Global State ---
pipeline = None
last_activity_time = time.time()
server_shutdown_event = asyncio.Event()

# --- Model Management ---
def unload_model():
    """Unload the model to free VRAM."""
    global pipeline
    if pipeline is not None:
        logger.info("Unloading GLM-Image model to free VRAM...")
        del pipeline
        pipeline = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded.")

def load_model():
    """Load the GLM-Image pipeline with Sequential CPU Offloading."""
    global pipeline
    if pipeline is None:
        logger.info(f"Loading GLM-Image model (Sequential Offload): {MODEL_ID}...")
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            from diffusers.pipelines.glm_image import GlmImagePipeline

            # Load strictly in bfloat16 to avoid black images (VAE overflow) and bitsandbytes bugs.
            # We use sequential_cpu_offload to handle the 31GB footprint.
            load_args = {
                "cache_dir": CACHE_DIR,
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
            }

            # Load Pipeline (device_map=None because we use enable_sequential_cpu_offload)
            try:
                pipeline = GlmImagePipeline.from_pretrained(MODEL_ID, **load_args)
            except Exception as e:
                logger.warning(f"Initial load failed ({e}), trying local only...")
                load_args["local_files_only"] = True
                pipeline = GlmImagePipeline.from_pretrained(MODEL_ID, **load_args)

            # High-Stability Memory Management
            try:
                # Model offload is faster than sequential and still saves VRAM.
                pipeline.enable_model_cpu_offload()
                pipeline.vae.enable_tiling()
                logger.info("Model CPU offload and VAE tiling enabled.")
            except Exception as mem_err:
                logger.warning(f"Could not enable model offload: {mem_err}")

            # --- MONKEYPATCH: Final Device Firewall ---
            original_call = pipeline.__call__
            def patched_call(*args, **kwargs):
                if "generator" in kwargs and kwargs["generator"] is not None:
                    try:
                        kwargs["generator"] = kwargs["generator"].to("cuda")
                    except: pass
                
                # With sequential offload, we MUST ensure the context is correctly set
                with torch.device("cuda:0"):
                    return original_call(*args, **kwargs)
            pipeline.__call__ = patched_call
            logger.info("Universal Device Context patch applied.")

            logger.info("GLM-Image loaded successfully (Sequential Offload).")

        except Exception as e:
            logger.error(f"Failed to load GLM-Image: {e}")
            logger.error(traceback.format_exc())
            pipeline = None
            raise HTTPException(status_code=500, detail=str(e))

async def idle_check_loop():
    """Background task to unload model after inactivity."""
    global pipeline, last_activity_time
    unload_timeout = getattr(config, "VISION_IDLE_TIMEOUT", 600)
    
    logger.info(f"Idle check loop started (timeout={unload_timeout}s)")
    while not server_shutdown_event.is_set():
        if pipeline is not None:
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
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "url"
    num_inference_steps: int = 50
    guidance_scale: float = 1.5

# --- FastAPI App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(idle_check_loop())
    yield
    server_shutdown_event.set()
    unload_model()

app = FastAPI(title="Antigravity Vision Service", lifespan=lifespan)
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "model_loaded": pipeline is not None}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID, 
            "object": "model", 
            "created": int(time.time()), 
            "owned_by": "zai-org",
            "capabilities": ["text-to-image"]
        }]
    }

@app.post("/v1/images/generations")
async def generate_image(request: ImageGenerationRequest):
    global last_activity_time
    last_activity_time = time.time()
    
    load_model()
    width, height = parse_size(request.size)
    width, height = validate_and_round_size(width, height)
    
    try:
        logger.info(f"Generating image (Sequential Offload): '{request.prompt[:50]}...'")
        with torch.inference_mode():
            images = pipeline(
                prompt=request.prompt,
                width=width,
                height=height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                num_images_per_prompt=request.n
            ).images

        data = []
        for img in images:
            if request.response_format == "b64_json":
                data.append({"b64_json": image_to_base64(img)})
            else:
                filename = save_image(img)
                # Use PUBLIC_HOST so external agents can download the image instead of 0.0.0.0
                url = f"http://{config.PUBLIC_HOST}:{PORT}/outputs/{filename}"
                data.append({"url": url})
        
        return {"created": int(time.time()), "data": data}
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info(f"Vision Service starting (Sequential Offload) on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
