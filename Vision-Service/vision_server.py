"""
Antigravity AI - Vision Service (GLM-Image)

Serves GLM-Image (zai-org/GLM-Image) as an OpenAI-compatible image generation API.
Supports text-to-image (/v1/images/generations) and image-to-image (/v1/images/edits).

Usage:
    python vision_server.py
"""

import os
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

# Add parent dir for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# --- Configuration ---
HOST = config.HOST
PORT = config.VISION_PORT
MODEL_ID = config.VISION_MODEL
CACHE_DIR = config.HF_HOME
OFFLOAD_CPU = config.VISION_OFFLOAD_CPU
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated_images")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Vision-Service")

# --- Global State ---
pipeline = None
last_activity_time = time.time()
server_shutdown_event = asyncio.Event()


# --- Model Management ---
def load_model():
    """Load the GLM-Image pipeline."""
    global pipeline
    if pipeline is None:
        logger.info(f"Loading GLM-Image model: {MODEL_ID}...")
        try:
            from diffusers.pipelines.glm_image import GlmImagePipeline

            # Try online first, fallback to cached model if network fails
            try:
                pipeline = GlmImagePipeline.from_pretrained(
                    MODEL_ID,
                    cache_dir=CACHE_DIR,
                    torch_dtype=torch.bfloat16,
                )
            except (OSError, Exception) as net_err:
                if "ConnectTimeout" in str(type(net_err).__name__) or "WinError" in str(net_err) or "ConnectionError" in str(type(net_err).__name__) or "ConnectTimeout" in str(net_err):
                    logger.warning(f"Network unavailable ({net_err}), loading from local cache...")
                    pipeline = GlmImagePipeline.from_pretrained(
                        MODEL_ID,
                        cache_dir=CACHE_DIR,
                        torch_dtype=torch.bfloat16,
                        local_files_only=True,
                    )
                else:
                    raise

            if OFFLOAD_CPU:
                pipeline.enable_model_cpu_offload()
                logger.info("CPU offload enabled to reduce VRAM usage.")
            else:
                pipeline.to("cuda")

            logger.info("GLM-Image loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")


def unload_model():
    """Unload the model to free VRAM."""
    global pipeline
    if pipeline is not None:
        logger.info("Unloading GLM-Image to free VRAM...")
        try:
            if hasattr(pipeline, "to"):
                pipeline.to("cpu")
            del pipeline
            pipeline = None
            torch.cuda.empty_cache()
            logger.info("Model unloaded.")
        except Exception as e:
            logger.error(f"Error unloading model: {e}")


async def idle_check_loop():
    """Checks for inactivity and unloads model if idle for too long."""
    while not server_shutdown_event.is_set():
        if pipeline is not None:
            if time.time() - last_activity_time > 60:  # 60 seconds idle timeout
                logger.info("Idle timeout reached. Unloading model.")
                unload_model()
        await asyncio.sleep(10)


# --- Helpers ---
def validate_and_round_size(width: int, height: int) -> tuple[int, int]:
    """Ensure dimensions are divisible by 32 (GLM-Image requirement)."""
    w = max(32, (width // 32) * 32)
    h = max(32, (height // 32) * 32)
    if w != width or h != height:
        logger.warning(f"Adjusted size from {width}x{height} to {w}x{h} (must be divisible by 32)")
    return w, h


def parse_size(size_str: str) -> tuple[int, int]:
    """Parse 'WIDTHxHEIGHT' string into (width, height) tuple."""
    try:
        parts = size_str.lower().split("x")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        raise HTTPException(status_code=400, detail=f"Invalid size format: '{size_str}'. Expected 'WIDTHxHEIGHT'.")


def image_to_base64(img: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def save_image(img: Image.Image) -> str:
    """Save image to disk and return the filename."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{uuid4()}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    img.save(filepath)
    return filename


# --- API Models ---
class ImageGenerationRequest(BaseModel):
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "url"  # "url" or "b64_json"
    num_inference_steps: int = 50
    guidance_scale: float = 1.5


# --- FastAPI App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Vision Service (GLM-Image) Starting...")
    asyncio.create_task(idle_check_loop())
    yield
    # Shutdown
    logger.info("Vision Service Shutting Down...")
    server_shutdown_event.set()
    unload_model()


app = FastAPI(title="Vision Service (GLM-Image)", version="2.0", lifespan=lifespan)


# --- Endpoints ---

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "model_loaded": pipeline is not None,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "zai-org",
                "capabilities": ["text-to-image", "image-to-image"],
            }
        ],
    }


@app.post("/v1/images/generations")
async def generate_image(request: ImageGenerationRequest):
    """Text-to-image generation (OpenAI-compatible)."""
    global last_activity_time
    last_activity_time = time.time()

    load_model()

    width, height = parse_size(request.size)
    width, height = validate_and_round_size(width, height)

    try:
        logger.info(f"Generating image: prompt='{request.prompt[:80]}...', size={width}x{height}")

        images = []
        for i in range(request.n):
            with torch.inference_mode():
                gen_device = "cpu" if config.VISION_OFFLOAD_CPU else "cuda"
                result = pipeline(
                    prompt=request.prompt,
                    height=height,
                    width=width,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    generator=torch.Generator(device=gen_device).manual_seed(int(time.time()) + i),
                ).images[0]
            images.append(result)

        # Build response
        data = []
        for img in images:
            if request.response_format == "b64_json":
                data.append({
                    "b64_json": image_to_base64(img),
                    "revised_prompt": request.prompt,
                })
            else:
                filename = save_image(img)
                data.append({
                    "url": f"http://{HOST}:{PORT}/files/{filename}",
                    "revised_prompt": request.prompt,
                })

        last_activity_time = time.time()
        return {"created": int(time.time()), "data": data}

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/images/edits")
async def edit_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    size: Optional[str] = Form(None),
    response_format: str = Form("url"),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(1.5),
):
    """Image-to-image generation (OpenAI-compatible edits endpoint)."""
    global last_activity_time
    last_activity_time = time.time()

    load_model()

    try:
        # Read and validate input image
        image_data = await image.read()
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        logger.info(f"Input image size: {input_image.size}")

        # Determine output size
        if size:
            width, height = parse_size(size)
        else:
            width, height = input_image.size
        width, height = validate_and_round_size(width, height)

        logger.info(f"Image editing: prompt='{prompt[:80]}...', size={width}x{height}")

        gen_device = "cpu" if config.VISION_OFFLOAD_CPU else "cuda"
        with torch.inference_mode():
            result = pipeline(
                prompt=prompt,
                image=input_image,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=gen_device).manual_seed(int(time.time())),
            ).images[0]

        # Build response
        if response_format == "b64_json":
            data = [{"b64_json": image_to_base64(result), "revised_prompt": prompt}]
        else:
            filename = save_image(result)
            data = [{"url": f"http://{HOST}:{PORT}/files/{filename}", "revised_prompt": prompt}]

        last_activity_time = time.time()
        return {"created": int(time.time()), "data": data}

    except Exception as e:
        logger.error(f"Image editing failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/internal/unload")
async def manual_unload():
    """Manually unload the model to free VRAM."""
    unload_model()
    return {"status": "Model unloaded"}


# Mount static files for serving generated images
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
