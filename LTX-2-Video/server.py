import os
import io
import time
import torch
import gc
import threading
import logging
import traceback
import uvicorn
from uuid import uuid4
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Literal
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "LTX-Core", "packages", "ltx-core", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "LTX-Core", "packages", "ltx-pipelines", "src")))

# HOTFIX: The unquantized Gemma-3 model requires 'rope_local_base_freq'
try:
    import transformers
    if hasattr(transformers.models, "gemma3"):
        from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
        original_init = Gemma3TextConfig.__init__
        def patched_init(self, *args, **kwargs):
            if "rope_local_base_freq" not in kwargs:
                kwargs["rope_local_base_freq"] = 10000.0  # Default value
            original_init(self, *args, **kwargs)
        Gemma3TextConfig.__init__ = patched_init
except ImportError:
    pass
except Exception as e:
    print(f"[LTX-2] Warning: Could not patch Gemma3TextConfig: {e}")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "LTX-Core", "packages", "ltx-pipelines", "src")))

# We'll need ltx_pipelines, assuming we run this from a script that sets PYTHONPATH
try:
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.quantization import QuantizationPolicy
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
except ImportError:
    print("Warning: ltx-pipelines not found. Ensure the package is installed and accessible.")

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

HOST = config.HOST
PORT = config.VIDEO_PORT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated_videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = logging.getLogger("LTX-2")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# ---- LTX-2 Specific Config (Ideally this comes from config.py) ----
# We use standard defaults for 19B if config doesn't specify.
LTX_MODEL_PATH = getattr(config, "VIDEO_MODEL_BASE", os.path.join(config.HF_HOME, "Lightricks/LTX-2/ltx-2-19b-dev-fp8.safetensors"))
GEMMA_ROOT = getattr(config, "VIDEO_GEMMA_ROOT", os.path.join(config.HF_HOME, "google/gemma-3-12b-it-qat-q4_0-unquantized"))
DISTILLED_LORA = getattr(config, "VIDEO_DISTILLED_LORA", os.path.join(config.HF_HOME, "Lightricks/LTX-2/ltx-2-19b-distilled-lora-384.safetensors"))
SPATIAL_UPSAMPLER = getattr(config, "VIDEO_SPATIAL_UPSAMPLER", os.path.join(config.HF_HOME, "Lightricks/LTX-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"))


# --- Model Management (Auto-Unload via Thread RLock) ---
class ModelManager:
    def __init__(self, idle_timeout: float = 60.0, check_interval: float = 10.0):
        self.pipeline = None
        self.last_active = time.time()
        self.idle_timeout = idle_timeout
        self.check_interval = check_interval
        self.lock = threading.RLock()
        
        self.monitor_thread = threading.Thread(target=self._idle_monitor, daemon=True)
        self.monitor_thread.start()

    def _idle_monitor(self):
        while True:
            time.sleep(self.check_interval)
            # Lock-free check first (reading a float is atomic in CPython)
            if self.pipeline and (time.time() - self.last_active > self.idle_timeout):
                with self.lock:
                    # Double-check under lock before unloading
                    if self.pipeline and (time.time() - self.last_active > self.idle_timeout):
                        print(f"[LTX-2] Idle timeout reached ({self.idle_timeout}s). Unloading pipeline...")
                        self._unload_internal()

    def _unload_internal(self):
        """Unload without acquiring lock (caller must hold lock)."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
            print("[LTX-2] Pipeline unloaded, VRAM freed.")

    def unload_all(self):
        with self.lock:
            self._unload_internal()

    def touch(self):
        """Reset the idle timer. Call after every generation completes."""
        self.last_active = time.time()

    def get_pipeline(self):
        with self.lock:
            self.last_active = time.time()
            if self.pipeline is not None:
                return self.pipeline
            
            print(f"[LTX-2] Loading LTX-2 Pipeline models...")
            # Create distilled LORA params
            distilled_lora_config = None
            if os.path.exists(DISTILLED_LORA):
                distilled_lora_config = [
                    LoraPathStrengthAndSDOps(DISTILLED_LORA, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)
                ]
            else:
                print(f"[LTX-2] WARNING: Distilled LORA not found at {DISTILLED_LORA}. Will attempt to run without it, but Quality will drop.")

            # To avoid huge footprint, we default FP8 to True in production unless forced otherwise.
            # Expect PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in the environment
            try:
                self.pipeline = TI2VidTwoStagesPipeline(
                    checkpoint_path=LTX_MODEL_PATH,
                    distilled_lora=distilled_lora_config or [],
                    spatial_upsampler_path=SPATIAL_UPSAMPLER,
                    gemma_root=GEMMA_ROOT,
                    loras=[],
                    quantization=QuantizationPolicy.fp8_cast(),  # uses scale-aware upcast (patched)
                )
            except Exception as e:
                print(f"[LTX-2] Initialization failed. Check paths.\nLTX: {LTX_MODEL_PATH}\nGEMMA: {GEMMA_ROOT}\nError: {str(e)}")
                raise e

            return self.pipeline

manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    manager.unload_all()

app = FastAPI(title="LTX-2 Video Generation API", lifespan=lifespan)
try:
    from api_utils import GracefulJSONRoute
    app.router.route_class = GracefulJSONRoute
except ImportError as e:
    logger.warning(f"Could not load GracefulJSONRoute: {e}")

# --- Schemas ---
# For robust JSON handling, Pydantic defaults are set correctly.
class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Chronological, literal description of scene")
    height: int = Field(default=512, description="Base height before upsampling")
    width: int = Field(default=768, description="Base width before upsampling")
    num_frames: int = Field(default=121, description="Number of frames")
    frame_rate: float = Field(default=25.0, description="FPS for the mp4 file")
    num_inference_steps: int = Field(default=60, description="Denoising steps")
    seed: int = Field(default=42, description="Random seed")
    
    # Advanced Optional Guider params mapped
    cfg_scale_video: float = 4.0
    stg_scale_video: float = 1.2
    cfg_scale_audio: float = 7.0
    modality_scale: float = 3.0


# --- Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "running", "device": DEVICE, "vram_loaded": manager.pipeline is not None}

@app.get("/v1/internal/unload")
def manual_unload():
    manager.unload_all()
    return {"status": "models unloaded"}

# --- Async Polling Infrastructure ---
video_tasks: Dict[str, Dict[str, Any]] = {}
generation_lock = threading.Lock()
is_generating = False

@app.get("/outputs/{filename}")
async def get_output_file(filename: str):
    """Serve a generated video file."""
    filepath = os.path.abspath(os.path.join(OUTPUT_DIR, filename))
    if not filepath.startswith(os.path.abspath(OUTPUT_DIR)):
        raise HTTPException(status_code=403, detail="Forbidden")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath, media_type="video/mp4")

@app.get("/v1/video/tasks/{task_id}")
def get_task_status(task_id: str):
    """Poll this endpoint for generation status."""
    if task_id not in video_tasks:
        return {
            "status": "failed",
            "error": "Task not found (server may have restarted or ID is invalid). Please submit a new request."
        }
    return video_tasks[task_id]


def _encode_and_save_video(video, audio, num_frames, frame_rate, task_id: str):
    """Encode video/audio tensors to an MP4 file in OUTPUT_DIR. Returns the URL."""
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    filename = f"ltx2_{task_id}.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)
    encode_video(
        video=video,
        fps=frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=output_path,
        video_chunks_number=video_chunks_number,
    )
    url = f"http://{config.PUBLIC_HOST}:{PORT}/outputs/{filename}"
    return url


def _process_async_t2v(task_id: str, prompt: str, height: int, width: int,
                       num_frames: int, frame_rate: float, num_inference_steps: int,
                       seed: int, cfg_scale_video: float, stg_scale_video: float,
                       cfg_scale_audio: float, modality_scale: float):
    """Background worker for async T2V generation."""
    global is_generating
    with generation_lock:
        is_generating = True
        try:
            pipeline = manager.get_pipeline()
            video_guider = MultiModalGuiderParams(
                cfg_scale=cfg_scale_video, stg_scale=stg_scale_video,
                rescale_scale=0.7, modality_scale=modality_scale,
                skip_step=0, stg_blocks=[29],
            )
            audio_guider = MultiModalGuiderParams(
                cfg_scale=cfg_scale_audio, stg_scale=stg_scale_video,
                rescale_scale=0.7, modality_scale=modality_scale,
                skip_step=0, stg_blocks=[29],
            )
            logger.info(f"[Task {task_id}] T2V generating: '{prompt[:50]}...'")
            with torch.inference_mode():
                video, audio = pipeline(
                    prompt=prompt, negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    seed=seed, height=height, width=width,
                    num_frames=num_frames, frame_rate=frame_rate,
                    num_inference_steps=num_inference_steps,
                    video_guider_params=video_guider, audio_guider_params=audio_guider,
                    images=[]
                )
                url = _encode_and_save_video(video, audio, num_frames, frame_rate, task_id)

            video_tasks[task_id] = {"status": "completed", "url": url}
            logger.info(f"[Task {task_id}] T2V completed.")
        except Exception as e:
            logger.error(f"[Task {task_id}] T2V failed: {e}")
            logger.error(traceback.format_exc())
            video_tasks[task_id] = {"status": "failed", "error": str(e)}
        finally:
            is_generating = False
            manager.touch()


def _process_async_i2v(task_id: str, image_path: str, prompt: str,
                       height: int, width: int, num_frames: int, frame_rate: float,
                       num_inference_steps: int, seed: int,
                       cfg_scale_video: float, stg_scale_video: float):
    """Background worker for async I2V generation."""
    global is_generating
    with generation_lock:
        is_generating = True
        try:
            pipeline = manager.get_pipeline()
            video_guider = MultiModalGuiderParams(
                cfg_scale=cfg_scale_video, stg_scale=stg_scale_video,
                rescale_scale=0.7, modality_scale=3.0,
                skip_step=0, stg_blocks=[29],
            )
            audio_guider = MultiModalGuiderParams(
                cfg_scale=7.0, stg_scale=stg_scale_video,
                rescale_scale=0.7, modality_scale=3.0,
                skip_step=0, stg_blocks=[29],
            )
            image_conditions = [(image_path, 0, 1.0)]
            logger.info(f"[Task {task_id}] I2V generating: '{prompt[:50]}...'")
            with torch.inference_mode():
                video, audio = pipeline(
                    prompt=prompt, negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    seed=seed, height=height, width=width,
                    num_frames=num_frames, frame_rate=frame_rate,
                    num_inference_steps=num_inference_steps,
                    video_guider_params=video_guider, audio_guider_params=audio_guider,
                    images=image_conditions
                )
                url = _encode_and_save_video(video, audio, num_frames, frame_rate, task_id)

            video_tasks[task_id] = {"status": "completed", "url": url}
            logger.info(f"[Task {task_id}] I2V completed.")
        except Exception as e:
            logger.error(f"[Task {task_id}] I2V failed: {e}")
            logger.error(traceback.format_exc())
            video_tasks[task_id] = {"status": "failed", "error": str(e)}
        finally:
            is_generating = False
            manager.touch()
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
            except:
                pass


@app.post("/v1/video/async_t2v")
def async_t2v(req: VideoGenerationRequest):
    """Submit a T2V job. Returns immediately with a task_id. Poll GET /v1/video/tasks/{task_id}."""
    task_id = uuid4().hex
    video_tasks[task_id] = {"status": "processing"}
    t = threading.Thread(
        target=_process_async_t2v, daemon=True,
        args=(task_id, req.prompt, req.height, req.width,
              req.num_frames, req.frame_rate, req.num_inference_steps, req.seed,
              req.cfg_scale_video, req.stg_scale_video, req.cfg_scale_audio, req.modality_scale)
    )
    t.start()
    return {"task_id": task_id, "status": "processing", "message": "Task queued. Poll GET /v1/video/tasks/{task_id}"}


@app.post("/v1/video/async_i2v")
async def async_i2v(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    height: int = Form(512),
    width: int = Form(768),
    num_frames: int = Form(121),
    frame_rate: float = Form(25.0),
    num_inference_steps: int = Form(60),
    seed: int = Form(42),
    cfg_scale_video: float = Form(4.0),
    stg_scale_video: float = Form(1.2)
):
    """Submit an I2V job. Returns immediately with a task_id. Poll GET /v1/video/tasks/{task_id}."""
    import tempfile
    task_id = uuid4().hex

    img_bytes = await image.read()
    ext = os.path.splitext(image.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix="ltx2_in_", mode="wb") as tmp_img:
        tmp_img.write(img_bytes)
        temp_in_path = tmp_img.name

    video_tasks[task_id] = {"status": "processing"}
    t = threading.Thread(
        target=_process_async_i2v, daemon=True,
        args=(task_id, temp_in_path, prompt,
              height, width, num_frames, frame_rate, num_inference_steps, seed,
              cfg_scale_video, stg_scale_video)
    )
    t.start()
    return {"task_id": task_id, "status": "processing", "message": "Task queued. Poll GET /v1/video/tasks/{task_id}"}


@app.get("/v1/internal/status")
async def internal_status():
    return {
        "status": "generating" if is_generating else "idle",
        "model_loaded": manager.pipeline is not None
    }


@app.post("/v1/video/t2v")
def generate_t2v(req: VideoGenerationRequest):
    """Text to Video Generation."""
    # Ensure this runs synchronously within the lock
    with manager.lock:
        try:
            pipeline = manager.get_pipeline()
            
            video_guider = MultiModalGuiderParams(
                cfg_scale=req.cfg_scale_video,
                stg_scale=req.stg_scale_video,
                rescale_scale=0.7,
                modality_scale=req.modality_scale,
                skip_step=0,
                stg_blocks=[29],
            )

            audio_guider = MultiModalGuiderParams(
                cfg_scale=req.cfg_scale_audio,
                stg_scale=req.stg_scale_video, # Usually matches
                rescale_scale=0.7,
                modality_scale=req.modality_scale,
                skip_step=0,
                stg_blocks=[29],
            )

            print(f"[LTX-2] Starting Generation (seed={req.seed}): {req.prompt[:50]}...")
            
            # Temporary file to save the MP4
            import tempfile
            temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix="ltx2_")
            temp_out.close()

            # Execute the pipeline
            with torch.inference_mode():
                video, audio = pipeline(
                    prompt=req.prompt,
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    seed=req.seed,
                    height=req.height,
                    width=req.width,
                    num_frames=req.num_frames,
                    frame_rate=req.frame_rate,
                    num_inference_steps=req.num_inference_steps,
                    video_guider_params=video_guider,
                    audio_guider_params=audio_guider,
                    images=[]
                )
                
                from ltx_pipelines.utils.media_io import encode_video
                from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
                from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    
                tiling_config = TilingConfig.default()
                video_chunks_number = get_video_chunks_number(req.num_frames, tiling_config)
    
                encode_video(
                    video=video,
                    fps=req.frame_rate,
                    audio=audio,
                    audio_sample_rate=AUDIO_SAMPLE_RATE,
                    output_path=temp_out.name,
                    video_chunks_number=video_chunks_number,
                )
            
            # Stream the file back and cleanup
            def iterfile():
                try:
                    with open(temp_out.name, "rb") as f:
                        yield from f
                finally:
                    if os.path.exists(temp_out.name):
                        os.remove(temp_out.name)
            
            return StreamingResponse(iterfile(), media_type="video/mp4")

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            manager.touch()

@app.post("/v1/video/i2v")
async def generate_i2v(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    height: int = Form(512),
    width: int = Form(768),
    num_frames: int = Form(121),
    frame_rate: float = Form(25.0),
    num_inference_steps: int = Form(60),
    seed: int = Form(42),
    cfg_scale_video: float = Form(4.0),
    stg_scale_video: float = Form(1.2)
):
    """Image to Video Generation."""
    
    # Read the image 
    img_bytes = await image.read()
    import tempfile
    
    ext = os.path.splitext(image.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix="ltx2_in_", mode="wb") as tmp_img:
        tmp_img.write(img_bytes)
        temp_in_path = tmp_img.name

    with manager.lock:
        try:
            pipeline = manager.get_pipeline()
            
            video_guider = MultiModalGuiderParams(
                cfg_scale=cfg_scale_video,
                stg_scale=stg_scale_video,
                rescale_scale=0.7,
                modality_scale=3.0,
                skip_step=0,
                stg_blocks=[29],
            )
            audio_guider = MultiModalGuiderParams(
                cfg_scale=7.0,
                stg_scale=stg_scale_video,
                rescale_scale=0.7,
                modality_scale=3.0,
                skip_step=0,
                stg_blocks=[29],
            )

            temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix="ltx2_out_")
            temp_out.close()

            print(f"[LTX-2] Starting I2V Generation (seed={seed}).")
            
            # Format: [(path, frame_index, strength)]
            image_conditions = [(temp_in_path, 0, 1.0)]

            with torch.inference_mode():
                video, audio = pipeline(
                    prompt=prompt,
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    num_inference_steps=num_inference_steps,
                    video_guider_params=video_guider,
                    audio_guider_params=audio_guider,
                    images=image_conditions
                )
    
                from ltx_pipelines.utils.media_io import encode_video
                from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
                from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    
                tiling_config = TilingConfig.default()
                video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
    
                encode_video(
                    video=video,
                    fps=frame_rate,
                    audio=audio,
                    audio_sample_rate=AUDIO_SAMPLE_RATE,
                    output_path=temp_out.name,
                    video_chunks_number=video_chunks_number,
                )

            # Cleanup input image immediately
            if os.path.exists(temp_in_path):
                os.remove(temp_in_path)

            def iterfile():
                try:
                    with open(temp_out.name, "rb") as f:
                        yield from f
                finally:
                    if os.path.exists(temp_out.name):
                        os.remove(temp_out.name)
            
            return StreamingResponse(iterfile(), media_type="video/mp4")

        except Exception as e:
            if os.path.exists(temp_in_path):
                os.remove(temp_in_path)
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            manager.touch()

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
