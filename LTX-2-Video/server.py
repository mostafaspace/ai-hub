import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "LTX-Core", "packages", "ltx-core", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "LTX-Core", "packages", "ltx-pipelines", "src")))

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

# We'll need ltx_pipelines, assuming we run this from a script that sets PYTHONPATH
import ltx_core.model.transformer.model_configurator as mc
print(f"[LTX-2] DEBUG: Loading model_configurator from: {mc.__file__}")

from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline # Keep for reference if needed
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT

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
LTX_MODEL_PATH = getattr(config, "VIDEO_MODEL_BASE", os.path.join(config.HF_HOME, "Lightricks/LTX-2.3/ltx-2.3-22b-dev-fp8.safetensors"))
GEMMA_ROOT = getattr(config, "VIDEO_GEMMA_ROOT", os.path.join(config.HF_HOME, "google/gemma-3-12b-it-qat-q4_0-unquantized"))
DISTILLED_LORA = getattr(config, "VIDEO_DISTILLED_LORA", os.path.join(config.HF_HOME, "Lightricks/LTX-2.3/ltx-2.3-22b-distilled-lora-384.safetensors"))
DISTILLED_LORA_STRENGTH = getattr(config, "VIDEO_DISTILLED_LORA_STRENGTH", 0.6)
SPATIAL_UPSAMPLER = getattr(config, "VIDEO_SPATIAL_UPSAMPLER", os.path.join(config.HF_HOME, "Lightricks/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"))
ENABLE_DISTILLED_LORA = bool(getattr(config, "VIDEO_ENABLE_DISTILLED_LORA", True))
RETRY_WITHOUT_DISTILLED_LORA_ON_OOM = bool(getattr(config, "VIDEO_RETRY_WITHOUT_DISTILLED_LORA_ON_OOM", True))

# Quality-biased defaults based on the official LTX-2 guidance and the linked
# Kaggle notebook: prompt enhancement on, 24 fps, and more stage-1 denoising.
DEFAULT_FRAME_RATE = 24.0
DEFAULT_NUM_INFERENCE_STEPS = 75
DEFAULT_VIDEO_CFG_SCALE = 3.0
DEFAULT_VIDEO_STG_SCALE = 1.0
DEFAULT_AUDIO_CFG_SCALE = 7.0
DEFAULT_MODALITY_SCALE = 3.0
DEFAULT_RESCALE_SCALE = 0.7
DEFAULT_STG_BLOCKS = [29]
DEFAULT_I2V_CONDITIONING_STRENGTH = 1.0


# --- Model Management (Auto-Unload via Thread RLock) ---
class ModelManager:
    def __init__(self, idle_timeout: float = 60.0, check_interval: float = 10.0):
        self.pipeline = None
        self.use_distilled_lora = ENABLE_DISTILLED_LORA
        self.last_active = time.time()
        self.idle_timeout = idle_timeout
        self.check_interval = check_interval
        self.lock = threading.RLock()
        
        self.monitor_thread = threading.Thread(target=self._idle_monitor, daemon=True)
        self.monitor_thread.start()

    def _idle_monitor(self):
        global is_generating
        while True:
            time.sleep(self.check_interval)
            # Lock-free check first (reading a float is atomic in CPython)
            if self.pipeline and not is_generating and (time.time() - self.last_active > self.idle_timeout):
                with self.lock:
                    # Double-check under lock before unloading
                    if self.pipeline and not is_generating and (time.time() - self.last_active > self.idle_timeout):
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

    def set_use_distilled_lora(self, enabled: bool, unload_existing: bool = True):
        with self.lock:
            self.use_distilled_lora = bool(enabled)
            if unload_existing:
                self._unload_internal()

    def touch(self):
        """Reset the idle timer. Call after every generation completes."""
        self.last_active = time.time()

    def get_pipeline(self):
        with self.lock:
            self.last_active = time.time()
            if self.pipeline is not None:
                return self.pipeline
            
            print(f"[LTX-2] Loading LTX-2 Pipeline models... distilled_lora={'on' if self.use_distilled_lora else 'off'}")
            # Create distilled LORA params
            distilled_lora_config = None
            if self.use_distilled_lora and os.path.exists(DISTILLED_LORA):
                distilled_lora_config = [
                    # The official two-stage examples use a milder 0.6 distilled
                    # LoRA strength for cleaner refinement with the full model.
                    LoraPathStrengthAndSDOps(
                        DISTILLED_LORA,
                        DISTILLED_LORA_STRENGTH,
                        LTXV_LORA_COMFY_RENAMING_MAP,
                    )
                ]
            elif self.use_distilled_lora:
                print(f"[LTX-2] WARNING: Distilled LORA not found at {DISTILLED_LORA}. Will attempt to run without it, but Quality will drop.")

            # To avoid huge footprint, we default FP8 to True in production unless forced otherwise.
            # Expect PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in the environment
            try:
                self.pipeline = TI2VidOneStagePipeline(
                    checkpoint_path=LTX_MODEL_PATH,
                    device=DEVICE,
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
    height: int = Field(default=576, description="Base height (1-stage native)")
    width: int = Field(default=1024, description="Base width (1-stage native)")
    num_frames: int = Field(default=121, description="Number of frames")
    frame_rate: float = Field(default=DEFAULT_FRAME_RATE, description="FPS for the mp4 file")
    num_inference_steps: int = Field(
        default=DEFAULT_NUM_INFERENCE_STEPS,
        description="Stage-1 denoising steps. Higher values improve quality at the cost of speed.",
    )
    seed: int = Field(default=42, description="Random seed")
    negative_prompt: str = Field(
        default=DEFAULT_NEGATIVE_PROMPT,
        description="Prompt describing artifacts or unwanted traits to avoid.",
    )
    enhance_prompt: bool = Field(
        default=True,
        description="Use Gemma prompt enhancement before generation for better scene detail and motion planning.",
    )
    
    # Advanced Optional Guider params mapped
    cfg_scale_video: float = DEFAULT_VIDEO_CFG_SCALE
    stg_scale_video: float = DEFAULT_VIDEO_STG_SCALE
    cfg_scale_audio: float = DEFAULT_AUDIO_CFG_SCALE
    modality_scale: float = DEFAULT_MODALITY_SCALE


def _build_guiders(
    cfg_scale_video: float,
    stg_scale_video: float,
    cfg_scale_audio: float,
    modality_scale: float,
) -> tuple[MultiModalGuiderParams, MultiModalGuiderParams]:
    video_guider = MultiModalGuiderParams(
        cfg_scale=cfg_scale_video,
        stg_scale=stg_scale_video,
        rescale_scale=DEFAULT_RESCALE_SCALE,
        modality_scale=modality_scale,
        skip_step=0,
        stg_blocks=DEFAULT_STG_BLOCKS,
    )
    audio_guider = MultiModalGuiderParams(
        cfg_scale=cfg_scale_audio,
        stg_scale=stg_scale_video,
        rescale_scale=DEFAULT_RESCALE_SCALE,
        modality_scale=modality_scale,
        skip_step=0,
        stg_blocks=DEFAULT_STG_BLOCKS,
    )
    return video_guider, audio_guider


def _warn_if_quality_risky(height: int, width: int, num_frames: int, frame_rate: float) -> None:
    stage_1_height = height // 2
    stage_1_width = width // 2
    if min(stage_1_height, stage_1_width) < 256:
        logger.warning(
            "[LTX-2] Low base resolution detected (%sx%s -> stage 1 %sx%s). "
            "This can increase artifacts in two-stage generation.",
            width,
            height,
            stage_1_width,
            stage_1_height,
        )
    if num_frames > 241:
        logger.info(
            "[LTX-2] Long generation requested (%s frames at %.1f fps). "
            "Shorter clips usually produce cleaner motion and fewer artifacts.",
            num_frames,
            frame_rate,
        )


def _resolve_enhance_prompt(prompt: str, enhance_prompt: bool, has_image_conditioning: bool) -> bool:
    if not enhance_prompt:
        return False

    char_limit = 220 if has_image_conditioning else 500
    if len(prompt) > char_limit:
        logger.info(
            "[LTX-2] Disabling prompt enhancement for a %s prompt (%s chars > %s-char safety limit).",
            "conditioned" if has_image_conditioning else "text-only",
            len(prompt),
            char_limit,
        )
        return False

    return True


def _encode_video_file(video, audio, num_frames: int, frame_rate: float, output_path: str) -> None:
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
    from ltx_core.model.video_vae import (
        TilingConfig,
        get_video_chunks_number,
        SpatialTilingConfig,
        TemporalTilingConfig,
    )

    tiling_config = TilingConfig(
        spatial_config=SpatialTilingConfig(tile_size_in_pixels=384, tile_overlap_in_pixels=64),
        temporal_config=TemporalTilingConfig(tile_size_in_frames=64, tile_overlap_in_frames=24),
    )
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
    encode_video(
        video=video,
        fps=frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=output_path,
        video_chunks_number=video_chunks_number,
    )


# --- Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "running", "device": DEVICE, "vram_loaded": manager.pipeline is not None}

@app.post("/v1/internal/unload")
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
    filename = f"ltx2_{task_id}.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)
    _encode_video_file(video, audio, num_frames, frame_rate, output_path)
    url = f"http://{config.PUBLIC_HOST}:{PORT}/outputs/{filename}"
    return url


def _cleanup_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oom_error(exc: Exception) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _run_t2v_once(
    task_id: str,
    prompt: str,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    num_inference_steps: int,
    seed: int,
    cfg_scale_video: float,
    stg_scale_video: float,
    cfg_scale_audio: float,
    modality_scale: float,
    negative_prompt: str,
    enhance_prompt: bool,
) -> str:
    pipeline = manager.get_pipeline()
    _warn_if_quality_risky(height, width, num_frames, frame_rate)
    video_guider, audio_guider = _build_guiders(
        cfg_scale_video=cfg_scale_video,
        stg_scale_video=stg_scale_video,
        cfg_scale_audio=cfg_scale_audio,
        modality_scale=modality_scale,
    )
    enhance_prompt = _resolve_enhance_prompt(
        prompt=prompt,
        enhance_prompt=enhance_prompt,
        has_image_conditioning=False,
    )
    logger.info(f"[Task {task_id}] T2V generating: '{prompt[:50]}...'")
    with torch.inference_mode():
        video, audio = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            video_guider_params=video_guider,
            audio_guider_params=audio_guider,
            images=[],
            enhance_prompt=enhance_prompt,
        )
        return _encode_and_save_video(video, audio, num_frames, frame_rate, task_id)


def _run_i2v_once(
    task_id: str,
    image_path: str,
    prompt: str,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    num_inference_steps: int,
    seed: int,
    cfg_scale_video: float,
    stg_scale_video: float,
    modality_scale: float,
    conditioning_strength: float,
    negative_prompt: str,
    enhance_prompt: bool,
) -> str:
    pipeline = manager.get_pipeline()
    _warn_if_quality_risky(height, width, num_frames, frame_rate)
    video_guider, audio_guider = _build_guiders(
        cfg_scale_video=cfg_scale_video,
        stg_scale_video=stg_scale_video,
        cfg_scale_audio=DEFAULT_AUDIO_CFG_SCALE,
        modality_scale=modality_scale,
    )
    condition_strength = max(0.0, min(float(conditioning_strength), 1.0))
    image_conditions = [(image_path, 0, condition_strength)]
    enhance_prompt = _resolve_enhance_prompt(
        prompt=prompt,
        enhance_prompt=enhance_prompt,
        has_image_conditioning=True,
    )
    logger.info(f"[Task {task_id}] I2V generating: '{prompt[:50]}...'")
    with torch.inference_mode():
        video, audio = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            video_guider_params=video_guider,
            audio_guider_params=audio_guider,
            images=image_conditions,
            enhance_prompt=enhance_prompt,
        )
        return _encode_and_save_video(video, audio, num_frames, frame_rate, task_id)


def _maybe_retry_without_distilled_lora(task_id: str, exc: Exception) -> bool:
    if not _is_oom_error(exc):
        return False
    if not RETRY_WITHOUT_DISTILLED_LORA_ON_OOM:
        return False
    if not manager.use_distilled_lora:
        return False

    logger.warning(
        "[Task %s] OOM while distilled LoRA is enabled. Unloading and retrying without distilled LoRA.",
        task_id,
    )
    manager.set_use_distilled_lora(False, unload_existing=True)
    _cleanup_cuda_memory()
    return True


def _process_async_t2v(task_id: str, prompt: str, height: int, width: int,
                       num_frames: int, frame_rate: float, num_inference_steps: int,
                       seed: int, cfg_scale_video: float, stg_scale_video: float,
                       cfg_scale_audio: float, modality_scale: float,
                       negative_prompt: str, enhance_prompt: bool):
    """Background worker for async T2V generation."""
    global is_generating
    with generation_lock:
        is_generating = True
        try:
            try:
                url = _run_t2v_once(
                    task_id, prompt, height, width, num_frames, frame_rate, num_inference_steps,
                    seed, cfg_scale_video, stg_scale_video, cfg_scale_audio, modality_scale,
                    negative_prompt, enhance_prompt,
                )
            except Exception as first_exc:
                if not _maybe_retry_without_distilled_lora(task_id, first_exc):
                    raise
                url = _run_t2v_once(
                    task_id, prompt, height, width, num_frames, frame_rate, num_inference_steps,
                    seed, cfg_scale_video, stg_scale_video, cfg_scale_audio, modality_scale,
                    negative_prompt, enhance_prompt,
                )
            video_tasks[task_id] = {"status": "completed", "url": url}
            logger.info(f"[Task {task_id}] T2V completed.")
        except Exception as e:
            logger.error(f"[Task {task_id}] T2V failed: {e}")
            logger.error(traceback.format_exc())
            video_tasks[task_id] = {"status": "failed", "error": str(e)}
        finally:
            is_generating = False
            manager.touch()
            _cleanup_cuda_memory()


def _process_async_i2v(task_id: str, image_path: str, prompt: str,
                       height: int, width: int, num_frames: int, frame_rate: float,
                       num_inference_steps: int, seed: int,
                       cfg_scale_video: float, stg_scale_video: float,
                       modality_scale: float,
                       conditioning_strength: float,
                       negative_prompt: str, enhance_prompt: bool):
    """Background worker for async I2V generation."""
    global is_generating
    with generation_lock:
        is_generating = True
        try:
            try:
                url = _run_i2v_once(
                    task_id, image_path, prompt, height, width, num_frames, frame_rate,
                    num_inference_steps, seed, cfg_scale_video, stg_scale_video, modality_scale,
                    conditioning_strength,
                    negative_prompt, enhance_prompt,
                )
            except Exception as first_exc:
                if not _maybe_retry_without_distilled_lora(task_id, first_exc):
                    raise
                url = _run_i2v_once(
                    task_id, image_path, prompt, height, width, num_frames, frame_rate,
                    num_inference_steps, seed, cfg_scale_video, stg_scale_video, modality_scale,
                    conditioning_strength,
                    negative_prompt, enhance_prompt,
                )
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
            _cleanup_cuda_memory()


@app.post("/v1/video/async_t2v")
def async_t2v(req: VideoGenerationRequest):
    """Submit a T2V job. Returns immediately with a task_id. Poll GET /v1/video/tasks/{task_id}."""
    task_id = uuid4().hex
    video_tasks[task_id] = {"status": "processing"}
    t = threading.Thread(
        target=_process_async_t2v, daemon=True,
        args=(task_id, req.prompt, req.height, req.width,
              req.num_frames, req.frame_rate, req.num_inference_steps, req.seed,
              req.cfg_scale_video, req.stg_scale_video, req.cfg_scale_audio, req.modality_scale,
              req.negative_prompt, req.enhance_prompt)
    )
    t.start()
    return {"task_id": task_id, "status": "processing", "message": "Task queued. Poll GET /v1/video/tasks/{task_id}"}


@app.post("/v1/video/async_i2v")
async def async_i2v(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    height: int = Form(576),
    width: int = Form(1024),
    num_frames: int = Form(121),
    frame_rate: float = Form(DEFAULT_FRAME_RATE),
    num_inference_steps: int = Form(DEFAULT_NUM_INFERENCE_STEPS),
    seed: int = Form(42),
    cfg_scale_video: float = Form(DEFAULT_VIDEO_CFG_SCALE),
    stg_scale_video: float = Form(DEFAULT_VIDEO_STG_SCALE),
    modality_scale: float = Form(DEFAULT_MODALITY_SCALE),
    conditioning_strength: float = Form(DEFAULT_I2V_CONDITIONING_STRENGTH),
    negative_prompt: str = Form(DEFAULT_NEGATIVE_PROMPT),
    enhance_prompt: bool = Form(True),
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
              cfg_scale_video, stg_scale_video, modality_scale, conditioning_strength, negative_prompt, enhance_prompt)
    )
    t.start()
    return {"task_id": task_id, "status": "processing", "message": "Task queued. Poll GET /v1/video/tasks/{task_id}"}


@app.get("/v1/internal/status")
async def internal_status():
    return {
        "status": "generating" if is_generating else "idle",
        "model_loaded": manager.pipeline is not None,
        "distilled_lora_enabled": manager.use_distilled_lora,
    }


@app.post("/v1/video/t2v")
def generate_t2v(req: VideoGenerationRequest):
    """Text to Video Generation."""
    # Ensure this runs synchronously within the lock
    with manager.lock:
        try:
            pipeline = manager.get_pipeline()
            _warn_if_quality_risky(req.height, req.width, req.num_frames, req.frame_rate)
            video_guider, audio_guider = _build_guiders(
                cfg_scale_video=req.cfg_scale_video,
                stg_scale_video=req.stg_scale_video,
                cfg_scale_audio=req.cfg_scale_audio,
                modality_scale=req.modality_scale,
            )
            enhance_prompt = _resolve_enhance_prompt(
                prompt=req.prompt,
                enhance_prompt=req.enhance_prompt,
                has_image_conditioning=False,
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
                    negative_prompt=req.negative_prompt,
                    seed=req.seed,
                    height=req.height,
                    width=req.width,
                    num_frames=req.num_frames,
                    frame_rate=req.frame_rate,
                    num_inference_steps=req.num_inference_steps,
                    video_guider_params=video_guider,
                    audio_guider_params=audio_guider,
                    images=[],
                    enhance_prompt=enhance_prompt,
                )
                _encode_video_file(video, audio, req.num_frames, req.frame_rate, temp_out.name)
            
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
    height: int = Form(576),
    width: int = Form(1024),
    num_frames: int = Form(121),
    frame_rate: float = Form(DEFAULT_FRAME_RATE),
    num_inference_steps: int = Form(DEFAULT_NUM_INFERENCE_STEPS),
    seed: int = Form(42),
    cfg_scale_video: float = Form(DEFAULT_VIDEO_CFG_SCALE),
    stg_scale_video: float = Form(DEFAULT_VIDEO_STG_SCALE),
    modality_scale: float = Form(DEFAULT_MODALITY_SCALE),
    conditioning_strength: float = Form(DEFAULT_I2V_CONDITIONING_STRENGTH),
    negative_prompt: str = Form(DEFAULT_NEGATIVE_PROMPT),
    enhance_prompt: bool = Form(True),
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
            _warn_if_quality_risky(height, width, num_frames, frame_rate)
            video_guider, audio_guider = _build_guiders(
                cfg_scale_video=cfg_scale_video,
                stg_scale_video=stg_scale_video,
                cfg_scale_audio=DEFAULT_AUDIO_CFG_SCALE,
                modality_scale=modality_scale,
            )

            temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix="ltx2_out_")
            temp_out.close()

            print(f"[LTX-2] Starting I2V Generation (seed={seed}).")
            
            # Format: [(path, frame_index, strength)]
            condition_strength = max(0.0, min(float(conditioning_strength), 1.0))
            image_conditions = [(temp_in_path, 0, condition_strength)]
            enhance_prompt = _resolve_enhance_prompt(
                prompt=prompt,
                enhance_prompt=enhance_prompt,
                has_image_conditioning=True,
            )

            with torch.inference_mode():
                video, audio = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    num_inference_steps=num_inference_steps,
                    video_guider_params=video_guider,
                    audio_guider_params=audio_guider,
                    images=image_conditions,
                    enhance_prompt=enhance_prompt,
                )
                _encode_video_file(video, audio, num_frames, frame_rate, temp_out.name)

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
