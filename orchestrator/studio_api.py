import asyncio
import json
import os
import re
import time
import uuid
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import httpx
import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import config
from orchestrator.studio_media import (
    FORMAT_PROFILES,
    build_timeline_plan,
    burn_subtitles_to_video,
    create_contact_sheet,
    detect_duration,
    extract_audio_for_transcription,
    extract_thumbnail,
    ffmpeg_available,
    resolve_format_profile,
    transcode_media,
    write_approximate_srt,
)


studio_router = APIRouter(prefix="/v1/studio", tags=["Practical Studio"])

ROOT_DIR = os.path.dirname(__file__)
PROJECTS_DIR = os.path.join(ROOT_DIR, "studio_projects")
PACKS_DIR = os.path.join(ROOT_DIR, "studio_character_packs")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "director_outputs", "practical")
REGISTRY_PATH = os.path.join(ROOT_DIR, "models.yaml")
MAX_TRACKED_TASKS = 100
STUDIO_TASKS: dict[str, dict[str, Any]] = {}

for path in (PROJECTS_DIR, PACKS_DIR, OUTPUTS_DIR):
    os.makedirs(path, exist_ok=True)


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _now() -> float:
    return time.time()


def _safe_slug(value: str, fallback: str = "item") -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", (value or "").strip()).strip("-").lower()
    return slug or fallback


def _project_path(project_id: str) -> str:
    return os.path.join(PROJECTS_DIR, f"{project_id}.json")


def _pack_path(pack_id: str) -> str:
    return os.path.join(PACKS_DIR, f"{pack_id}.json")


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def _list_documents(directory: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for name in os.listdir(directory):
        if not name.endswith(".json"):
            continue
        try:
            items.append(_read_json(os.path.join(directory, name)))
        except Exception:
            continue
    items.sort(key=lambda item: float(item.get("updated_at", item.get("created_at", 0.0))), reverse=True)
    return items


def _load_project(project_id: str) -> dict[str, Any]:
    path = _project_path(project_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Unknown project: {project_id}")
    return _read_json(path)


def _save_project(project: dict[str, Any]) -> dict[str, Any]:
    project["updated_at"] = _now()
    _write_json(_project_path(project["project_id"]), project)
    return project


def _load_pack(pack_id: str) -> dict[str, Any]:
    path = _pack_path(pack_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Unknown character pack: {pack_id}")
    return _read_json(path)


def _save_pack(pack: dict[str, Any]) -> dict[str, Any]:
    pack["updated_at"] = _now()
    _write_json(_pack_path(pack["pack_id"]), pack)
    return pack


def _relative_output_url(abs_path: str) -> str:
    rel_path = os.path.relpath(abs_path, os.path.join(ROOT_DIR, "director_outputs")).replace("\\", "/")
    host = getattr(config, "PUBLIC_HOST", None) or getattr(config, "HOST", "127.0.0.1")
    if host == "0.0.0.0":
        host = "127.0.0.1"
    return f"http://{host}:{config.ORCHESTRATOR_PORT}/outputs/{rel_path}"


def _output_path(task_id: str, filename: str) -> str:
    task_dir = os.path.join(OUTPUTS_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    return os.path.join(task_dir, filename)


def _load_backends() -> dict[str, str]:
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load models.yaml: {exc}") from exc
    return data.get("backends", {}) or {}


def _require_backend(name: str) -> str:
    backends = _load_backends()
    if name not in backends:
        raise HTTPException(status_code=503, detail=f"{name} backend not configured in registry.")
    return backends[name]


def _resolve_remote_url(url: str, backend_url: Optional[str] = None) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if backend_url:
        return urljoin(f"{backend_url}/", url.lstrip("/"))
    host = getattr(config, "PUBLIC_HOST", None) or getattr(config, "HOST", "127.0.0.1")
    if host == "0.0.0.0":
        host = "127.0.0.1"
    return urljoin(f"http://{host}:{config.ORCHESTRATOR_PORT}/", url.lstrip("/"))


def _guess_extension(url: str, default_ext: str) -> str:
    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    return ext or default_ext


def _attach_asset(project_id: Optional[str], kind: str, label: str, url: str, metadata: Optional[dict[str, Any]] = None) -> None:
    if not project_id:
        return
    project = _load_project(project_id)
    project.setdefault("assets", []).append(
        {
            "asset_id": str(uuid.uuid4()),
            "kind": kind,
            "label": label,
            "url": url,
            "metadata": metadata or {},
            "created_at": _now(),
        }
    )
    _save_project(project)


def _trim_tasks() -> None:
    if len(STUDIO_TASKS) <= MAX_TRACKED_TASKS:
        return
    ordered_ids = sorted(STUDIO_TASKS, key=lambda key: STUDIO_TASKS[key].get("updated_at", 0.0))
    for task_id in ordered_ids[:-MAX_TRACKED_TASKS]:
        STUDIO_TASKS.pop(task_id, None)


def _create_task(kind: str, project_id: Optional[str] = None) -> dict[str, Any]:
    task_id = str(uuid.uuid4())
    task = {
        "task_id": task_id,
        "kind": kind,
        "status": "processing",
        "project_id": project_id,
        "created_at": _now(),
        "updated_at": _now(),
        "result": None,
        "error": None,
    }
    STUDIO_TASKS[task_id] = task
    _trim_tasks()
    return task


def _complete_task(task_id: str, result: dict[str, Any]) -> None:
    task = STUDIO_TASKS[task_id]
    task["status"] = "completed"
    task["result"] = result
    task["updated_at"] = _now()


def _fail_task(task_id: str, error: str) -> None:
    task = STUDIO_TASKS[task_id]
    task["status"] = "failed"
    task["error"] = error
    task["updated_at"] = _now()


async def _run_task(task_id: str, runner) -> None:
    try:
        result = await runner(task_id)
        _complete_task(task_id, result)
    except Exception as exc:
        _fail_task(task_id, str(exc))


def start_background_task(task_id: str, runner) -> None:
    asyncio.create_task(_run_task(task_id, runner))


async def _download_to_path(url: str, output_path: str, timeout: float = 300.0) -> str:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
    with open(output_path, "wb") as handle:
        handle.write(response.content)
    return output_path


async def _poll_task_url(client: httpx.AsyncClient, status_url: str, backend_url: str) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + (30 * 60)
    while True:
        if asyncio.get_running_loop().time() > deadline:
            raise RuntimeError(f"Timed out polling task: {status_url}")
        response = await client.get(status_url)
        response.raise_for_status()
        data = response.json()
        status = str(data.get("status", "")).lower()
        if status in {"completed", "succeeded"}:
            return data
        if status in {"failed", "error"}:
            raise RuntimeError(data.get("error") or f"Task failed: {data}")
        await asyncio.sleep(2.0)


def _apply_character_pack(prompt: str, pack: Optional[dict[str, Any]]) -> str:
    if not pack:
        return prompt.strip()
    parts = [pack.get("prompt_prefix", "").strip(), prompt.strip(), pack.get("prompt_suffix", "").strip()]
    return " ".join(part for part in parts if part)


class ProjectCreateRequest(BaseModel):
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    default_profile: str = "youtube_short"
    notes: str = ""


class ProjectAssetRequest(BaseModel):
    kind: str
    label: str
    url: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CharacterPackCreateRequest(BaseModel):
    name: str
    voice: str = "Vivian"
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    negative_prompt: str = ""
    style_notes: str = ""
    reference_asset_urls: list[str] = Field(default_factory=list)
    default_image_options: dict[str, Any] = Field(default_factory=dict)
    default_voice_options: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class TimelineClip(BaseModel):
    asset_url: str
    label: str = ""
    start_sec: float = 0.0
    duration_sec: Optional[float] = None
    trim_in_sec: float = 0.0
    track: int = 0


class TimelineAudioTrack(BaseModel):
    asset_url: str
    label: str = ""
    start_sec: float = 0.0
    duration_sec: Optional[float] = None
    volume: float = 1.0


class TimelineSubtitleTrack(BaseModel):
    subtitle_url: str
    label: str = ""


class TimelineUpdateRequest(BaseModel):
    video_clips: list[TimelineClip] = Field(default_factory=list)
    audio_tracks: list[TimelineAudioTrack] = Field(default_factory=list)
    subtitle_tracks: list[TimelineSubtitleTrack] = Field(default_factory=list)
    notes: str = ""


class TimelinePlanRequest(BaseModel):
    format_profile: Optional[str] = None


class CaptionTaskRequest(BaseModel):
    media_url: str
    burn_in: bool = False
    project_id: Optional[str] = None
    output_profile: str = "youtube_short"
    prompt: str = "Transcribe this media accurately."
    label: str = "captions"


class ThumbnailTaskRequest(BaseModel):
    media_url: str
    project_id: Optional[str] = None
    timestamp_sec: float = 0.0
    label: str = "thumbnail"


class ContactSheetTaskRequest(BaseModel):
    media_url: str
    project_id: Optional[str] = None
    columns: int = 4
    rows: int = 2
    thumb_width: int = 320
    label: str = "contact-sheet"


class TranscodeTaskRequest(BaseModel):
    media_url: str
    profile_name: str
    project_id: Optional[str] = None
    label: str = "transcode"


class VoiceAuditionRequest(BaseModel):
    text: str
    voices: list[str] = Field(default_factory=list)
    project_id: Optional[str] = None
    character_pack_id: Optional[str] = None
    response_format: str = "mp3"
    language: Optional[str] = None
    instruct: Optional[str] = None
    label: str = "voice-audition"


class PromptCompareRequest(BaseModel):
    prompt_variants: list[str] = Field(default_factory=list)
    shared_options: dict[str, Any] = Field(default_factory=dict)
    project_id: Optional[str] = None
    character_pack_id: Optional[str] = None
    label: str = "prompt-compare"


@studio_router.get("/format-profiles")
def list_format_profiles():
    return {"profiles": FORMAT_PROFILES}


@studio_router.get("/tasks")
def list_tasks():
    tasks = sorted(STUDIO_TASKS.values(), key=lambda item: float(item.get("updated_at", item.get("created_at", 0.0))), reverse=True)
    return {"tasks": tasks}

@studio_router.get("/tasks/{task_id}")
def get_task(task_id: str):
    if task_id not in STUDIO_TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
    return STUDIO_TASKS[task_id]


@studio_router.get("/projects")
def list_projects():
    return {"projects": _list_documents(PROJECTS_DIR)}


@studio_router.post("/projects")
def create_project(req: ProjectCreateRequest):
    resolve_format_profile(req.default_profile)
    project_id = f"proj-{_safe_slug(req.name, 'project')}-{uuid.uuid4().hex[:8]}"
    project = {
        "project_id": project_id,
        "name": req.name,
        "description": req.description,
        "tags": req.tags,
        "default_profile": req.default_profile,
        "notes": req.notes,
        "created_at": _now(),
        "updated_at": _now(),
        "assets": [],
        "timeline": {
            "video_clips": [],
            "audio_tracks": [],
            "subtitle_tracks": [],
            "notes": "",
            "last_render_plan": None,
        },
    }
    _write_json(_project_path(project_id), project)
    return project


@studio_router.get("/projects/{project_id}")
def get_project(project_id: str):
    return _load_project(project_id)


@studio_router.post("/projects/{project_id}/assets")
def add_project_asset(project_id: str, req: ProjectAssetRequest):
    project = _load_project(project_id)
    project.setdefault("assets", []).append(
        {
            "asset_id": str(uuid.uuid4()),
            "kind": req.kind,
            "label": req.label,
            "url": req.url,
            "metadata": req.metadata,
            "created_at": _now(),
        }
    )
    return _save_project(project)


@studio_router.get("/projects/{project_id}/timeline")
def get_project_timeline(project_id: str):
    project = _load_project(project_id)
    return {"project_id": project_id, "timeline": project.get("timeline", {})}


@studio_router.put("/projects/{project_id}/timeline")
def update_project_timeline(project_id: str, req: TimelineUpdateRequest):
    project = _load_project(project_id)
    timeline = _model_dump(req)
    timeline["last_render_plan"] = project.get("timeline", {}).get("last_render_plan")
    project["timeline"] = timeline
    return _save_project(project)


@studio_router.post("/projects/{project_id}/timeline/plan")
def build_project_timeline_plan(project_id: str, req: TimelinePlanRequest):
    project = _load_project(project_id)
    profile_name = req.format_profile or project.get("default_profile", "youtube_short")
    resolve_format_profile(profile_name)
    timeline = project.get("timeline", {})
    plan = build_timeline_plan(
        timeline.get("video_clips", []),
        timeline.get("audio_tracks", []),
        timeline.get("subtitle_tracks", []),
        profile_name,
    )
    timeline["last_render_plan"] = plan
    project["timeline"] = timeline
    _save_project(project)
    return {"project_id": project_id, "plan": plan}


@studio_router.get("/character-packs")
def list_character_packs():
    return {"character_packs": _list_documents(PACKS_DIR)}


@studio_router.post("/character-packs")
def create_character_pack(req: CharacterPackCreateRequest):
    pack_id = f"pack-{_safe_slug(req.name, 'character')}-{uuid.uuid4().hex[:8]}"
    pack = {
        "pack_id": pack_id,
        "created_at": _now(),
        "updated_at": _now(),
        **_model_dump(req),
    }
    _write_json(_pack_path(pack_id), pack)
    return pack


@studio_router.get("/character-packs/{pack_id}")
def get_character_pack(pack_id: str):
    return _load_pack(pack_id)


async def _run_caption_task(task_id: str, req: CaptionTaskRequest) -> dict[str, Any]:
    if req.project_id:
        _load_project(req.project_id)
    if req.burn_in and not ffmpeg_available():
        raise RuntimeError("FFmpeg is required for subtitle burn-in.")

    asr_backend = _require_backend("asr")
    source_ext = _guess_extension(req.media_url, ".bin")
    source_path = _output_path(task_id, f"source{source_ext}")
    await _download_to_path(_resolve_remote_url(req.media_url), source_path)

    video_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
    working_audio_path = source_path
    duration_target = source_path
    is_video = source_ext in video_exts
    if is_video:
        working_audio_path = _output_path(task_id, "transcription.wav")
        ok, message = await asyncio.to_thread(extract_audio_for_transcription, source_path, working_audio_path)
        if not ok:
            raise RuntimeError(f"Failed to extract audio for captions: {message}")
        duration_target = source_path

    duration_seconds = await asyncio.to_thread(detect_duration, duration_target)

    async with httpx.AsyncClient(timeout=None) as client:
        with open(working_audio_path, "rb") as audio_handle:
            response = await client.post(
                f"{asr_backend}/v1/audio/transcriptions",
                files={"file": (os.path.basename(working_audio_path), audio_handle, "audio/wav")},
                data={"prompt": req.prompt},
            )
        response.raise_for_status()
        transcript = (response.json() or {}).get("text", "").strip()

    if not transcript:
        raise RuntimeError("ASR did not return transcript text.")

    srt_path = _output_path(task_id, "captions.srt")
    await asyncio.to_thread(write_approximate_srt, transcript, duration_seconds, srt_path)

    result = {
        "task_id": task_id,
        "status": "completed",
        "transcript": transcript,
        "subtitle_url": _relative_output_url(srt_path),
        "approximate_timing": True,
    }

    _attach_asset(req.project_id, "subtitle", req.label, result["subtitle_url"], {"transcript": transcript})

    if req.burn_in:
        if not is_video:
            raise RuntimeError("Subtitle burn-in currently requires a video input.")
        profile = resolve_format_profile(req.output_profile)
        if profile["kind"] != "video":
            raise RuntimeError("Subtitle burn-in requires a video output profile.")
        burned_path = _output_path(task_id, f"captioned.{profile['container']}")
        ok, message = await asyncio.to_thread(
            burn_subtitles_to_video,
            source_path,
            srt_path,
            burned_path,
            req.output_profile,
        )
        if not ok:
            raise RuntimeError(f"Failed to burn subtitles: {message}")
        result["burned_video_url"] = _relative_output_url(burned_path)
        _attach_asset(req.project_id, "video", f"{req.label}-burned", result["burned_video_url"], {"profile": req.output_profile})

    return result


async def _run_thumbnail_task(task_id: str, req: ThumbnailTaskRequest) -> dict[str, Any]:
    if req.project_id:
        _load_project(req.project_id)
    if not ffmpeg_available():
        raise RuntimeError("FFmpeg is required for thumbnail generation.")

    source_ext = _guess_extension(req.media_url, ".mp4")
    source_path = _output_path(task_id, f"source{source_ext}")
    await _download_to_path(_resolve_remote_url(req.media_url), source_path)

    output_path = _output_path(task_id, "thumbnail.jpg")
    ok, message = await asyncio.to_thread(extract_thumbnail, source_path, output_path, req.timestamp_sec)
    if not ok:
        raise RuntimeError(f"Failed to create thumbnail: {message}")

    output_url = _relative_output_url(output_path)
    _attach_asset(req.project_id, "image", req.label, output_url, {"timestamp_sec": req.timestamp_sec})
    return {"task_id": task_id, "status": "completed", "thumbnail_url": output_url}


async def _run_contact_sheet_task(task_id: str, req: ContactSheetTaskRequest) -> dict[str, Any]:
    if req.project_id:
        _load_project(req.project_id)
    if not ffmpeg_available():
        raise RuntimeError("FFmpeg is required for contact sheet generation.")

    source_ext = _guess_extension(req.media_url, ".mp4")
    source_path = _output_path(task_id, f"source{source_ext}")
    await _download_to_path(_resolve_remote_url(req.media_url), source_path)

    output_path = _output_path(task_id, "contact-sheet.jpg")
    ok, message = await asyncio.to_thread(
        create_contact_sheet,
        source_path,
        output_path,
        req.columns,
        req.rows,
        req.thumb_width,
    )
    if not ok:
        raise RuntimeError(f"Failed to create contact sheet: {message}")

    output_url = _relative_output_url(output_path)
    _attach_asset(
        req.project_id,
        "image",
        req.label,
        output_url,
        {"columns": req.columns, "rows": req.rows, "thumb_width": req.thumb_width},
    )
    return {"task_id": task_id, "status": "completed", "contact_sheet_url": output_url}


async def _run_transcode_task(task_id: str, req: TranscodeTaskRequest) -> dict[str, Any]:
    if req.project_id:
        _load_project(req.project_id)
    if not ffmpeg_available():
        raise RuntimeError("FFmpeg is required for transcoding.")

    profile = resolve_format_profile(req.profile_name)
    source_ext = _guess_extension(req.media_url, ".bin")
    source_path = _output_path(task_id, f"source{source_ext}")
    await _download_to_path(_resolve_remote_url(req.media_url), source_path)

    output_path = _output_path(task_id, f"transcoded.{profile['container']}")
    ok, message = await asyncio.to_thread(transcode_media, source_path, output_path, req.profile_name)
    if not ok:
        raise RuntimeError(f"Failed to transcode media: {message}")

    output_url = _relative_output_url(output_path)
    _attach_asset(req.project_id, profile["kind"], req.label, output_url, {"profile_name": req.profile_name})
    return {"task_id": task_id, "status": "completed", "output_url": output_url, "profile": req.profile_name}


async def _run_voice_audition_task(task_id: str, req: VoiceAuditionRequest) -> dict[str, Any]:
    if req.project_id:
        _load_project(req.project_id)

    tts_backend = _require_backend("tts")
    pack = _load_pack(req.character_pack_id) if req.character_pack_id else None
    response_format = req.response_format
    pack_voice_options = pack.get("default_voice_options", {}) if pack else {}
    voices = list(req.voices)
    if not voices and pack and pack.get("voice"):
        voices = [pack["voice"]]
    if not voices:
        raise RuntimeError("Voice audition requires at least one voice or a character pack with a default voice.")

    samples = []
    async with httpx.AsyncClient(timeout=None) as client:
        for index, voice in enumerate(voices, start=1):
            payload = {
                **pack_voice_options,
                "input": req.text,
                "voice": voice,
                "response_format": response_format,
            }
            if req.language:
                payload["language"] = req.language
            if req.instruct:
                payload["instruct"] = req.instruct

            response = await client.post(f"{tts_backend}/v1/audio/speech", json=payload)
            response.raise_for_status()

            safe_voice = _safe_slug(voice, f"voice-{index}")
            output_path = _output_path(task_id, f"{index:02d}-{safe_voice}.{response_format}")
            with open(output_path, "wb") as handle:
                handle.write(response.content)
            output_url = _relative_output_url(output_path)
            samples.append({"voice": voice, "url": output_url})
            _attach_asset(req.project_id, "audio", f"{req.label}-{voice}", output_url, {"voice": voice})

    return {"task_id": task_id, "status": "completed", "samples": samples}


async def _run_prompt_compare_task(task_id: str, req: PromptCompareRequest) -> dict[str, Any]:
    if req.project_id:
        _load_project(req.project_id)

    if not req.prompt_variants:
        raise RuntimeError("Prompt compare requires at least one prompt variant.")
    if len(req.prompt_variants) > 8:
        raise RuntimeError("Prompt compare supports up to 8 prompt variants per job.")

    vision_backend = _require_backend("vision")
    pack = _load_pack(req.character_pack_id) if req.character_pack_id else None
    image_defaults = dict(pack.get("default_image_options", {})) if pack else {}

    results = []
    async with httpx.AsyncClient(timeout=None) as client:
        for index, prompt in enumerate(req.prompt_variants, start=1):
            final_prompt = _apply_character_pack(prompt, pack)
            payload = {**image_defaults, **req.shared_options, "prompt": final_prompt}
            if pack and pack.get("negative_prompt") and "negative_prompt" not in payload:
                payload["negative_prompt"] = pack["negative_prompt"]

            init_response = await client.post(f"{vision_backend}/v1/images/async_generate", json=payload)
            init_response.raise_for_status()
            init_data = init_response.json()
            vision_task_id = init_data.get("task_id")
            if not vision_task_id:
                raise RuntimeError(f"Vision backend did not return task_id: {init_data}")

            status_url = f"{vision_backend}/v1/images/tasks/{vision_task_id}"
            status_data = await _poll_task_url(client, status_url, vision_backend)
            data_items = status_data.get("data") or []
            if not data_items:
                raise RuntimeError(f"Vision task completed without image data: {status_data}")

            source_url = _resolve_remote_url(data_items[0].get("url", ""), vision_backend)
            if not source_url:
                raise RuntimeError(f"Vision task completed without image URL: {status_data}")
            output_path = _output_path(task_id, f"variant-{index:02d}.png")
            await _download_to_path(source_url, output_path)
            output_url = _relative_output_url(output_path)
            results.append({"index": index, "prompt": final_prompt, "url": output_url})
            _attach_asset(req.project_id, "image", f"{req.label}-{index}", output_url, {"prompt": final_prompt})

    return {"task_id": task_id, "status": "completed", "variants": results}


@studio_router.post("/captions")
async def create_captions(req: CaptionTaskRequest):
    task = _create_task("captions", req.project_id)
    start_background_task(task["task_id"], lambda task_id: _run_caption_task(task_id, req))
    return {"task_id": task["task_id"], "status": "processing"}


@studio_router.post("/thumbnails")
async def create_thumbnail_task(req: ThumbnailTaskRequest):
    task = _create_task("thumbnails", req.project_id)
    start_background_task(task["task_id"], lambda task_id: _run_thumbnail_task(task_id, req))
    return {"task_id": task["task_id"], "status": "processing"}


@studio_router.post("/contact-sheets")
async def create_contact_sheet_task(req: ContactSheetTaskRequest):
    task = _create_task("contact_sheets", req.project_id)
    start_background_task(task["task_id"], lambda task_id: _run_contact_sheet_task(task_id, req))
    return {"task_id": task["task_id"], "status": "processing"}


@studio_router.post("/transcodes")
async def create_transcode_task(req: TranscodeTaskRequest):
    resolve_format_profile(req.profile_name)
    task = _create_task("transcodes", req.project_id)
    start_background_task(task["task_id"], lambda task_id: _run_transcode_task(task_id, req))
    return {"task_id": task["task_id"], "status": "processing"}


@studio_router.post("/voice-auditions")
async def create_voice_audition_task(req: VoiceAuditionRequest):
    task = _create_task("voice_auditions", req.project_id)
    start_background_task(task["task_id"], lambda task_id: _run_voice_audition_task(task_id, req))
    return {"task_id": task["task_id"], "status": "processing"}


@studio_router.post("/prompt-compare")
async def create_prompt_compare_task(req: PromptCompareRequest):
    task = _create_task("prompt_compare", req.project_id)
    start_background_task(task["task_id"], lambda task_id: _run_prompt_compare_task(task_id, req))
    return {"task_id": task["task_id"], "status": "processing"}
