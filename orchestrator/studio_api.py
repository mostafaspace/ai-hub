import ast
import asyncio
import json
import os
import re
import shutil
import time
import uuid
import zipfile
from collections import Counter, defaultdict
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import httpx
import yaml
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, ValidationError

import config
from orchestrator.studio_media import (
    FORMAT_PROFILES,
    attach_audio_bed,
    build_timeline_plan,
    burn_subtitles_to_video,
    concat_video_clips,
    create_contact_sheet,
    detect_duration,
    extract_audio_for_transcription,
    extract_thumbnail,
    ffmpeg_available,
    normalize_video_clip,
    resolve_format_profile,
    transcode_media,
    write_approximate_srt,
)


studio_router = APIRouter(prefix="/v1/studio", tags=["Practical Studio"])

ROOT_DIR = os.path.dirname(__file__)
STATE_DIR = os.path.join(ROOT_DIR, "studio_state")
TASKS_DIR = os.path.join(STATE_DIR, "tasks")
PROJECTS_DIR = os.path.join(ROOT_DIR, "studio_projects")
PACKS_DIR = os.path.join(ROOT_DIR, "studio_character_packs")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "director_outputs", "practical")
UPLOADS_DIR = os.path.join(OUTPUTS_DIR, "uploads")
EXPORTS_DIR = os.path.join(OUTPUTS_DIR, "exports")
IMPORTS_DIR = os.path.join(OUTPUTS_DIR, "imports")
REGISTRY_PATH = os.path.join(ROOT_DIR, "models.yaml")
MAX_TASK_EVENTS = 60
MAX_LIST_TASKS = 500

STUDIO_TASKS: dict[str, dict[str, Any]] = {}
RUNNING_TASKS: dict[str, asyncio.Task] = {}
ASYNC_LOCKS: dict[str, asyncio.Lock] = {}


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
    photo_url: Optional[str] = None
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


class StudioTaskBase(BaseModel):
    project_id: Optional[str] = None
    webhook_url: Optional[str] = None
    label: str = "studio-task"


class CaptionTaskRequest(StudioTaskBase):
    media_url: str
    burn_in: bool = False
    output_profile: str = "youtube_short"
    prompt: str = "Transcribe this media accurately."
    label: str = "captions"


class ThumbnailTaskRequest(StudioTaskBase):
    media_url: str
    timestamp_sec: float = 0.0
    label: str = "thumbnail"


class ContactSheetTaskRequest(StudioTaskBase):
    media_url: str
    columns: int = 4
    rows: int = 2
    thumb_width: int = 320
    label: str = "contact-sheet"


class TranscodeTaskRequest(StudioTaskBase):
    media_url: str
    profile_name: str
    label: str = "transcode"


class VoiceAuditionRequest(StudioTaskBase):
    text: str
    voices: list[str] = Field(default_factory=list)
    character_pack_id: Optional[str] = None
    response_format: str = "mp3"
    language: Optional[str] = None
    instruct: Optional[str] = None
    label: str = "voice-audition"


class PromptCompareRequest(StudioTaskBase):
    prompt_variants: list[str] = Field(default_factory=list)
    shared_options: dict[str, Any] = Field(default_factory=dict)
    character_pack_id: Optional[str] = None
    label: str = "prompt-compare"


class TimelineRenderRequest(StudioTaskBase):
    format_profile: Optional[str] = None
    include_audio_bed: bool = True
    burn_subtitles: bool = True
    label: str = "timeline-render"


class DirectorProjectRequest(StudioTaskBase):
    image_prompt: str
    voiceover_text: str
    voice: str = "Vivian"
    label: str = "director-output"


class ProjectExportRequest(BaseModel):
    webhook_url: Optional[str] = None
    include_local_outputs: bool = True
    label: str = "project-export"


def _ensure_dirs() -> None:
    for path in (STATE_DIR, TASKS_DIR, PROJECTS_DIR, PACKS_DIR, OUTPUTS_DIR, UPLOADS_DIR, EXPORTS_DIR, IMPORTS_DIR):
        os.makedirs(path, exist_ok=True)


_ensure_dirs()


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _now() -> float:
    return time.time()


def _safe_slug(value: str, fallback: str = "item") -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", (value or "").strip()).strip("-").lower()
    return slug or fallback


def _sanitize_filename(filename: str, fallback: str = "asset.bin") -> str:
    name = os.path.basename((filename or "").strip())
    name = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-.")
    return name or fallback


def _public_host() -> str:
    host = getattr(config, "PUBLIC_HOST", None) or getattr(config, "HOST", "127.0.0.1")
    return "127.0.0.1" if host == "0.0.0.0" else host


def _public_base_url() -> str:
    return f"http://{_public_host()}:{getattr(config, 'ORCHESTRATOR_PORT', 9000)}"


def _loopback_base_url() -> str:
    return f"http://127.0.0.1:{getattr(config, 'ORCHESTRATOR_PORT', 9000)}"


def _project_path(project_id: str) -> str:
    return os.path.join(PROJECTS_DIR, f"{project_id}.json")


def _pack_path(pack_id: str) -> str:
    return os.path.join(PACKS_DIR, f"{pack_id}.json")


def _task_path(task_id: str) -> str:
    return os.path.join(TASKS_DIR, f"{task_id}.json")


def _output_path(task_id: str, filename: str) -> str:
    task_dir = os.path.join(OUTPUTS_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    return os.path.join(task_dir, _sanitize_filename(filename))


def _atomic_write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    os.replace(tmp_path, path)


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=True))


def _list_documents(directory: str) -> list[dict[str, Any]]:
    _ensure_dirs()
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
    return f"{_public_base_url()}/outputs/{rel_path}"


def _output_local_path_from_url(url: str) -> Optional[str]:
    parsed = urlparse(url)
    path = parsed.path or url
    if path.startswith("/outputs/"):
        rel_path = path[len("/outputs/"):].strip("/")
        local_path = os.path.join(ROOT_DIR, "director_outputs", *rel_path.split("/"))
        if os.path.exists(local_path):
            return local_path
    return None


def _resolve_remote_url(url: str, backend_url: Optional[str] = None) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if backend_url:
        return urljoin(f"{backend_url}/", url.lstrip("/"))
    return urljoin(f"{_public_base_url()}/", url.lstrip("/"))


def _guess_extension(url: str, default_ext: str) -> str:
    path = urlparse(url).path or url
    ext = os.path.splitext(path)[1].lower()
    return ext or default_ext


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


def _pythonize_json_literals(raw: str) -> str:
    fixed = re.sub(r"\btrue\b", "True", raw, flags=re.IGNORECASE)
    fixed = re.sub(r"\bfalse\b", "False", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"\bnull\b", "None", fixed, flags=re.IGNORECASE)
    return fixed


async def _parse_payload(request: Request) -> dict[str, Any]:
    raw = (await request.body()).decode("utf-8", errors="replace").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        try:
            payload = ast.literal_eval(raw)
        except Exception:
            try:
                payload = ast.literal_eval(_pythonize_json_literals(raw))
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Expected a JSON object payload.")
    return payload


async def _parse_model(request: Request, model_cls):
    payload = await _parse_payload(request)
    try:
        return model_cls(**payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc

def _task_duration(task: dict[str, Any]) -> Optional[float]:
    started = task.get("started_at")
    finished = task.get("finished_at")
    if started is None or finished is None:
        return None
    return max(float(finished) - float(started), 0.0)


def _project_assets(project_id: str) -> list[dict[str, Any]]:
    project = _load_project(project_id)
    return list(project.get("assets", []))


def _project_tasks(project_id: str) -> list[dict[str, Any]]:
    items = [task for task in STUDIO_TASKS.values() if task.get("project_id") == project_id]
    items.sort(key=lambda item: float(item.get("updated_at", item.get("created_at", 0.0))), reverse=True)
    return items


def _group_assets(assets: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for asset in assets:
        grouped[str(asset.get("kind", "other"))].append(asset)
    return dict(grouped)


def _apply_character_pack(prompt: str, pack: Optional[dict[str, Any]]) -> str:
    if not pack:
        return prompt.strip()
    parts = [pack.get("prompt_prefix", "").strip(), prompt.strip(), pack.get("prompt_suffix", "").strip()]
    return " ".join(part for part in parts if part)


def _get_async_lock(name: str) -> asyncio.Lock:
    lock = ASYNC_LOCKS.get(name)
    if lock is None:
        lock = asyncio.Lock()
        ASYNC_LOCKS[name] = lock
    return lock


def _persist_task(task: dict[str, Any]) -> None:
    task["updated_at"] = _now()
    _write_json(_task_path(task["task_id"]), task)


def _record_task_event(task_id: str, message: str, progress: Optional[str] = None) -> None:
    task = STUDIO_TASKS[task_id]
    events = task.setdefault("events", [])
    events.append({"ts": _now(), "message": message})
    if len(events) > MAX_TASK_EVENTS:
        del events[:-MAX_TASK_EVENTS]
    if progress is not None:
        task["progress"] = progress
    _persist_task(task)


def _create_task(kind: str, runner_key: str, spec: dict[str, Any], project_id: Optional[str], webhook_url: Optional[str], label: str) -> dict[str, Any]:
    task_id = str(uuid.uuid4())
    task = {
        "task_id": task_id,
        "kind": kind,
        "runner_key": runner_key,
        "label": label,
        "status": "processing",
        "project_id": project_id,
        "webhook_url": webhook_url,
        "created_at": _now(),
        "updated_at": _now(),
        "started_at": None,
        "finished_at": None,
        "attempt": 1,
        "progress": "queued",
        "cancel_requested": False,
        "result": None,
        "error": None,
        "events": [],
        "spec": spec,
    }
    STUDIO_TASKS[task_id] = task
    _record_task_event(task_id, f"Task created for {kind}.", "queued")
    return task


def _load_persisted_tasks() -> None:
    _ensure_dirs()
    STUDIO_TASKS.clear()
    for name in os.listdir(TASKS_DIR):
        if not name.endswith(".json"):
            continue
        path = os.path.join(TASKS_DIR, name)
        try:
            task = _read_json(path)
        except Exception:
            continue
        task.setdefault("events", [])
        task.setdefault("attempt", 1)
        task.setdefault("progress", task.get("status", "unknown"))
        task.setdefault("cancel_requested", False)
        if task.get("status") == "processing":
            task["status"] = "interrupted"
            task["error"] = "Task was interrupted by an orchestrator restart."
            task["finished_at"] = task.get("finished_at") or _now()
            _write_json(path, task)
        STUDIO_TASKS[task["task_id"]] = task


_load_persisted_tasks()


def _require_task(task_id: str) -> dict[str, Any]:
    task = STUDIO_TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
    return task


def _attach_asset(project_id: Optional[str], kind: str, label: str, url: str, metadata: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    if not project_id:
        return None
    project = _load_project(project_id)
    asset = {
        "asset_id": str(uuid.uuid4()),
        "kind": kind,
        "label": label,
        "url": url,
        "metadata": metadata or {},
        "created_at": _now(),
    }
    project.setdefault("assets", []).append(asset)
    _save_project(project)
    return asset


def _attach_pack_photo(pack_id: str, photo_url: str, label: str = "character-photo", metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    pack = _load_pack(pack_id)
    pack["photo_url"] = photo_url
    refs = list(pack.get("reference_asset_urls", []))
    if photo_url not in refs:
        refs.append(photo_url)
    pack["reference_asset_urls"] = refs
    pack.setdefault("photo_metadata", {}).update(metadata or {})
    pack.setdefault("photo_label", label)
    return _save_pack(pack)


async def _download_to_path(url: str, output_path: str, timeout: float = 300.0) -> str:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
    with open(output_path, "wb") as handle:
        handle.write(response.content)
    return output_path


async def _materialize_source(task_id: str, source: str, filename: str) -> str:
    output_path = _output_path(task_id, filename)
    local_path = _output_local_path_from_url(source)
    if local_path and os.path.exists(local_path):
        shutil.copy2(local_path, output_path)
        return output_path
    if os.path.isabs(source) and os.path.exists(source):
        shutil.copy2(source, output_path)
        return output_path
    await _download_to_path(_resolve_remote_url(source), output_path)
    return output_path


async def _store_upload(file: UploadFile, subdir: str, suggested_name: Optional[str] = None) -> dict[str, Any]:
    _ensure_dirs()
    upload_id = str(uuid.uuid4())
    folder = os.path.join(UPLOADS_DIR, subdir, upload_id)
    os.makedirs(folder, exist_ok=True)
    filename = _sanitize_filename(suggested_name or file.filename or "upload.bin")
    local_path = os.path.join(folder, filename)
    data = await file.read()
    with open(local_path, "wb") as handle:
        handle.write(data)
    return {
        "upload_id": upload_id,
        "filename": filename,
        "size": len(data),
        "content_type": file.content_type,
        "local_path": local_path,
        "url": _relative_output_url(local_path),
    }


def _raise_if_cancel_requested(task_id: str) -> None:
    task = STUDIO_TASKS[task_id]
    if task.get("cancel_requested"):
        raise asyncio.CancelledError()


async def _notify_webhook(task: dict[str, Any]) -> None:
    webhook_url = task.get("webhook_url")
    if not webhook_url:
        return
    payload = {
        "task_id": task["task_id"],
        "kind": task.get("kind"),
        "status": task.get("status"),
        "project_id": task.get("project_id"),
        "label": task.get("label"),
        "result": task.get("result"),
        "error": task.get("error"),
        "updated_at": task.get("updated_at"),
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(webhook_url, json=payload)
        task["webhook_delivery"] = {"status_code": response.status_code, "delivered_at": _now()}
    except Exception as exc:
        task["webhook_delivery"] = {"error": str(exc), "delivered_at": _now()}
    _persist_task(task)


async def _poll_task_url(client: httpx.AsyncClient, status_url: str) -> dict[str, Any]:
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

async def _run_caption_task(task_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    req = CaptionTaskRequest(**spec)
    if req.project_id:
        _load_project(req.project_id)
    if req.burn_in and not ffmpeg_available():
        raise RuntimeError("FFmpeg is required for subtitle burn-in.")

    source_ext = _guess_extension(req.media_url, ".bin")
    source_path = await _materialize_source(task_id, req.media_url, f"source{source_ext}")
    _record_task_event(task_id, "Source media prepared.", "preparing-media")
    _raise_if_cancel_requested(task_id)

    video_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
    working_audio_path = source_path
    duration_target = source_path
    is_video = source_ext in video_exts
    if is_video:
        working_audio_path = _output_path(task_id, "transcription.wav")
        async with _get_async_lock("studio-media"):
            ok, message = await asyncio.to_thread(extract_audio_for_transcription, source_path, working_audio_path)
        if not ok:
            raise RuntimeError(f"Failed to extract audio for captions: {message}")
        duration_target = source_path
        _record_task_event(task_id, "Audio extracted for transcription.", "extracting-audio")

    duration_seconds = await asyncio.to_thread(detect_duration, duration_target)
    _raise_if_cancel_requested(task_id)

    asr_backend = _require_backend("asr")
    async with _get_async_lock("asr"):
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
    subtitle_url = _relative_output_url(srt_path)
    _attach_asset(req.project_id, "subtitle", req.label, subtitle_url, {"transcript": transcript, "approximate_timing": True})
    result = {
        "task_id": task_id,
        "status": "completed",
        "transcript": transcript,
        "subtitle_url": subtitle_url,
        "approximate_timing": True,
    }
    _record_task_event(task_id, "Subtitle file generated.", "captions-ready")
    _raise_if_cancel_requested(task_id)

    if req.burn_in:
        if not is_video:
            raise RuntimeError("Subtitle burn-in currently requires a video input.")
        profile = resolve_format_profile(req.output_profile)
        if profile["kind"] != "video":
            raise RuntimeError("Subtitle burn-in requires a video output profile.")
        burned_path = _output_path(task_id, f"captioned.{profile['container']}")
        async with _get_async_lock("studio-media"):
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
        _record_task_event(task_id, "Captioned video rendered.", "burn-in-complete")
    return result


async def _run_thumbnail_task(task_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    req = ThumbnailTaskRequest(**spec)
    if req.project_id:
        _load_project(req.project_id)
    if not ffmpeg_available():
        raise RuntimeError("FFmpeg is required for thumbnail generation.")

    source_ext = _guess_extension(req.media_url, ".mp4")
    source_path = await _materialize_source(task_id, req.media_url, f"source{source_ext}")
    output_path = _output_path(task_id, "thumbnail.jpg")
    async with _get_async_lock("studio-media"):
        ok, message = await asyncio.to_thread(extract_thumbnail, source_path, output_path, req.timestamp_sec)
    if not ok:
        raise RuntimeError(f"Failed to create thumbnail: {message}")
    output_url = _relative_output_url(output_path)
    _attach_asset(req.project_id, "image", req.label, output_url, {"timestamp_sec": req.timestamp_sec})
    return {"task_id": task_id, "status": "completed", "thumbnail_url": output_url}


async def _run_contact_sheet_task(task_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    req = ContactSheetTaskRequest(**spec)
    if req.project_id:
        _load_project(req.project_id)
    if not ffmpeg_available():
        raise RuntimeError("FFmpeg is required for contact sheet generation.")

    source_ext = _guess_extension(req.media_url, ".mp4")
    source_path = await _materialize_source(task_id, req.media_url, f"source{source_ext}")
    output_path = _output_path(task_id, "contact-sheet.jpg")
    async with _get_async_lock("studio-media"):
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
    _attach_asset(req.project_id, "image", req.label, output_url, {"columns": req.columns, "rows": req.rows, "thumb_width": req.thumb_width})
    return {"task_id": task_id, "status": "completed", "contact_sheet_url": output_url}


async def _run_transcode_task(task_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    req = TranscodeTaskRequest(**spec)
    if req.project_id:
        _load_project(req.project_id)
    if not ffmpeg_available():
        raise RuntimeError("FFmpeg is required for transcoding.")

    profile = resolve_format_profile(req.profile_name)
    source_ext = _guess_extension(req.media_url, ".bin")
    source_path = await _materialize_source(task_id, req.media_url, f"source{source_ext}")
    output_path = _output_path(task_id, f"transcoded.{profile['container']}")
    async with _get_async_lock("studio-media"):
        ok, message = await asyncio.to_thread(transcode_media, source_path, output_path, req.profile_name)
    if not ok:
        raise RuntimeError(f"Failed to transcode media: {message}")
    output_url = _relative_output_url(output_path)
    _attach_asset(req.project_id, profile["kind"], req.label, output_url, {"profile_name": req.profile_name})
    return {"task_id": task_id, "status": "completed", "output_url": output_url, "profile": req.profile_name}


async def _run_voice_audition_task(task_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    req = VoiceAuditionRequest(**spec)
    if req.project_id:
        _load_project(req.project_id)
    pack = _load_pack(req.character_pack_id) if req.character_pack_id else None
    tts_backend = _require_backend("tts")
    pack_voice_options = dict(pack.get("default_voice_options", {})) if pack else {}
    voices = list(req.voices)
    if not voices and pack and pack.get("voice"):
        voices = [pack["voice"]]
    if not voices:
        raise RuntimeError("Voice audition requires at least one voice or a character pack with a default voice.")

    samples = []
    async with _get_async_lock("tts"):
        async with httpx.AsyncClient(timeout=None) as client:
            for index, voice in enumerate(voices, start=1):
                _raise_if_cancel_requested(task_id)
                payload = {**pack_voice_options, "input": req.text, "voice": voice, "response_format": req.response_format}
                if req.language:
                    payload["language"] = req.language
                if req.instruct:
                    payload["instruct"] = req.instruct
                response = await client.post(f"{tts_backend}/v1/audio/speech", json=payload)
                response.raise_for_status()
                safe_voice = _safe_slug(voice, f"voice-{index}")
                output_path = _output_path(task_id, f"{index:02d}-{safe_voice}.{req.response_format}")
                with open(output_path, "wb") as handle:
                    handle.write(response.content)
                output_url = _relative_output_url(output_path)
                samples.append({"voice": voice, "url": output_url})
                _attach_asset(req.project_id, "audio", f"{req.label}-{voice}", output_url, {"voice": voice})
                _record_task_event(task_id, f"Voice sample ready for {voice}.", f"sample-{index}")
    return {"task_id": task_id, "status": "completed", "samples": samples}


async def _run_prompt_compare_task(task_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    req = PromptCompareRequest(**spec)
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

    async with _get_async_lock("vision"):
        async with httpx.AsyncClient(timeout=None) as client:
            for index, prompt in enumerate(req.prompt_variants, start=1):
                _raise_if_cancel_requested(task_id)
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
                status_data = await _poll_task_url(client, f"{vision_backend}/v1/images/tasks/{vision_task_id}")
                data_items = status_data.get("data") or []
                if not data_items:
                    raise RuntimeError(f"Vision task completed without image data: {status_data}")
                source_url = _resolve_remote_url(data_items[0].get("url", ""), vision_backend)
                output_path = _output_path(task_id, f"variant-{index:02d}.png")
                await _download_to_path(source_url, output_path)
                output_url = _relative_output_url(output_path)
                results.append({"index": index, "prompt": final_prompt, "url": output_url})
                _attach_asset(req.project_id, "image", f"{req.label}-{index}", output_url, {"prompt": final_prompt})
                _record_task_event(task_id, f"Prompt variant {index} rendered.", f"variant-{index}")
    return {"task_id": task_id, "status": "completed", "variants": results}

async def _run_timeline_render_task(task_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    req = TimelineRenderRequest(**spec)
    if not req.project_id:
        raise RuntimeError("Timeline rendering requires a project_id.")
    if not ffmpeg_available():
        raise RuntimeError("FFmpeg is required for timeline rendering.")

    project = _load_project(req.project_id)
    profile_name = req.format_profile or project.get("default_profile", "youtube_short")
    profile = resolve_format_profile(profile_name)
    if profile["kind"] != "video":
        raise RuntimeError("Timeline rendering requires a video output profile.")

    timeline = project.get("timeline", {})
    clips = sorted(timeline.get("video_clips", []), key=lambda item: float(item.get("start_sec", 0.0)))
    audio_tracks = sorted(timeline.get("audio_tracks", []), key=lambda item: float(item.get("start_sec", 0.0)))
    subtitle_tracks = list(timeline.get("subtitle_tracks", []))
    if not clips:
        raise RuntimeError("Timeline rendering requires at least one video clip.")

    plan = build_timeline_plan(clips, audio_tracks, subtitle_tracks, profile_name)
    normalized_clips = []
    async with _get_async_lock("studio-media"):
        for index, clip in enumerate(clips, start=1):
            _raise_if_cancel_requested(task_id)
            source_ext = _guess_extension(clip.get("asset_url", ""), ".mp4")
            source_path = await _materialize_source(task_id, clip["asset_url"], f"clip-{index:02d}{source_ext}")
            normalized_path = _output_path(task_id, f"normalized-{index:02d}.mp4")
            ok, message = await asyncio.to_thread(
                normalize_video_clip,
                source_path,
                normalized_path,
                profile_name,
                float(clip.get("trim_in_sec", 0.0) or 0.0),
                float(clip.get("duration_sec") or 0.0) or None,
            )
            if not ok:
                raise RuntimeError(f"Failed to normalize clip {index}: {message}")
            normalized_clips.append(normalized_path)
            _record_task_event(task_id, f"Normalized clip {index}.", f"clip-{index}")

        assembled_path = _output_path(task_id, "timeline-assembled.mp4")
        ok, message = await asyncio.to_thread(concat_video_clips, normalized_clips, assembled_path)
        if not ok:
            raise RuntimeError(f"Failed to concatenate clips: {message}")
        current_video = assembled_path

        if req.include_audio_bed and audio_tracks:
            first_audio = audio_tracks[0]
            audio_ext = _guess_extension(first_audio.get("asset_url", ""), ".mp3")
            audio_path = await _materialize_source(task_id, first_audio["asset_url"], f"audio-bed{audio_ext}")
            mixed_path = _output_path(task_id, "timeline-with-audio.mp4")
            ok, message = await asyncio.to_thread(
                attach_audio_bed,
                current_video,
                audio_path,
                mixed_path,
                float(first_audio.get("start_sec", 0.0) or 0.0),
                float(first_audio.get("volume", 1.0) or 1.0),
            )
            if not ok:
                raise RuntimeError(f"Failed to attach audio bed: {message}")
            current_video = mixed_path
            _record_task_event(task_id, "Audio bed attached.", "audio-bed")

        if req.burn_subtitles and subtitle_tracks:
            subtitle_path = await _materialize_source(task_id, subtitle_tracks[0]["subtitle_url"], "timeline-subtitles.srt")
            burned_path = _output_path(task_id, f"timeline-render.{profile['container']}")
            ok, message = await asyncio.to_thread(
                burn_subtitles_to_video,
                current_video,
                subtitle_path,
                burned_path,
                profile_name,
            )
            if not ok:
                raise RuntimeError(f"Failed to burn subtitles into timeline render: {message}")
            current_video = burned_path
            _record_task_event(task_id, "Subtitles burned into render.", "subtitle-burn")
        elif current_video != assembled_path:
            pass
        else:
            final_path = _output_path(task_id, f"timeline-render.{profile['container']}")
            shutil.copy2(current_video, final_path)
            current_video = final_path

    output_url = _relative_output_url(current_video)
    _attach_asset(req.project_id, "video", req.label, output_url, {"format_profile": profile_name, "plan": plan})
    project = _load_project(req.project_id)
    project.setdefault("timeline", {})["last_render_plan"] = plan
    _save_project(project)
    return {"task_id": task_id, "status": "completed", "output_url": output_url, "plan": plan}


async def _run_director_attach_task(task_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    req = DirectorProjectRequest(**spec)
    if not req.project_id:
        raise RuntimeError("Director project runs require a project_id.")
    _load_project(req.project_id)
    payload = {"image_prompt": req.image_prompt, "voiceover_text": req.voiceover_text, "voice": req.voice}
    async with _get_async_lock("director"):
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(f"{_loopback_base_url()}/v1/workflows/director", json=payload)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        data = response.json()
    output_url = data.get("output_url")
    if not output_url:
        raise RuntimeError(f"Director did not return output_url: {data}")
    _attach_asset(req.project_id, "video", req.label, output_url, {"workflow": "director", "voice": req.voice, "image_prompt": req.image_prompt})
    return {"task_id": task_id, "status": "completed", "director_task_id": data.get("task_id"), "output_url": output_url}


async def _run_project_export_task(task_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    project_id = spec["project_id"]
    include_local_outputs = bool(spec.get("include_local_outputs", True))
    project = _load_project(project_id)
    export_path = _output_path(task_id, f"{_safe_slug(project.get('name', project_id), 'project')}-export.zip")
    related_tasks = _project_tasks(project_id)
    manifest = {
        "version": 1,
        "exported_at": _now(),
        "project": project,
        "tasks": related_tasks,
        "bundled_assets": [],
    }
    with zipfile.ZipFile(export_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("project.json", json.dumps(project, indent=2, ensure_ascii=True))
        archive.writestr("tasks.json", json.dumps(related_tasks, indent=2, ensure_ascii=True))
        if include_local_outputs:
            for asset in project.get("assets", []):
                local_path = _output_local_path_from_url(asset.get("url", ""))
                if not local_path or not os.path.exists(local_path):
                    continue
                bundle_name = f"assets/{asset.get('asset_id', uuid.uuid4().hex)}-{_sanitize_filename(os.path.basename(local_path))}"
                archive.write(local_path, bundle_name)
                manifest["bundled_assets"].append({"asset_id": asset.get("asset_id"), "bundle_path": bundle_name})
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=True))
    return {"task_id": task_id, "status": "completed", "export_url": _relative_output_url(export_path)}


TASK_RUNNERS = {
    "captions": _run_caption_task,
    "thumbnails": _run_thumbnail_task,
    "contact_sheets": _run_contact_sheet_task,
    "transcodes": _run_transcode_task,
    "voice_auditions": _run_voice_audition_task,
    "prompt_compare": _run_prompt_compare_task,
    "timeline_render": _run_timeline_render_task,
    "director_attach": _run_director_attach_task,
    "project_export": _run_project_export_task,
}


async def _execute_task(task_id: str) -> None:
    task = _require_task(task_id)
    runner = TASK_RUNNERS.get(task.get("runner_key"))
    if runner is None:
        task["status"] = "failed"
        task["error"] = f"Unknown runner: {task.get('runner_key')}"
        task["finished_at"] = _now()
        _persist_task(task)
        return

    task["status"] = "processing"
    task["started_at"] = task.get("started_at") or _now()
    task["error"] = None
    task["cancel_requested"] = False
    _persist_task(task)
    _record_task_event(task_id, "Task execution started.", "processing")

    try:
        result = await runner(task_id, task.get("spec") or {})
        _raise_if_cancel_requested(task_id)
        task["status"] = "completed"
        task["result"] = result
        task["finished_at"] = _now()
        _record_task_event(task_id, "Task completed successfully.", "completed")
    except asyncio.CancelledError:
        task["status"] = "cancelled"
        task["error"] = "Task was cancelled."
        task["finished_at"] = _now()
        _record_task_event(task_id, "Task cancelled.", "cancelled")
    except Exception as exc:
        task["status"] = "failed"
        task["error"] = str(exc)
        task["finished_at"] = _now()
        _record_task_event(task_id, f"Task failed: {exc}", "failed")
    finally:
        _persist_task(task)
        RUNNING_TASKS.pop(task_id, None)
        asyncio.create_task(_notify_webhook(dict(task)))


def start_background_task(task_id: str) -> None:
    RUNNING_TASKS[task_id] = asyncio.create_task(_execute_task(task_id))


def _queue_task(kind: str, runner_key: str, spec: dict[str, Any], project_id: Optional[str], webhook_url: Optional[str], label: str) -> dict[str, Any]:
    task = _create_task(kind, runner_key, spec, project_id, webhook_url, label)
    start_background_task(task["task_id"])
    return task

def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


@studio_router.get("/format-profiles")
def list_format_profiles():
    return {"profiles": FORMAT_PROFILES}


@studio_router.get("/tasks")
def list_tasks(project_id: Optional[str] = None, status: Optional[str] = None, limit: int = 100):
    tasks = list(STUDIO_TASKS.values())
    if project_id:
        tasks = [item for item in tasks if item.get("project_id") == project_id]
    if status:
        tasks = [item for item in tasks if str(item.get("status", "")).lower() == status.lower()]
    tasks.sort(key=lambda item: float(item.get("updated_at", item.get("created_at", 0.0))), reverse=True)
    safe_limit = max(1, min(limit, MAX_LIST_TASKS))
    return {"tasks": tasks[:safe_limit]}


@studio_router.get("/tasks/{task_id}")
def get_task(task_id: str):
    return _require_task(task_id)


@studio_router.get("/tasks/{task_id}/events")
def get_task_events(task_id: str):
    task = _require_task(task_id)
    return {"task_id": task_id, "events": task.get("events", [])}


@studio_router.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    task = _require_task(task_id)
    if task.get("status") not in {"processing"}:
        return task
    task["cancel_requested"] = True
    _record_task_event(task_id, "Cancel requested by client.", "cancelling")
    running = RUNNING_TASKS.get(task_id)
    if running is not None:
        running.cancel()
    else:
        task["status"] = "cancelled"
        task["finished_at"] = _now()
        _persist_task(task)
    return task


@studio_router.post("/tasks/{task_id}/resume")
def resume_task(task_id: str):
    task = _require_task(task_id)
    if task.get("status") not in {"failed", "cancelled", "interrupted"}:
        raise HTTPException(status_code=409, detail="Only failed, cancelled, or interrupted tasks can be resumed.")
    if task_id in RUNNING_TASKS:
        raise HTTPException(status_code=409, detail="Task is already running.")
    task["status"] = "processing"
    task["error"] = None
    task["result"] = None
    task["finished_at"] = None
    task["cancel_requested"] = False
    task["attempt"] = int(task.get("attempt") or 1) + 1
    _record_task_event(task_id, "Task resume requested.", "queued")
    start_background_task(task_id)
    return task


@studio_router.post("/tasks/{task_id}/retry")
def retry_task(task_id: str):
    return resume_task(task_id)


@studio_router.get("/observability")
def studio_observability():
    tasks = list(STUDIO_TASKS.values())
    tasks_by_status = Counter(task.get("status", "unknown") for task in tasks)
    tasks_by_kind = Counter(task.get("kind", "unknown") for task in tasks)
    avg_duration_by_kind: dict[str, float] = {}
    duration_buckets: dict[str, list[float]] = defaultdict(list)
    for task in tasks:
        duration = _task_duration(task)
        if duration is not None:
            duration_buckets[str(task.get("kind", "unknown"))].append(round(duration, 3))
    for kind, durations in duration_buckets.items():
        avg_duration_by_kind[kind] = round(sum(durations) / len(durations), 3)
    failures = [
        {
            "task_id": task.get("task_id"),
            "kind": task.get("kind"),
            "error": task.get("error"),
            "updated_at": task.get("updated_at"),
        }
        for task in sorted(tasks, key=lambda item: float(item.get("updated_at", 0.0)), reverse=True)
        if task.get("status") == "failed"
    ][:5]
    return {
        "projects": len(_list_documents(PROJECTS_DIR)),
        "character_packs": len(_list_documents(PACKS_DIR)),
        "tasks_total": len(tasks),
        "running_tasks": len(RUNNING_TASKS),
        "tasks_by_status": dict(tasks_by_status),
        "tasks_by_kind": dict(tasks_by_kind),
        "average_duration_sec_by_kind": avg_duration_by_kind,
        "recent_failures": failures,
        "storage_bytes": {
            "projects": _dir_size_bytes(PROJECTS_DIR),
            "packs": _dir_size_bytes(PACKS_DIR),
            "tasks": _dir_size_bytes(TASKS_DIR),
            "outputs": _dir_size_bytes(OUTPUTS_DIR),
        },
    }


@studio_router.get("/projects")
def list_projects():
    return {"projects": _list_documents(PROJECTS_DIR)}


@studio_router.post("/projects")
async def create_project(request: Request):
    req = await _parse_model(request, ProjectCreateRequest)
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
    project = _load_project(project_id)
    project["asset_count"] = len(project.get("assets", []))
    project["task_count"] = len(_project_tasks(project_id))
    return project


@studio_router.get("/projects/{project_id}/assets")
def get_project_assets(project_id: str):
    assets = _project_assets(project_id)
    return {"project_id": project_id, "assets": assets, "gallery": _group_assets(assets)}


@studio_router.get("/projects/{project_id}/tasks")
def get_project_tasks(project_id: str):
    _load_project(project_id)
    return {"project_id": project_id, "tasks": _project_tasks(project_id)}


@studio_router.post("/projects/{project_id}/assets")
async def add_project_asset(project_id: str, request: Request):
    _load_project(project_id)
    req = await _parse_model(request, ProjectAssetRequest)
    _attach_asset(project_id, req.kind, req.label, req.url, req.metadata)
    return _load_project(project_id)


@studio_router.get("/projects/{project_id}/timeline")
def get_project_timeline(project_id: str):
    project = _load_project(project_id)
    return {"project_id": project_id, "timeline": project.get("timeline", {})}


@studio_router.put("/projects/{project_id}/timeline")
async def update_project_timeline(project_id: str, request: Request):
    project = _load_project(project_id)
    req = await _parse_model(request, TimelineUpdateRequest)
    timeline = _model_dump(req)
    timeline["last_render_plan"] = project.get("timeline", {}).get("last_render_plan")
    project["timeline"] = timeline
    return _save_project(project)


@studio_router.post("/projects/{project_id}/timeline/plan")
async def build_project_timeline_plan(project_id: str, request: Request):
    project = _load_project(project_id)
    req = await _parse_model(request, TimelinePlanRequest)
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


@studio_router.post("/projects/{project_id}/timeline/render")
async def create_timeline_render(project_id: str, request: Request):
    _load_project(project_id)
    req = await _parse_model(request, TimelineRenderRequest)
    payload = _model_dump(req)
    payload["project_id"] = project_id
    task = _queue_task("timeline_render", "timeline_render", payload, project_id, req.webhook_url, req.label)
    return {"task_id": task["task_id"], "status": "processing"}


@studio_router.post("/projects/{project_id}/director-runs")
async def create_director_project_run(project_id: str, request: Request):
    _load_project(project_id)
    req = await _parse_model(request, DirectorProjectRequest)
    payload = _model_dump(req)
    payload["project_id"] = project_id
    task = _queue_task("director_attach", "director_attach", payload, project_id, req.webhook_url, req.label)
    return {"task_id": task["task_id"], "status": "processing"}


@studio_router.post("/projects/{project_id}/export")
async def export_project(project_id: str, request: Request):
    _load_project(project_id)
    req = await _parse_model(request, ProjectExportRequest)
    task = _queue_task(
        "project_export",
        "project_export",
        {"project_id": project_id, "include_local_outputs": req.include_local_outputs},
        project_id,
        req.webhook_url,
        req.label,
    )
    return {"task_id": task["task_id"], "status": "processing"}

@studio_router.post("/projects/import")
async def import_project_archive(file: UploadFile = File(...)):
    stored = await _store_upload(file, "imports", file.filename or "project-import.zip")
    local_path = stored["local_path"]
    with zipfile.ZipFile(local_path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8")) if "manifest.json" in archive.namelist() else {}
        project = json.loads(archive.read("project.json").decode("utf-8"))
        bundled_map = {item["asset_id"]: item["bundle_path"] for item in manifest.get("bundled_assets", []) if item.get("asset_id") and item.get("bundle_path")}
        old_project_id = project.get("project_id") or "project"
        new_project_id = f"proj-{_safe_slug(project.get('name', old_project_id), 'project')}-{uuid.uuid4().hex[:8]}"
        project["project_id"] = new_project_id
        project["created_at"] = _now()
        project["updated_at"] = _now()
        imported_assets = 0
        for asset in project.get("assets", []):
            bundle_path = bundled_map.get(asset.get("asset_id"))
            if not bundle_path or bundle_path not in archive.namelist():
                continue
            target_folder = os.path.join(IMPORTS_DIR, new_project_id)
            os.makedirs(target_folder, exist_ok=True)
            filename = _sanitize_filename(os.path.basename(bundle_path))
            local_asset_path = os.path.join(target_folder, filename)
            with archive.open(bundle_path) as source, open(local_asset_path, "wb") as target:
                shutil.copyfileobj(source, target)
            asset["url"] = _relative_output_url(local_asset_path)
            asset.setdefault("metadata", {})["imported_from_project"] = old_project_id
            imported_assets += 1
    _write_json(_project_path(project["project_id"]), project)
    return {"project": project, "imported_assets": imported_assets}


@studio_router.get("/character-packs")
def list_character_packs():
    return {"character_packs": _list_documents(PACKS_DIR)}


@studio_router.post("/character-packs")
async def create_character_pack(request: Request):
    req = await _parse_model(request, CharacterPackCreateRequest)
    pack_id = f"pack-{_safe_slug(req.name, 'character')}-{uuid.uuid4().hex[:8]}"
    pack = {
        "pack_id": pack_id,
        "created_at": _now(),
        "updated_at": _now(),
        **_model_dump(req),
    }
    if pack.get("photo_url") and pack["photo_url"] not in pack.get("reference_asset_urls", []):
        pack.setdefault("reference_asset_urls", []).append(pack["photo_url"])
    _write_json(_pack_path(pack_id), pack)
    return pack


@studio_router.get("/character-packs/{pack_id}")
def get_character_pack(pack_id: str):
    return _load_pack(pack_id)


@studio_router.post("/character-packs/{pack_id}/photo")
async def upload_character_pack_photo(
    pack_id: str,
    file: UploadFile = File(...),
    label: str = Form("character-photo"),
):
    _load_pack(pack_id)
    stored = await _store_upload(file, "character-pack-photos", file.filename or "character-photo.jpg")
    pack = _attach_pack_photo(pack_id, stored["url"], label, {"filename": stored["filename"], "size": stored["size"]})
    return {"pack": pack, "photo_url": stored["url"]}


@studio_router.post("/uploads")
async def upload_media(
    file: UploadFile = File(...),
    project_id: Optional[str] = Form(None),
    pack_id: Optional[str] = Form(None),
    kind: str = Form("file"),
    label: str = Form("upload"),
    use_as_pack_photo: bool = Form(False),
):
    if project_id:
        _load_project(project_id)
    if pack_id:
        _load_pack(pack_id)
    stored = await _store_upload(file, "generic", file.filename or "upload.bin")
    asset = None
    if project_id:
        asset = _attach_asset(project_id, kind, label, stored["url"], {"filename": stored["filename"], "size": stored["size"]})
    if pack_id and use_as_pack_photo:
        _attach_pack_photo(pack_id, stored["url"], label, {"filename": stored["filename"], "size": stored["size"]})
    elif pack_id:
        pack = _load_pack(pack_id)
        refs = list(pack.get("reference_asset_urls", []))
        if stored["url"] not in refs:
            refs.append(stored["url"])
        pack["reference_asset_urls"] = refs
        _save_pack(pack)
    return {"upload": stored, "asset": asset}


@studio_router.post("/captions")
async def create_captions(request: Request):
    req = await _parse_model(request, CaptionTaskRequest)
    task = _queue_task("captions", "captions", _model_dump(req), req.project_id, req.webhook_url, req.label)
    return {"task_id": task["task_id"], "status": "processing"}


@studio_router.post("/thumbnails")
async def create_thumbnail_task(request: Request):
    req = await _parse_model(request, ThumbnailTaskRequest)
    task = _queue_task("thumbnails", "thumbnails", _model_dump(req), req.project_id, req.webhook_url, req.label)
    return {"task_id": task["task_id"], "status": "processing"}


@studio_router.post("/contact-sheets")
async def create_contact_sheet_task(request: Request):
    req = await _parse_model(request, ContactSheetTaskRequest)
    task = _queue_task("contact_sheets", "contact_sheets", _model_dump(req), req.project_id, req.webhook_url, req.label)
    return {"task_id": task["task_id"], "status": "processing"}


@studio_router.post("/transcodes")
async def create_transcode_task(request: Request):
    req = await _parse_model(request, TranscodeTaskRequest)
    resolve_format_profile(req.profile_name)
    task = _queue_task("transcodes", "transcodes", _model_dump(req), req.project_id, req.webhook_url, req.label)
    return {"task_id": task["task_id"], "status": "processing"}


@studio_router.post("/voice-auditions")
async def create_voice_audition_task(request: Request):
    req = await _parse_model(request, VoiceAuditionRequest)
    task = _queue_task("voice_auditions", "voice_auditions", _model_dump(req), req.project_id, req.webhook_url, req.label)
    return {"task_id": task["task_id"], "status": "processing"}


@studio_router.post("/prompt-compare")
async def create_prompt_compare_task(request: Request):
    req = await _parse_model(request, PromptCompareRequest)
    task = _queue_task("prompt_compare", "prompt_compare", _model_dump(req), req.project_id, req.webhook_url, req.label)
    return {"task_id": task["task_id"], "status": "processing"}