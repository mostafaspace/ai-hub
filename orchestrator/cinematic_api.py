import asyncio
import json
import math
import os
import shutil
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from orchestrator import studio_api as studio
from orchestrator.business_media import ffmpeg_available, render_image_hold_clip
from orchestrator.media_utils import get_media_duration, loop_video_to_duration, mix_audio_files, mux_video_and_audio
from orchestrator.studio_media import concat_video_clips, detect_duration, normalize_video_clip, resolve_format_profile


cinematic_router = APIRouter(prefix="/v1/studio", tags=["Cinematic Studio"])

DEFAULT_NEGATIVE_PROMPT = "blurry, flicker, low detail, warped anatomy, text artifacts, jitter, watermark"


class CinematicTaskBase(BaseModel):
    project_id: Optional[str] = None
    webhook_url: Optional[str] = None
    label: str = "cinematic-production"
    quality_preset: str = "premium"


class CinematicShot(BaseModel):
    label: str = ""
    image_prompt: str
    motion_prompt: str = ""
    negative_prompt: str = ""
    duration_sec: float = 6.0


class SeriesIntroRequest(CinematicTaskBase):
    title: str
    concept: str
    genre: str = "prestige sci-fi drama"
    style_notes: str = "cinematic lighting, premium lensing, rich atmosphere, netflix-level polish"
    duration_sec: float = 60.0
    scene_count: int = 10
    narration_text: Optional[str] = None
    voice: str = "Vivian"
    music_prompt: Optional[str] = None
    use_i2v: bool = True
    output_profile: str = "cinematic_wide"
    shot_plan: list[CinematicShot] = Field(default_factory=list)
    label: str = "series-intro"


class ImmersiveVideoRequest(CinematicTaskBase):
    concept: str
    style_notes: str = "immersive cinematic visuals, moody atmosphere, premium color, slow camera motion"
    duration_sec: float = 180.0
    scene_count: int = 12
    music_prompt: Optional[str] = None
    use_i2v: bool = True
    output_profile: str = "cinematic_wide"
    shot_plan: list[CinematicShot] = Field(default_factory=list)
    label: str = "immersive-video"


class CinematicDirectorRequest(CinematicTaskBase):
    production_type: str = "auto"
    title: Optional[str] = None
    concept: str
    genre: str = "prestige cinematic drama"
    style_notes: str = "cinematic lighting, premium lensing, rich atmosphere, netflix-level polish"
    duration_sec: float = 60.0
    scene_count: int = 8
    narration_text: Optional[str] = None
    voice: str = "Vivian"
    music_prompt: Optional[str] = None
    source_image_urls: list[str] = Field(default_factory=list)
    use_project_image_assets: bool = False
    use_i2v: bool = True
    output_profile: str = "cinematic_wide"
    shot_plan: list[CinematicShot] = Field(default_factory=list)
    label: str = "cinematic-director"


def _default_intro_narration(req: SeriesIntroRequest) -> str:
    return (
        f"In {req.title}, {req.concept}. "
        f"A {req.genre} world unfolds through danger, mystery, and unforgettable characters."
    )


def _default_intro_music(req: SeriesIntroRequest) -> str:
    return (
        f"prestige television opening theme, {req.genre}, {req.style_notes}, "
        "slow build, emotional tension, cinematic percussion, premium orchestral and electronic fusion"
    )


def _resolve_quality_preset(value: Optional[str]) -> str:
    preset = str(value or "premium").strip().lower()
    if preset in {"standard", "balanced", "default"}:
        return "standard"
    if preset in {"ultra", "hero", "max"}:
        return "ultra"
    return "premium"


def _quality_prompt_suffix(preset: str) -> str:
    resolved = _resolve_quality_preset(preset)
    if resolved == "ultra":
        return (
            "prestige television opening credits quality, cinematic realism, crisp details, "
            "stable geometry, elegant composition, premium production design"
        )
    if resolved == "premium":
        return (
            "prestige television quality, cinematic realism, stable geometry, rich texture detail, "
            "premium production design"
        )
    return "cinematic realism, stable geometry, clean composition"


def _effective_series_intro_duration(duration_sec: float) -> float:
    return max(float(duration_sec or 0.0), 45.0)


def _effective_series_intro_scene_count(scene_count: int, duration_sec: float) -> int:
    duration_target = _effective_series_intro_duration(duration_sec)
    recommended = int(math.ceil(duration_target / 6.0))
    return max(8, min(max(scene_count, recommended), 12))


def _effective_trailer_duration(duration_sec: float) -> float:
    return max(float(duration_sec or 0.0), 30.0)


def _effective_trailer_scene_count(scene_count: int, duration_sec: float) -> int:
    duration_target = _effective_trailer_duration(duration_sec)
    recommended = int(math.ceil(duration_target / 4.5))
    return max(6, min(max(scene_count, recommended), 10))


def _should_enforce_motion_quality(production_type: str, preset: str) -> bool:
    return production_type in {"series_intro", "trailer"} or _resolve_quality_preset(preset) in {"premium", "ultra"}


def _default_immersive_music(req: ImmersiveVideoRequest) -> str:
    return (
        f"immersive cinematic soundtrack for {req.concept}, {req.style_notes}, "
        "long-form atmosphere, evolving tension, premium sound design, emotional melodic arc"
    )


def _default_trailer_narration(req: CinematicDirectorRequest) -> str:
    title = (req.title or "Untitled").strip()
    return (
        f"In {title}, {req.concept}. "
        "Power shifts, loyalties fracture, and the world moves toward a collision no one can survive unchanged."
    )


def _default_trailer_music(req: CinematicDirectorRequest) -> str:
    return (
        f"epic cinematic trailer score, {req.genre}, {req.style_notes}, "
        "slow rise, modern braams, deep percussion, dark choir, emotional climax, premium trailer finish"
    )


def _default_intro_shots(req: SeriesIntroRequest) -> list[CinematicShot]:
    scene_count = _effective_series_intro_scene_count(req.scene_count, req.duration_sec)
    target_duration = _effective_series_intro_duration(req.duration_sec)
    shot_duration = max(min(target_duration / max(scene_count, 1), 7.0), 4.5)
    quality_suffix = _quality_prompt_suffix(req.quality_preset)
    prompts = [
        f"opening world reveal for {req.title}, {req.concept}, {req.genre}, {req.style_notes}, {quality_suffix}",
        f"iconic sigil or symbolic artifact from {req.title}, moody close-up, {req.style_notes}, {quality_suffix}",
        f"hero silhouette in the world of {req.title}, emotionally restrained, {req.style_notes}, {quality_suffix}",
        f"antagonist or looming threat from {req.title}, oppressive scale, {req.style_notes}, {quality_suffix}",
        f"ritual, machine, or political ceremony from {req.title}, premium production design, {req.style_notes}, {quality_suffix}",
        f"intimate character fracture inside {req.title}, emotionally charged, {req.style_notes}, {quality_suffix}",
        f"catastrophic environmental force overtaking {req.title}, scale and dread, {req.style_notes}, {quality_suffix}",
        f"power corridor, throne room, or decisive interior from {req.title}, elegant tension, {req.style_notes}, {quality_suffix}",
        f"dramatic confrontation tableau for {req.title}, prestige TV look, {req.style_notes}, {quality_suffix}",
        f"title reveal shot for {req.title}, iconic final intro frame, premium prestige television finish, {req.style_notes}, {quality_suffix}",
    ]
    motion_prompts = [
        "slow aerial reveal with layered parallax, weather movement, and cinematic drift",
        "measured macro push-in with floating dust, fabric motion, and breathing light",
        "slow heroic push with subtle environmental motion and strong depth",
        "tense creeping move as weather and atmosphere build around the threat",
        "controlled orbit through ritual energy, smoke, cloth, and practical light",
        "gentle emotional push-in with natural motion in hair, costume, and atmosphere",
        "scale move across environmental destruction with debris, fog, and light shafts",
        "tracked glide through architecture with flickering light and atmospheric depth",
        "orbiting confrontation move with sparks, wind, and looming threat",
        "final reveal move settling into a clean title-card composition",
    ]
    shots = []
    for index in range(scene_count):
        prompt_index = min(index, len(prompts) - 1)
        shots.append(
            CinematicShot(
                label=f"scene-{index + 1}",
                image_prompt=prompts[prompt_index],
                motion_prompt=motion_prompts[prompt_index],
                duration_sec=shot_duration,
            )
        )
    return shots


def _default_immersive_shots(req: ImmersiveVideoRequest) -> list[CinematicShot]:
    scene_count = max(6, min(req.scene_count, 18))
    shot_duration = max(min(req.duration_sec / max(scene_count, 1), 10.0), 6.0)
    quality_suffix = _quality_prompt_suffix(req.quality_preset)
    base_prompts = [
        f"wide establishing atmosphere for {req.concept}, {req.style_notes}, {quality_suffix}",
        f"dreamlike close environmental detail for {req.concept}, {req.style_notes}, {quality_suffix}",
        f"slow journey through the world of {req.concept}, {req.style_notes}, {quality_suffix}",
        f"emotional ambient tableau for {req.concept}, {req.style_notes}, {quality_suffix}",
        f"moody architectural or natural scale shot for {req.concept}, {req.style_notes}, {quality_suffix}",
        f"immersive moving light and texture shot for {req.concept}, {req.style_notes}, {quality_suffix}",
    ]
    shots = []
    for index in range(scene_count):
        prompt = base_prompts[index % len(base_prompts)]
        shots.append(
            CinematicShot(
                label=f"movement-{index + 1}",
                image_prompt=prompt,
                motion_prompt=f"slow cinematic drift, immersive movement, premium camera language, shot {index + 1}",
                duration_sec=shot_duration,
            )
        )
    return shots


def _default_trailer_shots(req: CinematicDirectorRequest) -> list[CinematicShot]:
    scene_count = _effective_trailer_scene_count(req.scene_count, req.duration_sec)
    target_duration = _effective_trailer_duration(req.duration_sec)
    shot_duration = max(min(target_duration / max(scene_count, 1), 5.0), 3.5)
    quality_suffix = _quality_prompt_suffix(req.quality_preset)
    prompts = [
        f"opening impact shot for {req.concept}, {req.genre}, {req.style_notes}, {quality_suffix}",
        f"protagonist reveal for {req.concept}, emotionally charged, {req.style_notes}, {quality_suffix}",
        f"world-scale danger escalating around {req.concept}, cinematic urgency, {req.style_notes}, {quality_suffix}",
        f"high energy transition shot for {req.concept}, cinematic momentum, {req.style_notes}, {quality_suffix}",
        f"confrontation beat for {req.concept}, premium dramatic lighting, {req.style_notes}, {quality_suffix}",
        f"iconic final reveal for {req.title or req.concept}, trailer ending image, {req.style_notes}, {quality_suffix}",
    ]
    motion_prompts = [
        "camera surges forward with strong parallax and environmental movement",
        "slow push-in with wind, cloth, hair, and atmospheric movement",
        "fast cinematic sweep across the threat as scale unfolds",
        "dynamic tracking shot with aggressive motion and debris movement",
        "tense orbit around the confrontation as sparks and weather intensify",
        "heroic reveal move ending on an iconic title-card feeling frame",
    ]
    shots = []
    for index in range(scene_count):
        shots.append(
            CinematicShot(
                label=f"trailer-{index + 1}",
                image_prompt=prompts[min(index, len(prompts) - 1)],
                motion_prompt=motion_prompts[min(index, len(motion_prompts) - 1)],
                duration_sec=shot_duration,
            )
        )
    return shots


def _generation_dimensions(profile_name: str) -> tuple[int, int, str]:
    profile = resolve_format_profile(profile_name)
    width = float(profile["width"])
    height = float(profile["height"])
    max_dim = 1536
    if max(width, height) > max_dim:
        scale = max_dim / max(width, height)
        width = width * scale
        height = height * scale

    def _snap_options(value: float) -> list[int]:
        lower = max(512, int(value // 64) * 64)
        upper = max(512, int(((value + 63) // 64)) * 64)
        return sorted({lower, upper})

    target_aspect = width / max(height, 1.0)
    candidates: list[tuple[float, float, int, int]] = []
    target_area = width * height
    for cand_width in _snap_options(width):
        for cand_height in _snap_options(height):
            if cand_width > max_dim or cand_height > max_dim:
                continue
            aspect_error = abs((cand_width / max(cand_height, 1)) - target_aspect)
            area_error = abs((cand_width * cand_height) - target_area) / max(target_area, 1.0)
            candidates.append((aspect_error, area_error, cand_width, cand_height))

    if not candidates:
        final_width = max(512, int(width // 64) * 64)
        final_height = max(512, int(height // 64) * 64)
    else:
        _, _, final_width, final_height = min(candidates)
    return final_width, final_height, f"{final_width}x{final_height}"


def _video_generation_dimensions(profile_name: str, quality_preset: str = "premium") -> tuple[int, int]:
    width, height, _ = _generation_dimensions(profile_name)
    resolved_preset = _resolve_quality_preset(quality_preset)
    max_dim = 1024
    if resolved_preset == "premium":
        max_dim = 1088
    elif resolved_preset == "ultra":
        max_dim = 1152
    if max(width, height) <= max_dim:
        return width, height

    scale = max_dim / float(max(width, height))
    scaled_width = width * scale
    scaled_height = height * scale
    final_width = max(512, int(scaled_width // 64) * 64)
    final_height = max(512, int(scaled_height // 64) * 64)
    return final_width, final_height


def _clip_frame_count(duration_sec: float, fps: float) -> int:
    clip_seconds = max(min(float(duration_sec or 0.0), 3.0), 2.0)
    frames = int(round(clip_seconds * max(float(fps or 24.0), 12.0))) + 1
    return max(49, min(frames, 73))


def _clip_duration(duration_sec: float, fallback: float = 6.0) -> float:
    return max(float(duration_sec or 0.0), fallback)


def _shot_slug(shot: CinematicShot, index: int) -> str:
    return studio._safe_slug(shot.label or f"shot-{index}", f"shot-{index}")


def _shot_spec_payload(shots: list[CinematicShot]) -> list[dict[str, Any]]:
    return [studio._model_dump(shot) for shot in shots]


def _total_shot_duration(shots: list[CinematicShot]) -> float:
    return sum(_clip_duration(shot.duration_sec) for shot in shots)


def _resolve_production_type(req: CinematicDirectorRequest) -> str:
    value = (req.production_type or "auto").strip().lower()
    if value in {"series_intro", "series-intro", "intro"}:
        return "series_intro"
    if value in {"immersive", "immersive_video", "immersive-video", "visualizer"}:
        return "immersive_video"
    if value in {"trailer", "teaser"}:
        return "trailer"
    if req.title and float(req.duration_sec or 0.0) >= 45.0:
        return "series_intro"
    if req.source_image_urls or req.use_project_image_assets:
        return "trailer"
    if req.duration_sec >= 120:
        return "immersive_video"
    return "trailer"


def _project_image_urls(project_id: str) -> list[str]:
    assets = studio._project_assets(project_id)
    image_assets = [asset for asset in assets if str(asset.get("kind", "")).lower() == "image" and asset.get("url")]
    image_assets.sort(key=lambda item: float(item.get("created_at", 0.0)), reverse=True)
    return [str(asset["url"]) for asset in image_assets]


async def _resolve_source_image_urls(req: CinematicDirectorRequest) -> list[str]:
    urls = [url for url in req.source_image_urls if str(url).strip()]
    if req.use_project_image_assets and req.project_id:
        for url in _project_image_urls(req.project_id):
            if url not in urls:
                urls.append(url)
    return urls


async def _write_plan_asset(task_id: str, project_id: Optional[str], filename: str, payload: dict[str, Any], label: str) -> str:
    path = studio._output_path(task_id, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    url = studio._relative_output_url(path)
    studio._attach_asset(project_id, "document", label, url, {"workflow": label})
    return url


async def _synthesize_voice(text: str, voice: str, output_path: str) -> str:
    tts_backend = studio._require_backend("tts")
    payload = {
        "input": text,
        "voice": voice,
        "response_format": os.path.splitext(output_path)[1].lstrip(".") or "mp3",
    }
    async with studio._get_async_lock("tts"):
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(f"{tts_backend}/v1/audio/speech", json=payload)
            response.raise_for_status()
            with open(output_path, "wb") as handle:
                handle.write(response.content)
    return studio._relative_output_url(output_path)


async def _generate_music_track(task_id: str, prompt: str, duration_sec: float, output_name: str) -> dict[str, Any]:
    music_backend = studio._require_backend("music")
    output_path = studio._output_path(task_id, output_name)
    async with studio._get_async_lock("music"):
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{music_backend}/v1/audio/async_generations",
                json={
                    "prompt": prompt,
                    "audio_duration": max(float(duration_sec or 0.0), 8.0),
                    "audio_format": os.path.splitext(output_name)[1].lstrip(".") or "mp3",
                },
            )
            response.raise_for_status()
            init_data = response.json() or {}
            remote_task_id = init_data.get("task_id")
            if not remote_task_id:
                raise RuntimeError(f"Music backend did not return task_id: {init_data}")
            status_data = await studio._poll_task_url(client, f"{music_backend}/v1/audio/tasks/{remote_task_id}")
            data_items = status_data.get("data") or []
            if not data_items:
                raise RuntimeError(f"Music task completed without output data: {status_data}")
            source_url = studio._resolve_remote_url(data_items[0].get("url", ""), music_backend)
            await studio._download_to_path(source_url, output_path)
    return {"path": output_path, "url": studio._relative_output_url(output_path)}


async def _generate_storyboard_image(task_id: str, shot: CinematicShot, index: int, profile_name: str) -> dict[str, Any]:
    vision_backend = studio._require_backend("vision")
    _, _, size = _generation_dimensions(profile_name)
    slug = _shot_slug(shot, index)
    output_path = studio._output_path(task_id, f"{slug}.png")
    async with studio._get_async_lock("vision"):
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{vision_backend}/v1/images/async_generate",
                json={"prompt": shot.image_prompt, "size": size, "cfg_normalization": True},
            )
            response.raise_for_status()
            init_data = response.json() or {}
            remote_task_id = init_data.get("task_id")
            if not remote_task_id:
                raise RuntimeError(f"Vision backend did not return task_id: {init_data}")
            status_data = await studio._poll_task_url(client, f"{vision_backend}/v1/images/tasks/{remote_task_id}")
            data_items = status_data.get("data") or []
            if not data_items:
                raise RuntimeError(f"Vision task completed without image data: {status_data}")
            source_url = studio._resolve_remote_url(data_items[0].get("url", ""), vision_backend)
            await studio._download_to_path(source_url, output_path)
    return {"path": output_path, "url": studio._relative_output_url(output_path)}


async def _generate_i2v_clip(
    task_id: str,
    shot: CinematicShot,
    index: int,
    image_path: str,
    profile_name: str,
    quality_preset: str = "premium",
) -> dict[str, Any]:
    video_backend = studio._require_backend("video")
    profile = resolve_format_profile(profile_name)
    width, height = _video_generation_dimensions(profile_name, quality_preset)
    fps = float(profile["fps"])
    target_duration = _clip_duration(shot.duration_sec)
    slug = _shot_slug(shot, index)
    raw_path = studio._output_path(task_id, f"{slug}-raw.mp4")
    working_path = raw_path
    prompt = shot.motion_prompt.strip() or f"{shot.image_prompt}, slow cinematic camera motion"
    negative_prompt = shot.negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT

    async with studio._get_async_lock("video"):
        async with httpx.AsyncClient(timeout=None) as client:
            with open(image_path, "rb") as image_handle:
                response = await client.post(
                    f"{video_backend}/v1/video/async_i2v",
                    data={
                        "prompt": prompt,
                        "height": str(height),
                        "width": str(width),
                        "num_frames": str(_clip_frame_count(target_duration, fps)),
                        "frame_rate": str(fps),
                        "num_inference_steps": "20",
                        "seed": "-1",
                        "negative_prompt": negative_prompt,
                        "enhance_prompt": "true",
                    },
                    files={"image": (os.path.basename(image_path), image_handle, "image/png")},
                )
            response.raise_for_status()
            init_data = response.json() or {}
            remote_task_id = init_data.get("task_id")
            if not remote_task_id:
                raise RuntimeError(f"Video backend did not return task_id: {init_data}")
            status_data = await studio._poll_task_url(client, f"{video_backend}/v1/video/tasks/{remote_task_id}")
            source_url = studio._resolve_remote_url(status_data.get("url", ""), video_backend)
            if not source_url:
                raise RuntimeError(f"Video task completed without output url: {status_data}")
            await studio._download_to_path(source_url, raw_path)

    raw_duration = await asyncio.to_thread(detect_duration, raw_path)
    if target_duration > (raw_duration or 0.0) + 0.35:
        looped_path = studio._output_path(task_id, f"{slug}-looped.mp4")
        success = await asyncio.to_thread(loop_video_to_duration, raw_path, target_duration, looped_path)
        if not success:
            raise RuntimeError(f"Failed to extend shot {slug} to the requested duration.")
        working_path = looped_path

    output_path = studio._output_path(task_id, f"{slug}.mp4")
    success, message = await asyncio.to_thread(
        normalize_video_clip,
        working_path,
        output_path,
        profile_name,
        0.0,
        target_duration,
    )
    if not success:
        raise RuntimeError(f"Failed to normalize animated shot {slug}: {message}")
    return {"path": output_path, "url": studio._relative_output_url(output_path), "mode": "i2v"}


async def _generate_t2v_clip(
    task_id: str,
    shot: CinematicShot,
    index: int,
    profile_name: str,
    quality_preset: str = "premium",
) -> dict[str, Any]:
    video_backend = studio._require_backend("video")
    profile = resolve_format_profile(profile_name)
    width, height = _video_generation_dimensions(profile_name, quality_preset)
    fps = float(profile["fps"])
    target_duration = _clip_duration(shot.duration_sec)
    slug = _shot_slug(shot, index)
    raw_path = studio._output_path(task_id, f"{slug}-t2v-raw.mp4")
    working_path = raw_path
    prompt_parts = [
        shot.image_prompt.strip(),
        shot.motion_prompt.strip() or "cinematic motion, clear subject movement, strong sense of depth",
        "prestige television shot, cinematic realism, physically plausible motion, crisp details, stable geometry, no slideshow feel",
    ]
    prompt = ", ".join(part for part in prompt_parts if part)
    negative_prompt = shot.negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT

    async with studio._get_async_lock("video"):
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{video_backend}/v1/video/async_t2v",
                json={
                    "prompt": prompt,
                    "height": height,
                    "width": width,
                    "num_frames": _clip_frame_count(target_duration, fps),
                    "frame_rate": fps,
                    "num_inference_steps": 20,
                    "seed": -1,
                    "negative_prompt": negative_prompt,
                    "enhance_prompt": True,
                },
            )
            response.raise_for_status()
            init_data = response.json() or {}
            remote_task_id = init_data.get("task_id")
            if not remote_task_id:
                raise RuntimeError(f"Video backend did not return task_id: {init_data}")
            status_data = await studio._poll_task_url(client, f"{video_backend}/v1/video/tasks/{remote_task_id}")
            source_url = studio._resolve_remote_url(status_data.get("url", ""), video_backend)
            if not source_url:
                raise RuntimeError(f"Video task completed without output url: {status_data}")
            await studio._download_to_path(source_url, raw_path)

    raw_duration = await asyncio.to_thread(detect_duration, raw_path)
    if target_duration > (raw_duration or 0.0) + 0.35:
        looped_path = studio._output_path(task_id, f"{slug}-t2v-looped.mp4")
        success = await asyncio.to_thread(loop_video_to_duration, raw_path, target_duration, looped_path)
        if not success:
            raise RuntimeError(f"Failed to extend generated shot {slug} to the requested duration.")
        working_path = looped_path

    output_path = studio._output_path(task_id, f"{slug}.mp4")
    success, message = await asyncio.to_thread(
        normalize_video_clip,
        working_path,
        output_path,
        profile_name,
        0.0,
        target_duration,
    )
    if not success:
        raise RuntimeError(f"Failed to normalize generated shot {slug}: {message}")
    return {"path": output_path, "url": studio._relative_output_url(output_path), "mode": "t2v"}


async def _render_hold_clip(task_id: str, shot: CinematicShot, index: int, image_path: str, profile_name: str) -> dict[str, Any]:
    slug = _shot_slug(shot, index)
    output_path = studio._output_path(task_id, f"{slug}.mp4")
    success, message = await asyncio.to_thread(
        render_image_hold_clip,
        image_path,
        output_path,
        profile_name,
        _clip_duration(shot.duration_sec),
    )
    if not success:
        raise RuntimeError(f"Failed to render still clip {slug}: {message}")
    return {"path": output_path, "url": studio._relative_output_url(output_path), "mode": "hold"}


async def _build_image_guided_shot_assets(
    task_id: str,
    project_id: Optional[str],
    label: str,
    shots: list[CinematicShot],
    profile_name: str,
    source_image_urls: list[str],
    use_i2v: bool,
    quality_preset: str = "premium",
) -> list[dict[str, Any]]:
    if not source_image_urls:
        raise RuntimeError("Image-guided cinematic runs require at least one source image.")

    rendered_shots = []
    for index, shot in enumerate(shots, start=1):
        studio._raise_if_cancel_requested(task_id)
        source_url = source_image_urls[(index - 1) % len(source_image_urls)]
        ext = studio._guess_extension(source_url, ".png")
        local_image_path = await studio._materialize_source(task_id, source_url, f"{_shot_slug(shot, index)}-source{ext}")
        storyboard_url = studio._relative_output_url(local_image_path)
        studio._attach_asset(
            project_id,
            "image",
            f"{label}-source-{index}",
            storyboard_url,
            {"shot_index": index, "source_url": source_url, "prompt": shot.image_prompt},
        )
        studio._record_task_event(task_id, f"Source image {index} prepared.", f"source-{index}")

        clip = None
        if use_i2v:
            try:
                clip = await _generate_i2v_clip(task_id, shot, index, local_image_path, profile_name, quality_preset)
            except Exception as exc:
                studio._record_task_event(task_id, f"I2V fallback on shot {index}: {exc}", f"shot-{index}-fallback")
        if clip is None:
            try:
                clip = await _generate_t2v_clip(task_id, shot, index, profile_name, quality_preset)
            except Exception as exc:
                studio._record_task_event(task_id, f"T2V fallback on shot {index}: {exc}", f"shot-{index}-fallback")
        if clip is None:
            clip = await _render_hold_clip(task_id, shot, index, local_image_path, profile_name)

        studio._attach_asset(
            project_id,
            "video",
            f"{label}-shot-{index}",
            clip["url"],
            {
                "shot_index": index,
                "animation_mode": clip["mode"],
                "duration_sec": _clip_duration(shot.duration_sec),
                "prompt": shot.image_prompt,
                "source_url": source_url,
            },
        )
        studio._record_task_event(task_id, f"Shot {index} assembled using {clip['mode']}.", f"shot-{index}")
        rendered_shots.append(
            {
                "index": index,
                "label": shot.label or f"shot-{index}",
                "image_prompt": shot.image_prompt,
                "motion_prompt": shot.motion_prompt,
                "duration_sec": _clip_duration(shot.duration_sec),
                "storyboard_url": storyboard_url,
                "clip_url": clip["url"],
                "clip_path": clip["path"],
                "animation_mode": clip["mode"],
            }
        )
    return rendered_shots


async def _build_shot_assets(
    task_id: str,
    project_id: Optional[str],
    label: str,
    shots: list[CinematicShot],
    profile_name: str,
    use_i2v: bool,
    quality_preset: str = "premium",
) -> list[dict[str, Any]]:
    rendered_shots = []
    for index, shot in enumerate(shots, start=1):
        studio._raise_if_cancel_requested(task_id)
        storyboard = await _generate_storyboard_image(task_id, shot, index, profile_name)
        studio._attach_asset(
            project_id,
            "image",
            f"{label}-storyboard-{index}",
            storyboard["url"],
            {"shot_index": index, "prompt": shot.image_prompt},
        )
        studio._record_task_event(task_id, f"Storyboard frame {index} rendered.", f"storyboard-{index}")

        clip = None
        if use_i2v:
            try:
                clip = await _generate_i2v_clip(task_id, shot, index, storyboard["path"], profile_name, quality_preset)
            except Exception as exc:
                studio._record_task_event(
                    task_id,
                    f"I2V fallback on shot {index}: {exc}",
                    f"shot-{index}-fallback",
                )
        if clip is None:
            try:
                clip = await _generate_t2v_clip(task_id, shot, index, profile_name, quality_preset)
            except Exception as exc:
                studio._record_task_event(
                    task_id,
                    f"T2V fallback on shot {index}: {exc}",
                    f"shot-{index}-fallback",
                )
        if clip is None:
            clip = await _render_hold_clip(task_id, shot, index, storyboard["path"], profile_name)

        studio._attach_asset(
            project_id,
            "video",
            f"{label}-shot-{index}",
            clip["url"],
            {
                "shot_index": index,
                "animation_mode": clip["mode"],
                "duration_sec": _clip_duration(shot.duration_sec),
                "prompt": shot.image_prompt,
            },
        )
        studio._record_task_event(task_id, f"Shot {index} assembled using {clip['mode']}.", f"shot-{index}")
        rendered_shots.append(
            {
                "index": index,
                "label": shot.label or f"shot-{index}",
                "image_prompt": shot.image_prompt,
                "motion_prompt": shot.motion_prompt,
                "duration_sec": _clip_duration(shot.duration_sec),
                "storyboard_url": storyboard["url"],
                "clip_url": clip["url"],
                "clip_path": clip["path"],
                "animation_mode": clip["mode"],
            }
        )
    return rendered_shots


def _assert_cinematic_delivery_quality(
    production_type: str,
    quality_preset: str,
    rendered_shots: list[dict[str, Any]],
) -> None:
    if not rendered_shots:
        raise RuntimeError("No rendered cinematic shots were produced.")

    if not _should_enforce_motion_quality(production_type, quality_preset):
        return

    hold_shots = [shot for shot in rendered_shots if shot.get("animation_mode") == "hold"]
    if hold_shots:
        hold_labels = ", ".join(str(shot.get("label") or shot.get("index")) for shot in hold_shots[:4])
        raise RuntimeError(
            f"Cinematic quality floor not met: still-image fallback was used for {len(hold_shots)} shot(s) ({hold_labels})."
        )


async def _run_cinematic_director_task(task_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    req = CinematicDirectorRequest(**spec)
    if not req.project_id:
        raise RuntimeError("Cinematic director runs require a project_id.")
    studio._load_project(req.project_id)
    if not ffmpeg_available():
        raise RuntimeError("FFmpeg is required for cinematic assembly.")

    production_type = _resolve_production_type(req)
    source_image_urls = await _resolve_source_image_urls(req)
    routing_mode = "image_guided" if source_image_urls else "prompt_guided"
    shots = list(req.shot_plan)
    if not shots:
        if production_type == "series_intro":
            shots = _default_intro_shots(
                SeriesIntroRequest(
                    project_id=req.project_id,
                    webhook_url=req.webhook_url,
                    label=req.label,
                    quality_preset=req.quality_preset,
                    title=req.title or "Untitled",
                    concept=req.concept,
                    genre=req.genre,
                    style_notes=req.style_notes,
                    duration_sec=req.duration_sec,
                    scene_count=req.scene_count,
                    narration_text=req.narration_text,
                    voice=req.voice,
                    music_prompt=req.music_prompt,
                    use_i2v=req.use_i2v,
                    output_profile=req.output_profile,
                )
            )
        elif production_type == "immersive_video":
            shots = _default_immersive_shots(
                ImmersiveVideoRequest(
                    project_id=req.project_id,
                    webhook_url=req.webhook_url,
                    label=req.label,
                    quality_preset=req.quality_preset,
                    concept=req.concept,
                    style_notes=req.style_notes,
                    duration_sec=req.duration_sec,
                    scene_count=req.scene_count,
                    music_prompt=req.music_prompt,
                    use_i2v=req.use_i2v,
                    output_profile=req.output_profile,
                )
            )
        else:
            shots = _default_trailer_shots(req)

    include_narration = production_type in {"series_intro", "trailer"}
    title = (req.title or req.concept[:48]).strip() or "Untitled"
    narration_text = (req.narration_text or _default_trailer_narration(req)).strip() if include_narration else None
    if production_type == "series_intro" and not req.narration_text:
        narration_text = _default_intro_narration(
            SeriesIntroRequest(
                project_id=req.project_id,
                quality_preset=req.quality_preset,
                title=title,
                concept=req.concept,
                genre=req.genre,
            )
        )
    music_prompt = (req.music_prompt or _default_trailer_music(req)).strip()
    if production_type == "immersive_video" and not req.music_prompt:
        music_prompt = _default_immersive_music(
            ImmersiveVideoRequest(
                project_id=req.project_id,
                quality_preset=req.quality_preset,
                concept=req.concept,
                style_notes=req.style_notes,
            )
        )
    elif production_type == "series_intro" and not req.music_prompt:
        music_prompt = _default_intro_music(
            SeriesIntroRequest(
                project_id=req.project_id,
                quality_preset=req.quality_preset,
                title=title,
                concept=req.concept,
                genre=req.genre,
                style_notes=req.style_notes,
            )
        )

    plan_url = await _write_plan_asset(
        task_id,
        req.project_id,
        "cinematic-director-plan.json",
        {
            "workflow": "cinematic_director",
            "production_type": production_type,
            "routing_mode": routing_mode,
            "title": title,
            "concept": req.concept,
            "genre": req.genre,
            "style_notes": req.style_notes,
            "duration_sec": req.duration_sec,
            "narration_text": narration_text,
            "music_prompt": music_prompt,
            "output_profile": req.output_profile,
            "source_image_urls": source_image_urls,
            "shots": _shot_spec_payload(shots),
        },
        f"{req.label}-plan",
    )
    studio._record_task_event(task_id, f"Cinematic plan captured using {routing_mode} routing.", "planned")

    if source_image_urls:
        rendered_shots = await _build_image_guided_shot_assets(
            task_id,
            req.project_id,
            req.label,
            shots,
            req.output_profile,
            source_image_urls,
            req.use_i2v,
            req.quality_preset,
        )
    else:
        rendered_shots = await _build_shot_assets(
            task_id,
            req.project_id,
            req.label,
            shots,
            req.output_profile,
            req.use_i2v,
            req.quality_preset,
        )

    _assert_cinematic_delivery_quality(production_type, req.quality_preset, rendered_shots)

    visual = await _assemble_visual_sequence(task_id, [item["clip_path"] for item in rendered_shots])
    studio._attach_asset(req.project_id, "video", f"{req.label}-visual-sequence", visual["url"], {"workflow": req.label})

    narration_url = None
    narration_path = None
    if narration_text:
        narration_path = studio._output_path(task_id, "cinematic-narration.mp3")
        narration_url = await _synthesize_voice(narration_text, req.voice, narration_path)
        studio._attach_asset(req.project_id, "audio", f"{req.label}-narration", narration_url, {"voice": req.voice})
        studio._record_task_event(task_id, "Narration synthesized.", "narration-ready")

    music = await _generate_music_track(task_id, music_prompt, max(req.duration_sec, _total_shot_duration(shots)), "cinematic-score.mp3")
    studio._attach_asset(req.project_id, "audio", f"{req.label}-score", music["url"], {"prompt": music_prompt})
    studio._record_task_event(task_id, "Score generated.", "score-ready")

    final_audio_path = await _mix_or_select_audio(task_id, narration_path, music["path"])
    final_output = await _finalize_video(task_id, req.output_profile, visual["path"], final_audio_path)
    studio._attach_asset(
        req.project_id,
        "video",
        req.label,
        final_output["url"],
        {"workflow": req.label, "title": title, "production_type": production_type, "routing_mode": routing_mode},
    )
    studio._record_task_event(task_id, "Cinematic final cut ready.", "completed")

    return {
        "task_id": task_id,
        "status": "completed",
        "title": title,
        "production_type": production_type,
        "routing_mode": routing_mode,
        "source_image_count": len(source_image_urls),
        "plan_url": plan_url,
        "narration_url": narration_url,
        "music_url": music["url"],
        "visual_sequence_url": visual["url"],
        "final_video_url": final_output["url"],
        "duration_sec": final_output["duration_sec"],
        "shots": [
            {
                "index": item["index"],
                "label": item["label"],
                "storyboard_url": item["storyboard_url"],
                "clip_url": item["clip_url"],
                "animation_mode": item["animation_mode"],
                "duration_sec": item["duration_sec"],
            }
            for item in rendered_shots
        ],
    }


async def _assemble_visual_sequence(task_id: str, clip_paths: list[str]) -> dict[str, Any]:
    if not clip_paths:
        raise RuntimeError("No shot clips were available for cinematic assembly.")
    output_path = studio._output_path(task_id, "visual-sequence.mp4")
    if len(clip_paths) == 1:
        shutil.copy2(clip_paths[0], output_path)
    else:
        success, message = await asyncio.to_thread(concat_video_clips, clip_paths, output_path)
        if not success:
            raise RuntimeError(f"Failed to concatenate cinematic clips: {message}")
    duration = await asyncio.to_thread(detect_duration, output_path)
    return {"path": output_path, "url": studio._relative_output_url(output_path), "duration_sec": duration}


async def _mix_or_select_audio(
    task_id: str,
    narration_path: Optional[str],
    music_path: Optional[str],
    music_volume: float = 0.28,
) -> Optional[str]:
    if narration_path and music_path:
        output_path = studio._output_path(task_id, "mixed-audio.m4a")
        success = await asyncio.to_thread(mix_audio_files, narration_path, music_path, output_path, music_volume)
        if not success:
            raise RuntimeError("Failed to mix narration and score.")
        return output_path
    return narration_path or music_path


async def _finalize_video(task_id: str, profile_name: str, visual_path: str, audio_path: Optional[str]) -> dict[str, Any]:
    final_visual_path = visual_path
    visual_duration = await asyncio.to_thread(detect_duration, visual_path)
    audio_duration = 0.0
    if audio_path:
        audio_duration = await asyncio.to_thread(get_media_duration, audio_path)
        if audio_duration > visual_duration + 0.35:
            looped_path = studio._output_path(task_id, "visual-sequence-looped.mp4")
            success = await asyncio.to_thread(loop_video_to_duration, visual_path, audio_duration, looped_path)
            if not success:
                raise RuntimeError("Failed to extend the visual sequence to the soundtrack length.")
            normalized_path = studio._output_path(task_id, "visual-sequence-looped-normalized.mp4")
            success, message = await asyncio.to_thread(
                normalize_video_clip,
                looped_path,
                normalized_path,
                profile_name,
                0.0,
                audio_duration,
            )
            if not success:
                raise RuntimeError(f"Failed to normalize the extended visual sequence: {message}")
            final_visual_path = normalized_path
            visual_duration = audio_duration

    final_output_path = studio._output_path(task_id, "final-cinematic-cut.mp4")
    if audio_path:
        success = await asyncio.to_thread(mux_video_and_audio, final_visual_path, audio_path, final_output_path)
        if not success:
            raise RuntimeError("Failed to mux the final cinematic cut.")
    else:
        shutil.copy2(final_visual_path, final_output_path)

    return {
        "path": final_output_path,
        "url": studio._relative_output_url(final_output_path),
        "duration_sec": max(audio_duration, visual_duration),
    }


async def _run_series_intro_task(task_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    req = SeriesIntroRequest(**spec)
    if not req.project_id:
        raise RuntimeError("Series intros require a project_id.")
    studio._load_project(req.project_id)
    if not ffmpeg_available():
        raise RuntimeError("FFmpeg is required for cinematic assembly.")

    shots = req.shot_plan or _default_intro_shots(req)
    narration_text = (req.narration_text or _default_intro_narration(req)).strip()
    music_prompt = (req.music_prompt or _default_intro_music(req)).strip()
    plan_url = await _write_plan_asset(
        task_id,
        req.project_id,
        "series-intro-plan.json",
        {
            "workflow": "series_intro",
            "title": req.title,
            "concept": req.concept,
            "genre": req.genre,
            "style_notes": req.style_notes,
            "duration_sec": req.duration_sec,
            "narration_text": narration_text,
            "music_prompt": music_prompt,
            "output_profile": req.output_profile,
            "shots": _shot_spec_payload(shots),
        },
        f"{req.label}-plan",
    )
    studio._record_task_event(task_id, "Series intro plan captured.", "planned")

    rendered_shots = await _build_shot_assets(
        task_id,
        req.project_id,
        req.label,
        shots,
        req.output_profile,
        req.use_i2v,
        req.quality_preset,
    )
    _assert_cinematic_delivery_quality("series_intro", req.quality_preset, rendered_shots)
    visual = await _assemble_visual_sequence(task_id, [item["clip_path"] for item in rendered_shots])
    studio._attach_asset(req.project_id, "video", f"{req.label}-visual-sequence", visual["url"], {"workflow": req.label})

    narration_path = studio._output_path(task_id, "intro-narration.mp3")
    narration_url = await _synthesize_voice(narration_text, req.voice, narration_path)
    studio._attach_asset(req.project_id, "audio", f"{req.label}-narration", narration_url, {"voice": req.voice})
    studio._record_task_event(task_id, "Narration synthesized.", "narration-ready")

    music = await _generate_music_track(task_id, music_prompt, max(req.duration_sec, _total_shot_duration(shots)), "intro-score.mp3")
    studio._attach_asset(req.project_id, "audio", f"{req.label}-score", music["url"], {"prompt": music_prompt})
    studio._record_task_event(task_id, "Score generated.", "score-ready")

    final_audio_path = await _mix_or_select_audio(task_id, narration_path, music["path"])
    final_output = await _finalize_video(task_id, req.output_profile, visual["path"], final_audio_path)
    studio._attach_asset(
        req.project_id,
        "video",
        req.label,
        final_output["url"],
        {"workflow": req.label, "title": req.title, "genre": req.genre},
    )
    studio._record_task_event(task_id, "Series intro final cut ready.", "completed")

    return {
        "task_id": task_id,
        "status": "completed",
        "title": req.title,
        "plan_url": plan_url,
        "narration_url": narration_url,
        "music_url": music["url"],
        "visual_sequence_url": visual["url"],
        "final_video_url": final_output["url"],
        "duration_sec": final_output["duration_sec"],
        "shots": [
            {
                "index": item["index"],
                "label": item["label"],
                "storyboard_url": item["storyboard_url"],
                "clip_url": item["clip_url"],
                "animation_mode": item["animation_mode"],
                "duration_sec": item["duration_sec"],
            }
            for item in rendered_shots
        ],
    }


async def _run_immersive_video_task(task_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    req = ImmersiveVideoRequest(**spec)
    if not req.project_id:
        raise RuntimeError("Immersive videos require a project_id.")
    studio._load_project(req.project_id)
    if not ffmpeg_available():
        raise RuntimeError("FFmpeg is required for cinematic assembly.")

    shots = req.shot_plan or _default_immersive_shots(req)
    music_prompt = (req.music_prompt or _default_immersive_music(req)).strip()
    plan_url = await _write_plan_asset(
        task_id,
        req.project_id,
        "immersive-video-plan.json",
        {
            "workflow": "immersive_video",
            "concept": req.concept,
            "style_notes": req.style_notes,
            "duration_sec": req.duration_sec,
            "music_prompt": music_prompt,
            "output_profile": req.output_profile,
            "shots": _shot_spec_payload(shots),
        },
        f"{req.label}-plan",
    )
    studio._record_task_event(task_id, "Immersive video plan captured.", "planned")

    rendered_shots = await _build_shot_assets(
        task_id,
        req.project_id,
        req.label,
        shots,
        req.output_profile,
        req.use_i2v,
        req.quality_preset,
    )
    _assert_cinematic_delivery_quality("immersive_video", req.quality_preset, rendered_shots)
    visual = await _assemble_visual_sequence(task_id, [item["clip_path"] for item in rendered_shots])
    studio._attach_asset(req.project_id, "video", f"{req.label}-visual-sequence", visual["url"], {"workflow": req.label})

    music = await _generate_music_track(
        task_id,
        music_prompt,
        max(req.duration_sec, _total_shot_duration(shots)),
        "immersive-score.mp3",
    )
    studio._attach_asset(req.project_id, "audio", f"{req.label}-score", music["url"], {"prompt": music_prompt})
    studio._record_task_event(task_id, "Immersive score generated.", "score-ready")

    final_output = await _finalize_video(task_id, req.output_profile, visual["path"], music["path"])
    studio._attach_asset(
        req.project_id,
        "video",
        req.label,
        final_output["url"],
        {"workflow": req.label, "concept": req.concept},
    )
    studio._record_task_event(task_id, "Immersive final cut ready.", "completed")

    return {
        "task_id": task_id,
        "status": "completed",
        "concept": req.concept,
        "plan_url": plan_url,
        "music_url": music["url"],
        "visual_sequence_url": visual["url"],
        "final_video_url": final_output["url"],
        "duration_sec": final_output["duration_sec"],
        "shots": [
            {
                "index": item["index"],
                "label": item["label"],
                "storyboard_url": item["storyboard_url"],
                "clip_url": item["clip_url"],
                "animation_mode": item["animation_mode"],
                "duration_sec": item["duration_sec"],
            }
            for item in rendered_shots
        ],
    }


studio.TASK_RUNNERS.update(
    {
        "series_intro": _run_series_intro_task,
        "immersive_video": _run_immersive_video_task,
        "cinematic_director": _run_cinematic_director_task,
    }
)


@cinematic_router.post("/projects/{project_id}/cinematic-productions")
async def create_cinematic_production(project_id: str, request: Request):
    req = await studio._parse_model(request, CinematicDirectorRequest)
    payload = studio._model_dump(req)
    payload["project_id"] = project_id
    return studio._queue_task("cinematic_director", "cinematic_director", payload, project_id, req.webhook_url, req.label)


@cinematic_router.post("/projects/{project_id}/series-intros")
async def create_series_intro(project_id: str, request: Request):
    req = await studio._parse_model(request, SeriesIntroRequest)
    payload = studio._model_dump(req)
    payload["project_id"] = project_id
    return studio._queue_task("series_intro", "series_intro", payload, project_id, req.webhook_url, req.label)


@cinematic_router.post("/projects/{project_id}/immersive-videos")
async def create_immersive_video(project_id: str, request: Request):
    req = await studio._parse_model(request, ImmersiveVideoRequest)
    payload = studio._model_dump(req)
    payload["project_id"] = project_id
    return studio._queue_task("immersive_video", "immersive_video", payload, project_id, req.webhook_url, req.label)
