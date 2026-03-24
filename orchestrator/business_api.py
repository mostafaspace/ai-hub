import asyncio
import ast
import json
import os
import re
from html.parser import HTMLParser
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from orchestrator import studio_api as studio
from orchestrator.business_media import ffmpeg_available, render_video_clip, replace_audio_track
from orchestrator.studio_media import (
    create_contact_sheet,
    detect_duration,
    extract_audio_for_transcription,
    extract_thumbnail,
    split_subtitle_text,
    write_approximate_srt,
)


business_router = APIRouter(prefix="/v1/studio", tags=["Business Studio"])


class BusinessTaskBase(BaseModel):
    project_id: Optional[str] = None
    webhook_url: Optional[str] = None
    label: str = "business-workflow"


class ShortFormPackRequest(BusinessTaskBase):
    media_url: str
    target_count: int = 5
    clip_duration_sec: Optional[float] = None
    output_profiles: list[str] = Field(default_factory=lambda: ["youtube_short", "tiktok_vertical"])
    burn_captions: bool = True
    include_contact_sheet: bool = True
    label: str = "short-form-pack"


class EpisodePackRequest(BusinessTaskBase):
    source_text: Optional[str] = None
    source_url: Optional[str] = None
    brochure_url: Optional[str] = None
    title: Optional[str] = None
    voice: str = "Vivian"
    audition_voices: list[str] = Field(default_factory=lambda: ["Vivian", "Ethan", "Serena"])
    language: Optional[str] = None
    cover_prompt: Optional[str] = None
    generate_teaser_video: bool = True
    label: str = "episode-pack"


class LocalizationRunRequest(BusinessTaskBase):
    media_url: str
    target_languages: list[str] = Field(default_factory=lambda: ["Arabic"])
    voice: str = "Vivian"
    burn_subtitles: bool = True
    subtitle_profile: str = "discord_clip"
    label: str = "localization-run"


class MeetingDeliverablesRequest(BusinessTaskBase):
    media_url: str
    recipient_name: Optional[str] = None
    sender_name: Optional[str] = None
    briefing_voice: str = "Vivian"
    stakeholder_clip_duration_sec: float = 30.0
    label: str = "meeting-deliverables"


class MarketingKitRequest(BusinessTaskBase):
    source_text: Optional[str] = None
    source_url: Optional[str] = None
    brochure_url: Optional[str] = None
    product_name: Optional[str] = None
    target_audiences: list[str] = Field(default_factory=list)
    voice: str = "Vivian"
    generate_teaser_video: bool = True
    label: str = "marketing-kit"


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        cleaned = " ".join(data.split())
        if cleaned:
            self.parts.append(cleaned)

    def text(self) -> str:
        return "\n".join(self.parts)


def _clip_count(target_count: int) -> int:
    return max(1, min(int(target_count or 1), 20))


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", _clean_text(text))
    return [part.strip() for part in raw if part.strip()]


def _fallback_title(text: str, fallback: str = "launch-ready-asset") -> str:
    sentences = _sentences(text)
    title = sentences[0] if sentences else _clean_text(text)
    words = title.split()
    if not words:
        return fallback
    return " ".join(words[:8]).strip(" -:,.") or fallback


def _fallback_hook(text: str) -> str:
    words = _clean_text(text).split()
    if not words:
        return "Watch this quick breakdown."
    lead = " ".join(words[:12]).strip(" -:,.")
    return f"{lead}..."


def _jsonish_loads(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        try:
            return ast.literal_eval(raw)
        except Exception:
            try:
                return ast.literal_eval(studio._pythonize_json_literals(raw))
            except Exception:
                return None


def _estimate_windows(duration_seconds: float, target_count: int, clip_duration_sec: Optional[float]) -> list[dict[str, float]]:
    target_count = _clip_count(target_count)
    if duration_seconds <= 0:
        duration_seconds = max(float(clip_duration_sec or 30.0) * target_count, 30.0)

    preferred = float(clip_duration_sec or max(min(duration_seconds / target_count, 45.0), 15.0))
    preferred = max(min(preferred, max(duration_seconds, preferred)), 8.0)
    windows: list[dict[str, float]] = []
    last_end = 0.0
    for index in range(target_count):
        remaining = max(duration_seconds - last_end, 0.0)
        if index == target_count - 1:
            clip_duration = max(min(remaining, preferred), min(remaining or preferred, preferred))
        else:
            clips_left = target_count - index
            clip_duration = min(preferred, max(duration_seconds / max(target_count, 1), 8.0))
            max_for_balance = max((duration_seconds - last_end) / max(clips_left, 1), 8.0)
            clip_duration = min(clip_duration, max_for_balance)
        clip_duration = max(min(clip_duration, max(duration_seconds - last_end, preferred)), 8.0)
        if last_end + clip_duration > duration_seconds:
            clip_duration = max(duration_seconds - last_end, 8.0)
        start_sec = min(last_end, max(duration_seconds - clip_duration, 0.0))
        windows.append({"start_sec": round(start_sec, 3), "duration_sec": round(clip_duration, 3)})
        last_end = start_sec + clip_duration
        if last_end >= duration_seconds:
            break
    return windows or [{"start_sec": 0.0, "duration_sec": round(min(duration_seconds, preferred), 3)}]


def _transcript_chunks_for_windows(transcript: str, window_count: int) -> list[str]:
    sentences = _sentences(transcript)
    if not sentences:
        return [transcript.strip()] * max(window_count, 1)
    bucket_size = max(len(sentences) // max(window_count, 1), 1)
    chunks = []
    for index in range(window_count):
        start = index * bucket_size
        end = len(sentences) if index == window_count - 1 else min((index + 1) * bucket_size, len(sentences))
        chunk = " ".join(sentences[start:end]).strip()
        chunks.append(chunk or sentences[min(start, len(sentences) - 1)])
    return chunks


async def _call_writer(prompt: str, *, audio_path: Optional[str] = None, max_tokens: int = 512) -> Optional[str]:
    asr_backend = studio._require_backend("asr")
    if audio_path:
        content: Any = [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt
    payload = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.9,
    }
    async with studio._get_async_lock("asr"):
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(f"{asr_backend}/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json() or {}
    return (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()


async def _write_text_asset(task_id: str, filename: str, content: str, project_id: Optional[str], label: str, metadata: Optional[dict[str, Any]] = None) -> str:
    output_path = studio._output_path(task_id, filename)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    output_url = studio._relative_output_url(output_path)
    studio._attach_asset(project_id, "document", label, output_url, metadata or {})
    return output_url


async def _fetch_source_text(task_id: str, source_text: Optional[str], source_url: Optional[str], brochure_url: Optional[str]) -> str:
    if source_text and _clean_text(source_text):
        return _clean_text(source_text)

    source = source_url or brochure_url
    if not source:
        return ""

    local_path = await studio._materialize_source(task_id, source, f"brief{studio._guess_extension(source, '.txt')}")
    ext = os.path.splitext(local_path)[1].lower()
    if ext in {".txt", ".md", ".csv", ".json"}:
        with open(local_path, "r", encoding="utf-8", errors="replace") as handle:
            return _clean_text(handle.read())
    if ext in {".html", ".htm"}:
        parser = _HTMLTextExtractor()
        with open(local_path, "r", encoding="utf-8", errors="replace") as handle:
            parser.feed(handle.read())
        return _clean_text(parser.text())
    if ext == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(local_path)
            parts = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
            return _clean_text("\n".join(parts))
        except Exception:
            return ""
    with open(local_path, "r", encoding="utf-8", errors="replace") as handle:
        return _clean_text(handle.read())


async def _prepare_media(task_id: str, media_url: str) -> dict[str, Any]:
    source_ext = studio._guess_extension(media_url, ".bin")
    source_path = await studio._materialize_source(task_id, media_url, f"source{source_ext}")
    is_video = source_ext.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}
    audio_path = source_path
    if is_video:
        audio_path = studio._output_path(task_id, "source-audio.wav")
        async with studio._get_async_lock("studio-media"):
            ok, message = await asyncio.to_thread(extract_audio_for_transcription, source_path, audio_path)
        if not ok:
            raise RuntimeError(f"Failed to extract audio from source media: {message}")
    duration = await asyncio.to_thread(detect_duration, source_path if is_video else audio_path)
    return {
        "source_path": source_path,
        "audio_path": audio_path,
        "duration_seconds": duration,
        "source_ext": source_ext.lower(),
        "is_video": is_video,
    }


async def _transcribe_audio(audio_path: str, prompt: str = "Transcribe this media accurately.") -> str:
    asr_backend = studio._require_backend("asr")
    async with studio._get_async_lock("asr"):
        async with httpx.AsyncClient(timeout=None) as client:
            with open(audio_path, "rb") as audio_handle:
                response = await client.post(
                    f"{asr_backend}/v1/audio/transcriptions",
                    files={"file": (os.path.basename(audio_path), audio_handle, "audio/wav")},
                    data={"prompt": prompt},
                )
            response.raise_for_status()
            data = response.json() or {}
    return (data.get("text") or "").strip()


async def _synthesize_voice(text: str, voice: str, output_path: str, language: Optional[str] = None, instruct: Optional[str] = None) -> str:
    tts_backend = studio._require_backend("tts")
    payload: dict[str, Any] = {
        "input": text,
        "voice": voice,
        "response_format": os.path.splitext(output_path)[1].lstrip(".") or "mp3",
    }
    if language:
        payload["language"] = language
    if instruct:
        payload["instruct"] = instruct
    async with studio._get_async_lock("tts"):
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(f"{tts_backend}/v1/audio/speech", json=payload)
            response.raise_for_status()
            with open(output_path, "wb") as handle:
                handle.write(response.content)
    return studio._relative_output_url(output_path)


async def _render_image_from_prompt(task_id: str, prompt: str, output_name: str = "hero.png") -> str:
    vision_backend = studio._require_backend("vision")
    async with studio._get_async_lock("vision"):
        async with httpx.AsyncClient(timeout=None) as client:
            init_response = await client.post(
                f"{vision_backend}/v1/images/async_generate",
                json={"prompt": prompt, "size": "1024x1024", "cfg_normalization": True},
            )
            init_response.raise_for_status()
            init_data = init_response.json() or {}
            task_id_remote = init_data.get("task_id")
            if not task_id_remote:
                raise RuntimeError(f"Vision backend did not return task_id: {init_data}")
            status_data = await studio._poll_task_url(client, f"{vision_backend}/v1/images/tasks/{task_id_remote}")
            data_items = status_data.get("data") or []
            if not data_items:
                raise RuntimeError(f"Vision task completed without image data: {status_data}")
            source_url = studio._resolve_remote_url(data_items[0].get("url", ""), vision_backend)
            output_path = studio._output_path(task_id, output_name)
            await studio._download_to_path(source_url, output_path)
    return studio._relative_output_url(output_path)


async def _run_director_and_wait(image_prompt: str, voiceover_text: str, voice: str) -> dict[str, Any]:
    async with studio._get_async_lock("director"):
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{studio._loopback_base_url()}/v1/workflows/director",
                json={"image_prompt": image_prompt, "voiceover_text": voiceover_text, "voice": voice},
            )
            response.raise_for_status()
            data = response.json() or {}
            task_id = data.get("task_id")
            if not task_id:
                raise RuntimeError(f"Director did not return task_id: {data}")
            status = await studio._poll_task_url(client, f"{studio._loopback_base_url()}/v1/hub/tasks/{task_id}")
    return status


async def _writer_json(prompt: str, fallback: dict[str, Any], *, audio_path: Optional[str] = None, max_tokens: int = 768) -> dict[str, Any]:
    try:
        response = await _call_writer(prompt, audio_path=audio_path, max_tokens=max_tokens)
    except Exception:
        response = None
    if response:
        parsed = _jsonish_loads(response)
        if isinstance(parsed, dict):
            return parsed
    return fallback


async def _run_short_form_pack_task(task_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    req = ShortFormPackRequest(**spec)
    if not req.project_id:
        raise RuntimeError("Short-form packs require a project_id.")
    studio._load_project(req.project_id)
    if not ffmpeg_available():
        raise RuntimeError("FFmpeg is required for short-form clip packs.")

    media = await _prepare_media(task_id, req.media_url)
    transcript = await _transcribe_audio(media["audio_path"])
    if not transcript:
        raise RuntimeError("ASR did not return transcript text for short-form pack.")
    studio._record_task_event(task_id, "Transcript captured for clip planning.", "transcribed")

    windows = _estimate_windows(media["duration_seconds"], req.target_count, req.clip_duration_sec)
    excerpts = _transcript_chunks_for_windows(transcript, len(windows))
    primary_profile = req.output_profiles[0] if req.output_profiles else "youtube_short"
    clips = []

    if req.include_contact_sheet and media["is_video"]:
        sheet_path = studio._output_path(task_id, "source-contact-sheet.jpg")
        async with studio._get_async_lock("studio-media"):
            ok, _ = await asyncio.to_thread(create_contact_sheet, media["source_path"], sheet_path, 4, 2, 320)
        if ok:
            sheet_url = studio._relative_output_url(sheet_path)
            studio._attach_asset(req.project_id, "image", f"{req.label}-contact-sheet", sheet_url, {"workflow": req.label})

    for index, window in enumerate(windows, start=1):
        studio._raise_if_cancel_requested(task_id)
        excerpt = excerpts[index - 1] if index - 1 < len(excerpts) else transcript
        title = _fallback_title(excerpt, f"Clip {index}")
        hook = _fallback_hook(excerpt)
        subtitle_path = None
        subtitle_url = None
        if req.burn_captions:
            subtitle_path = studio._output_path(task_id, f"clip-{index:02d}.srt")
            await asyncio.to_thread(write_approximate_srt, excerpt or transcript, float(window["duration_sec"]), subtitle_path)
            subtitle_url = studio._relative_output_url(subtitle_path)
            studio._attach_asset(req.project_id, "subtitle", f"{req.label}-captions-{index}", subtitle_url, {"clip_index": index})

        primary_output = studio._output_path(task_id, f"clip-{index:02d}.{studio.resolve_format_profile(primary_profile)['container']}")
        async with studio._get_async_lock("studio-media"):
            ok, message = await asyncio.to_thread(
                render_video_clip,
                media["source_path"],
                primary_output,
                primary_profile,
                float(window["start_sec"]),
                float(window["duration_sec"]),
                subtitle_path,
            )
        if not ok:
            raise RuntimeError(f"Failed to render clip {index}: {message}")
        primary_url = studio._relative_output_url(primary_output)
        studio._attach_asset(req.project_id, "video", f"{req.label}-clip-{index}", primary_url, {"profile": primary_profile, "title": title})

        thumb_path = studio._output_path(task_id, f"clip-{index:02d}.jpg")
        thumb_offset = float(window["start_sec"]) + max(float(window["duration_sec"]) / 2.0, 0.0)
        async with studio._get_async_lock("studio-media"):
            ok, message = await asyncio.to_thread(extract_thumbnail, media["source_path"], thumb_path, thumb_offset)
        if not ok:
            raise RuntimeError(f"Failed to create thumbnail for clip {index}: {message}")
        thumb_url = studio._relative_output_url(thumb_path)
        studio._attach_asset(req.project_id, "image", f"{req.label}-thumb-{index}", thumb_url, {"clip_index": index})

        variants = [{"profile": primary_profile, "url": primary_url}]
        for profile_name in req.output_profiles[1:]:
            studio._raise_if_cancel_requested(task_id)
            profile = studio.resolve_format_profile(profile_name)
            variant_path = studio._output_path(task_id, f"clip-{index:02d}-{profile_name}.{profile['container']}")
            async with studio._get_async_lock("studio-media"):
                ok, message = await asyncio.to_thread(
                    render_video_clip,
                    media["source_path"],
                    variant_path,
                    profile_name,
                    float(window["start_sec"]),
                    float(window["duration_sec"]),
                    subtitle_path,
                )
            if not ok:
                raise RuntimeError(f"Failed to render {profile_name} variant for clip {index}: {message}")
            variant_url = studio._relative_output_url(variant_path)
            variants.append({"profile": profile_name, "url": variant_url})
            studio._attach_asset(req.project_id, "video", f"{req.label}-clip-{index}-{profile_name}", variant_url, {"profile": profile_name, "title": title})

        clips.append(
            {
                "index": index,
                "title": title,
                "hook": hook,
                "excerpt": excerpt,
                "thumbnail_url": thumb_url,
                "subtitle_url": subtitle_url,
                "start_sec": float(window["start_sec"]),
                "duration_sec": float(window["duration_sec"]),
                "variants": variants,
            }
        )
        studio._record_task_event(task_id, f"Short-form clip {index} packaged.", f"clip-{index}")

    transcript_url = await _write_text_asset(task_id, "short-form-transcript.txt", transcript, req.project_id, f"{req.label}-transcript")
    return {
        "task_id": task_id,
        "status": "completed",
        "transcript_url": transcript_url,
        "clips": clips,
    }


def _episode_fallback(text: str, title: Optional[str]) -> dict[str, Any]:
    source_sentences = _sentences(text)
    focus = source_sentences[:6] or [text or "Today we are covering the key ideas from this brief."]
    episode_title = title or _fallback_title(text, "Weekly Briefing")
    intro = f"Welcome back. Today we are unpacking {episode_title.lower()}."
    body = " ".join(focus[:4])
    outro = "That is the quick briefing. Thanks for tuning in."
    teaser = " ".join((focus[:2] or [body]))[:220]
    cover_prompt = f"cinematic podcast cover art for {episode_title}, studio lighting, premium broadcast branding"
    return {
        "title": episode_title,
        "script": " ".join(part for part in [intro, body, outro] if part).strip(),
        "show_notes": "\n".join(f"- {sentence}" for sentence in focus[:5]),
        "teaser_script": teaser or body,
        "cover_prompt": cover_prompt,
    }


async def _run_episode_pack_task(task_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    req = EpisodePackRequest(**spec)
    if not req.project_id:
        raise RuntimeError("Episode packs require a project_id.")
    studio._load_project(req.project_id)

    source_text = await _fetch_source_text(task_id, req.source_text, req.source_url, req.brochure_url)
    if not source_text:
        raise RuntimeError("Episode packs require source_text, source_url, or brochure_url.")

    fallback = _episode_fallback(source_text, req.title)
    writer_prompt = (
        "Create a compact podcast production package as strict JSON with keys "
        "`title`, `script`, `show_notes`, `teaser_script`, and `cover_prompt`. "
        "Use the following source material:\n\n"
        f"{source_text[:6000]}"
    )
    plan = await _writer_json(writer_prompt, fallback, max_tokens=900)
    title = _clean_text(str(plan.get("title") or fallback["title"]))
    script = _clean_text(str(plan.get("script") or fallback["script"]))
    show_notes = _clean_text(str(plan.get("show_notes") or fallback["show_notes"]))
    teaser_script = _clean_text(str(plan.get("teaser_script") or fallback["teaser_script"]))
    cover_prompt = _clean_text(str(req.cover_prompt or plan.get("cover_prompt") or fallback["cover_prompt"]))

    script_url = await _write_text_asset(
        task_id,
        "episode-script.md",
        "\n".join(
            [
                f"# {title}",
                "",
                "## Script",
                "",
                script,
                "",
                "## Show Notes",
                "",
                show_notes,
                "",
            ]
        ),
        req.project_id,
        f"{req.label}-script",
        {"title": title},
    )

    samples = []
    for index, voice in enumerate(req.audition_voices[:5], start=1):
        studio._raise_if_cancel_requested(task_id)
        sample_path = studio._output_path(task_id, f"audition-{index:02d}-{studio._safe_slug(voice, 'voice')}.mp3")
        sample_url = await _synthesize_voice(script[:420], voice, sample_path, language=req.language)
        studio._attach_asset(req.project_id, "audio", f"{req.label}-audition-{voice}", sample_url, {"voice": voice})
        samples.append({"voice": voice, "url": sample_url})

    episode_audio_path = studio._output_path(task_id, "episode.mp3")
    episode_audio_url = await _synthesize_voice(script, req.voice, episode_audio_path, language=req.language)
    studio._attach_asset(req.project_id, "audio", req.label, episode_audio_url, {"workflow": req.label, "title": title, "voice": req.voice})
    studio._record_task_event(task_id, "Episode narration synthesized.", "narration-ready")

    cover_url = await _render_image_from_prompt(task_id, cover_prompt, "episode-cover.png")
    studio._attach_asset(req.project_id, "image", f"{req.label}-cover", cover_url, {"prompt": cover_prompt, "title": title})

    teaser_result = None
    if req.generate_teaser_video:
        teaser_result = await _run_director_and_wait(cover_prompt, teaser_script or script[:220], req.voice)
        teaser_url = teaser_result.get("output_url")
        if teaser_url:
            studio._attach_asset(req.project_id, "video", f"{req.label}-teaser", teaser_url, {"workflow": "director", "title": title})

    return {
        "task_id": task_id,
        "status": "completed",
        "title": title,
        "script_url": script_url,
        "voice_auditions": samples,
        "episode_audio_url": episode_audio_url,
        "cover_image_url": cover_url,
        "teaser": teaser_result,
    }


async def _run_localization_task(task_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    req = LocalizationRunRequest(**spec)
    if not req.project_id:
        raise RuntimeError("Localization runs require a project_id.")
    studio._load_project(req.project_id)
    if not ffmpeg_available():
        raise RuntimeError("FFmpeg is required for localization runs.")

    media = await _prepare_media(task_id, req.media_url)
    transcript = await _transcribe_audio(media["audio_path"])
    if not transcript:
        raise RuntimeError("ASR did not return transcript text for localization.")

    localizations = []
    for language in req.target_languages[:5]:
        studio._raise_if_cancel_requested(task_id)
        translation_prompt = (
            f"Translate the following spoken content into {language}. "
            "Return only the translated narration text with no commentary.\n\n"
            f"{transcript}"
        )
        translated_text = None
        try:
            translated_text = await _call_writer(translation_prompt, max_tokens=900)
        except Exception:
            translated_text = None
        translated_text = _clean_text(translated_text or transcript)

        audio_path = studio._output_path(task_id, f"dub-{studio._safe_slug(language, 'lang')}.mp3")
        audio_url = await _synthesize_voice(
            translated_text,
            req.voice,
            audio_path,
            language=language,
            instruct=f"Speak naturally in {language} with clear timing.",
        )
        studio._attach_asset(req.project_id, "audio", f"{req.label}-{language}-audio", audio_url, {"language": language})

        dubbed_video_url = None
        burned_video_url = None
        subtitle_url = None
        if media["is_video"]:
            dubbed_path = studio._output_path(task_id, f"dubbed-{studio._safe_slug(language, 'lang')}.mp4")
            async with studio._get_async_lock("studio-media"):
                ok, message = await asyncio.to_thread(replace_audio_track, media["source_path"], audio_path, dubbed_path)
            if not ok:
                raise RuntimeError(f"Failed to replace audio track for {language}: {message}")
            dubbed_video_url = studio._relative_output_url(dubbed_path)
            studio._attach_asset(req.project_id, "video", f"{req.label}-{language}", dubbed_video_url, {"language": language, "dubbed": True})

            subtitle_path = studio._output_path(task_id, f"dubbed-{studio._safe_slug(language, 'lang')}.srt")
            await asyncio.to_thread(write_approximate_srt, translated_text, media["duration_seconds"], subtitle_path)
            subtitle_url = studio._relative_output_url(subtitle_path)
            studio._attach_asset(req.project_id, "subtitle", f"{req.label}-{language}-captions", subtitle_url, {"language": language})

            if req.burn_subtitles:
                burned_path = studio._output_path(task_id, f"dubbed-{studio._safe_slug(language, 'lang')}-burned.mp4")
                async with studio._get_async_lock("studio-media"):
                    ok, message = await asyncio.to_thread(
                        render_video_clip,
                        dubbed_path,
                        burned_path,
                        req.subtitle_profile,
                        0.0,
                        None,
                        subtitle_path,
                    )
                if not ok:
                    raise RuntimeError(f"Failed to burn subtitles for {language}: {message}")
                burned_video_url = studio._relative_output_url(burned_path)
                studio._attach_asset(req.project_id, "video", f"{req.label}-{language}-burned", burned_video_url, {"language": language, "burned": True})

        localizations.append(
            {
                "language": language,
                "translated_text": translated_text,
                "audio_url": audio_url,
                "dubbed_video_url": dubbed_video_url,
                "subtitle_url": subtitle_url,
                "burned_video_url": burned_video_url,
            }
        )
        studio._record_task_event(task_id, f"Localization package ready for {language}.", studio._safe_slug(language, "language"))

    transcript_url = await _write_text_asset(task_id, "source-transcript.txt", transcript, req.project_id, f"{req.label}-source-transcript")
    return {"task_id": task_id, "status": "completed", "transcript_url": transcript_url, "localizations": localizations}


def _meeting_fallback(transcript: str, req: MeetingDeliverablesRequest) -> dict[str, Any]:
    sentences = _sentences(transcript)
    summary = " ".join(sentences[:3]) if sentences else transcript[:240]
    actions = sentences[3:6] if len(sentences) > 3 else ["Review the transcript and assign clear owners for the next steps."]
    email_lines = [
        f"Hi {req.recipient_name or 'team'},",
        "",
        "Thanks for the meeting today. Here is the quick recap:",
        summary,
        "",
        "Next steps:",
    ]
    email_lines.extend(f"- {item}" for item in actions[:5])
    email_lines.extend(["", f"Best,", req.sender_name or "OpenClaw"])
    return {
        "summary": summary,
        "action_items": actions[:5],
        "follow_up_email": "\n".join(email_lines),
        "briefing_script": f"Here is your stakeholder briefing. {summary}",
    }


async def _run_meeting_deliverables_task(task_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    req = MeetingDeliverablesRequest(**spec)
    if not req.project_id:
        raise RuntimeError("Meeting deliverables require a project_id.")
    studio._load_project(req.project_id)
    if not ffmpeg_available():
        raise RuntimeError("FFmpeg is required for meeting deliverables.")

    media = await _prepare_media(task_id, req.media_url)
    transcript = await _transcribe_audio(media["audio_path"], "Transcribe this meeting recording accurately.")
    if not transcript:
        raise RuntimeError("ASR did not return transcript text for meeting deliverables.")
    studio._record_task_event(task_id, "Meeting transcript captured.", "transcribed")

    fallback = _meeting_fallback(transcript, req)
    prompt = (
        "Return strict JSON with keys `summary`, `action_items`, `follow_up_email`, and `briefing_script` "
        "for the following meeting transcript:\n\n"
        f"{transcript[:7000]}"
    )
    plan = await _writer_json(prompt, fallback, max_tokens=1000)
    summary = _clean_text(str(plan.get("summary") or fallback["summary"]))
    action_items = plan.get("action_items")
    if not isinstance(action_items, list):
        action_items = fallback["action_items"]
    action_items = [_clean_text(str(item)) for item in action_items if _clean_text(str(item))]
    follow_up_email = str(plan.get("follow_up_email") or fallback["follow_up_email"]).strip()
    briefing_script = _clean_text(str(plan.get("briefing_script") or fallback["briefing_script"]))

    summary_url = await _write_text_asset(
        task_id,
        "meeting-summary.md",
        "\n".join(
            [
                "# Meeting Summary",
                "",
                summary,
                "",
                "## Action Items",
                "",
                *[f"- {item}" for item in action_items],
                "",
            ]
        ),
        req.project_id,
        f"{req.label}-summary",
    )
    transcript_url = await _write_text_asset(task_id, "meeting-transcript.txt", transcript, req.project_id, f"{req.label}-transcript")
    email_url = await _write_text_asset(task_id, "follow-up-email.txt", follow_up_email, req.project_id, f"{req.label}-email")

    briefing_audio_path = studio._output_path(task_id, "stakeholder-briefing.mp3")
    briefing_audio_url = await _synthesize_voice(briefing_script, req.briefing_voice, briefing_audio_path)
    studio._attach_asset(req.project_id, "audio", f"{req.label}-briefing", briefing_audio_url, {"workflow": req.label})

    stakeholder_clip_url = None
    if media["is_video"]:
        clip_text = " ".join(_sentences(transcript)[:3]) or summary
        subtitle_path = studio._output_path(task_id, "stakeholder-clip.srt")
        await asyncio.to_thread(write_approximate_srt, clip_text, req.stakeholder_clip_duration_sec, subtitle_path)
        stakeholder_path = studio._output_path(task_id, "stakeholder-clip.mp4")
        async with studio._get_async_lock("studio-media"):
            ok, message = await asyncio.to_thread(
                render_video_clip,
                media["source_path"],
                stakeholder_path,
                "discord_clip",
                0.0,
                req.stakeholder_clip_duration_sec,
                subtitle_path,
            )
        if not ok:
            raise RuntimeError(f"Failed to render stakeholder clip: {message}")
        stakeholder_clip_url = studio._relative_output_url(stakeholder_path)
        studio._attach_asset(req.project_id, "video", f"{req.label}-stakeholder-clip", stakeholder_clip_url, {"workflow": req.label})

    return {
        "task_id": task_id,
        "status": "completed",
        "summary_url": summary_url,
        "transcript_url": transcript_url,
        "follow_up_email_url": email_url,
        "briefing_audio_url": briefing_audio_url,
        "stakeholder_clip_url": stakeholder_clip_url,
        "action_items": action_items,
    }


def _marketing_fallback(source_text: str, product_name: Optional[str], target_audiences: list[str]) -> dict[str, Any]:
    title = product_name or _fallback_title(source_text, "Offer")
    bullets = _sentences(source_text)[:4] or [source_text[:180]]
    audiences = ", ".join(target_audiences[:3]) or "ambitious buyers"
    hooks = [
        f"Why {audiences} are switching to {title}.",
        f"Stop wasting time. Start using {title}.",
        f"The faster way to get results with {title}.",
    ]
    hero_prompt = f"premium ecommerce hero image for {title}, polished studio lighting, conversion-focused product marketing"
    comparison_prompt = f"clean comparison marketing visual for {title}, premium SaaS landing page graphic, bold headline treatment"
    voice_script = " ".join([f"{title} helps teams move faster.", *bullets[:2], "Try it today."]).strip()
    return {
        "product_name": title,
        "hero_headline": title,
        "benefit_bullets": bullets[:4],
        "hooks": hooks,
        "voice_script": voice_script,
        "hero_prompt": hero_prompt,
        "comparison_prompt": comparison_prompt,
    }


async def _run_marketing_kit_task(task_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    req = MarketingKitRequest(**spec)
    if not req.project_id:
        raise RuntimeError("Marketing kits require a project_id.")
    studio._load_project(req.project_id)

    source_text = await _fetch_source_text(task_id, req.source_text, req.source_url, req.brochure_url)
    if not source_text:
        raise RuntimeError("Marketing kits require source_text, source_url, or brochure_url.")

    fallback = _marketing_fallback(source_text, req.product_name, req.target_audiences)
    prompt = (
        "Return strict JSON with keys `product_name`, `hero_headline`, `benefit_bullets`, `hooks`, "
        "`voice_script`, `hero_prompt`, and `comparison_prompt` for this product brief:\n\n"
        f"{source_text[:7000]}"
    )
    plan = await _writer_json(prompt, fallback, max_tokens=1000)
    product_name = _clean_text(str(plan.get("product_name") or fallback["product_name"]))
    hero_headline = _clean_text(str(plan.get("hero_headline") or fallback["hero_headline"]))
    benefit_bullets = plan.get("benefit_bullets")
    if not isinstance(benefit_bullets, list):
        benefit_bullets = fallback["benefit_bullets"]
    benefit_bullets = [_clean_text(str(item)) for item in benefit_bullets if _clean_text(str(item))]
    hooks = plan.get("hooks")
    if not isinstance(hooks, list):
        hooks = fallback["hooks"]
    hooks = [_clean_text(str(item)) for item in hooks if _clean_text(str(item))]
    voice_script = _clean_text(str(plan.get("voice_script") or fallback["voice_script"]))
    hero_prompt = _clean_text(str(plan.get("hero_prompt") or fallback["hero_prompt"]))
    comparison_prompt = _clean_text(str(plan.get("comparison_prompt") or fallback["comparison_prompt"]))

    brief_url = await _write_text_asset(
        task_id,
        "marketing-brief.md",
        "\n".join(
            [
                f"# {product_name} Marketing Kit",
                "",
                "## Hero Headline",
                "",
                hero_headline,
                "",
                "## Benefit Bullets",
                "",
                *[f"- {item}" for item in benefit_bullets],
                "",
                "## Hooks",
                "",
                *[f"- {item}" for item in hooks],
                "",
            ]
        ),
        req.project_id,
        f"{req.label}-brief",
        {"product_name": product_name},
    )

    hero_image_url = await _render_image_from_prompt(task_id, hero_prompt, "marketing-hero.png")
    studio._attach_asset(req.project_id, "image", f"{req.label}-hero", hero_image_url, {"prompt": hero_prompt, "product_name": product_name})

    comparison_image_url = await _render_image_from_prompt(task_id, comparison_prompt, "marketing-comparison.png")
    studio._attach_asset(req.project_id, "image", f"{req.label}-comparison", comparison_image_url, {"prompt": comparison_prompt, "product_name": product_name})

    ad_audio_path = studio._output_path(task_id, "voiceover-ad.mp3")
    ad_audio_url = await _synthesize_voice(voice_script, req.voice, ad_audio_path)
    studio._attach_asset(req.project_id, "audio", f"{req.label}-voiceover", ad_audio_url, {"product_name": product_name})

    teaser = None
    if req.generate_teaser_video:
        teaser = await _run_director_and_wait(hero_prompt, voice_script, req.voice)
        teaser_url = teaser.get("output_url")
        if teaser_url:
            studio._attach_asset(req.project_id, "video", f"{req.label}-teaser", teaser_url, {"product_name": product_name, "workflow": "director"})

    return {
        "task_id": task_id,
        "status": "completed",
        "product_name": product_name,
        "brief_url": brief_url,
        "hero_image_url": hero_image_url,
        "comparison_image_url": comparison_image_url,
        "voiceover_ad_url": ad_audio_url,
        "hooks": hooks,
        "teaser": teaser,
    }


studio.TASK_RUNNERS.update(
    {
        "short_form_pack": _run_short_form_pack_task,
        "episode_pack": _run_episode_pack_task,
        "localization_run": _run_localization_task,
        "meeting_deliverables": _run_meeting_deliverables_task,
        "marketing_kit": _run_marketing_kit_task,
    }
)


@business_router.post("/projects/{project_id}/short-form-packs")
async def create_short_form_pack(project_id: str, request: Request):
    req = await studio._parse_model(request, ShortFormPackRequest)
    payload = studio._model_dump(req)
    payload["project_id"] = project_id
    return studio._queue_task("short_form_pack", "short_form_pack", payload, project_id, req.webhook_url, req.label)


@business_router.post("/projects/{project_id}/episode-packs")
async def create_episode_pack(project_id: str, request: Request):
    req = await studio._parse_model(request, EpisodePackRequest)
    payload = studio._model_dump(req)
    payload["project_id"] = project_id
    return studio._queue_task("episode_pack", "episode_pack", payload, project_id, req.webhook_url, req.label)


@business_router.post("/projects/{project_id}/localization-runs")
async def create_localization_run(project_id: str, request: Request):
    req = await studio._parse_model(request, LocalizationRunRequest)
    payload = studio._model_dump(req)
    payload["project_id"] = project_id
    return studio._queue_task("localization_run", "localization_run", payload, project_id, req.webhook_url, req.label)


@business_router.post("/projects/{project_id}/meeting-deliverables")
async def create_meeting_deliverables(project_id: str, request: Request):
    req = await studio._parse_model(request, MeetingDeliverablesRequest)
    payload = studio._model_dump(req)
    payload["project_id"] = project_id
    return studio._queue_task("meeting_deliverables", "meeting_deliverables", payload, project_id, req.webhook_url, req.label)


@business_router.post("/projects/{project_id}/marketing-kits")
async def create_marketing_kit(project_id: str, request: Request):
    req = await studio._parse_model(request, MarketingKitRequest)
    payload = studio._model_dump(req)
    payload["project_id"] = project_id
    return studio._queue_task("marketing_kit", "marketing_kit", payload, project_id, req.webhook_url, req.label)
