import os
import re
import shutil
import subprocess
import tempfile
from typing import Iterable, Optional


BIN_DIR = os.path.join(os.path.dirname(__file__), "bin")
FFMPEG_EXE = os.path.join(BIN_DIR, "ffmpeg.exe")

FORMAT_PROFILES = {
    "youtube_short": {
        "label": "YouTube Short",
        "kind": "video",
        "container": "mp4",
        "video_codec": "libx264",
        "audio_codec": "aac",
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "video_bitrate": "6M",
        "audio_bitrate": "192k",
    },
    "tiktok_vertical": {
        "label": "TikTok Vertical",
        "kind": "video",
        "container": "mp4",
        "video_codec": "libx264",
        "audio_codec": "aac",
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "video_bitrate": "8M",
        "audio_bitrate": "192k",
    },
    "discord_clip": {
        "label": "Discord Clip",
        "kind": "video",
        "container": "mp4",
        "video_codec": "libx264",
        "audio_codec": "aac",
        "width": 1280,
        "height": 720,
        "fps": 30,
        "video_bitrate": "4M",
        "audio_bitrate": "160k",
    },
    "podcast_mp3": {
        "label": "Podcast MP3",
        "kind": "audio",
        "container": "mp3",
        "audio_codec": "libmp3lame",
        "audio_bitrate": "192k",
        "sample_rate": 44100,
    },
    "whatsapp_voice": {
        "label": "WhatsApp Voice",
        "kind": "audio",
        "container": "ogg",
        "audio_codec": "libopus",
        "audio_bitrate": "48k",
        "sample_rate": 48000,
    },
}


def _ffmpeg_cmd() -> Optional[str]:
    if os.path.exists(FFMPEG_EXE):
        return FFMPEG_EXE
    return shutil.which("ffmpeg")


def _run_ffmpeg(args: list[str]) -> tuple[bool, str]:
    ffmpeg_cmd = _ffmpeg_cmd()
    if ffmpeg_cmd is None:
        return False, "FFmpeg not found locally or on PATH."

    cmd = [ffmpeg_cmd, *args]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, result.stdout or result.stderr
    except subprocess.CalledProcessError as exc:
        return False, exc.stderr or exc.stdout or str(exc)
    except Exception as exc:
        return False, str(exc)


def ffmpeg_available() -> bool:
    return _ffmpeg_cmd() is not None


def resolve_format_profile(profile_name: str) -> dict:
    if profile_name not in FORMAT_PROFILES:
        raise ValueError(f"Unknown output profile: {profile_name}")
    return dict(FORMAT_PROFILES[profile_name])


def detect_duration(input_path: str) -> float:
    ffmpeg_cmd = _ffmpeg_cmd()
    if ffmpeg_cmd is None or not os.path.exists(input_path):
        return 0.0

    try:
        result = subprocess.run(
            [ffmpeg_cmd, "-i", input_path],
            capture_output=True,
            text=True,
            check=False,
        )
        stderr = result.stderr or ""
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
        if not match:
            return 0.0
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        return (hours * 3600) + (minutes * 60) + seconds
    except Exception:
        return 0.0


def extract_audio_for_transcription(input_path: str, output_path: str) -> tuple[bool, str]:
    return _run_ffmpeg(
        [
            "-y",
            "-i",
            input_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            output_path,
        ]
    )


def extract_thumbnail(input_path: str, output_path: str, timestamp_sec: float = 0.0) -> tuple[bool, str]:
    return _run_ffmpeg(
        [
            "-y",
            "-ss",
            f"{max(timestamp_sec, 0.0):.3f}",
            "-i",
            input_path,
            "-frames:v",
            "1",
            output_path,
        ]
    )


def create_contact_sheet(
    input_path: str,
    output_path: str,
    columns: int = 4,
    rows: int = 2,
    thumb_width: int = 320,
) -> tuple[bool, str]:
    columns = max(columns, 1)
    rows = max(rows, 1)
    thumb_width = max(thumb_width, 120)
    frames_needed = columns * rows
    duration = detect_duration(input_path)
    interval = max(duration / frames_needed, 0.5) if duration else 1.0
    filter_value = f"fps=1/{interval:.3f},scale={thumb_width}:-1,tile={columns}x{rows}"
    return _run_ffmpeg(
        [
            "-y",
            "-i",
            input_path,
            "-vf",
            filter_value,
            "-frames:v",
            "1",
            output_path,
        ]
    )


def transcode_media(input_path: str, output_path: str, profile_name: str) -> tuple[bool, str]:
    profile = resolve_format_profile(profile_name)
    args = ["-y", "-i", input_path]
    if profile["kind"] == "video":
        filter_value = (
            f"scale={profile['width']}:{profile['height']}:force_original_aspect_ratio=decrease,"
            f"pad={profile['width']}:{profile['height']}:(ow-iw)/2:(oh-ih)/2"
        )
        args.extend(
            [
                "-vf",
                filter_value,
                "-r",
                str(profile["fps"]),
                "-c:v",
                profile["video_codec"],
                "-b:v",
                profile["video_bitrate"],
                "-c:a",
                profile["audio_codec"],
                "-b:a",
                profile["audio_bitrate"],
                output_path,
            ]
        )
    else:
        args.extend(
            [
                "-vn",
                "-c:a",
                profile["audio_codec"],
                "-b:a",
                profile["audio_bitrate"],
                "-ar",
                str(profile["sample_rate"]),
                output_path,
            ]
        )
    return _run_ffmpeg(args)


def normalize_video_clip(
    input_path: str,
    output_path: str,
    profile_name: str,
    trim_in_sec: float = 0.0,
    duration_sec: Optional[float] = None,
) -> tuple[bool, str]:
    profile = resolve_format_profile(profile_name)
    if profile["kind"] != "video":
        return False, f"Profile {profile_name} is not a video profile."

    args = ["-y"]
    if trim_in_sec and trim_in_sec > 0:
        args.extend(["-ss", f"{trim_in_sec:.3f}"])
    args.extend(["-i", input_path])
    if duration_sec and duration_sec > 0:
        args.extend(["-t", f"{duration_sec:.3f}"])

    filter_value = (
        f"scale={profile['width']}:{profile['height']}:force_original_aspect_ratio=decrease,"
        f"pad={profile['width']}:{profile['height']}:(ow-iw)/2:(oh-ih)/2,fps={profile['fps']}"
    )
    args.extend(
        [
            "-vf",
            filter_value,
            "-an",
            "-c:v",
            profile["video_codec"],
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
    )
    return _run_ffmpeg(args)


def concat_video_clips(clip_paths: list[str], output_path: str) -> tuple[bool, str]:
    if not clip_paths:
        return False, "No clips were provided for concatenation."

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        list_path = handle.name
        for clip_path in clip_paths:
            safe_path = clip_path.replace("'", "'\\''")
            handle.write(f"file '{safe_path}'\n")

    try:
        return _run_ffmpeg(
            [
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-c",
                "copy",
                output_path,
            ]
        )
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass


def attach_audio_bed(
    video_path: str,
    audio_path: str,
    output_path: str,
    start_sec: float = 0.0,
    volume: float = 1.0,
) -> tuple[bool, str]:
    delay_ms = max(int(start_sec * 1000), 0)
    audio_filter = f"[1:a]volume={max(volume, 0.0)},adelay={delay_ms}|{delay_ms}[aud]"
    return _run_ffmpeg(
        [
            "-y",
            "-i",
            video_path,
            "-stream_loop",
            "-1",
            "-i",
            audio_path,
            "-filter_complex",
            audio_filter,
            "-map",
            "0:v:0",
            "-map",
            "[aud]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            output_path,
        ]
    )


def _escape_subtitles_path(path: str) -> str:
    return path.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def burn_subtitles_to_video(
    input_path: str,
    subtitles_path: str,
    output_path: str,
    profile_name: str = "youtube_short",
) -> tuple[bool, str]:
    profile = resolve_format_profile(profile_name)
    if profile["kind"] != "video":
        return False, f"Profile {profile_name} is not a video profile."

    subtitle_filter = f"subtitles='{_escape_subtitles_path(subtitles_path)}'"
    scale_filter = (
        f"scale={profile['width']}:{profile['height']}:force_original_aspect_ratio=decrease,"
        f"pad={profile['width']}:{profile['height']}:(ow-iw)/2:(oh-ih)/2"
    )
    filter_value = f"{scale_filter},{subtitle_filter}"
    return _run_ffmpeg(
        [
            "-y",
            "-i",
            input_path,
            "-vf",
            filter_value,
            "-r",
            str(profile["fps"]),
            "-c:v",
            profile["video_codec"],
            "-b:v",
            profile["video_bitrate"],
            "-c:a",
            profile["audio_codec"],
            "-b:a",
            profile["audio_bitrate"],
            output_path,
        ]
    )


def split_subtitle_text(text: str, max_chars: int = 70) -> list[str]:
    normalized = " ".join((text or "").split())
    if not normalized:
        return []

    sentence_candidates = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []

    for sentence in sentence_candidates:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            chunks.append(sentence)
            continue

        words = sentence.split()
        current: list[str] = []
        current_len = 0
        for word in words:
            next_len = current_len + len(word) + (1 if current else 0)
            if current and next_len > max_chars:
                chunks.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len = next_len
        if current:
            chunks.append(" ".join(current))

    return chunks or [normalized]


def _format_srt_timestamp(total_seconds: float) -> str:
    total_ms = max(int(total_seconds * 1000), 0)
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    seconds = (total_ms % 60_000) // 1000
    milliseconds = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def write_approximate_srt(text: str, duration_seconds: float, output_path: str) -> None:
    chunks = split_subtitle_text(text)
    if not chunks:
        raise ValueError("Cannot generate subtitles from empty transcript.")

    if duration_seconds <= 0:
        duration_seconds = max(len(chunks) * 2.0, 2.0)

    total_weight = sum(max(len(chunk), 1) for chunk in chunks)
    current_time = 0.0
    lines: list[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        weight = max(len(chunk), 1) / total_weight
        slot_duration = max(duration_seconds * weight, 1.2)
        start_time = current_time
        end_time = min(duration_seconds, start_time + slot_duration)
        if idx == len(chunks):
            end_time = max(end_time, duration_seconds)
        lines.extend(
            [
                str(idx),
                f"{_format_srt_timestamp(start_time)} --> {_format_srt_timestamp(end_time)}",
                chunk,
                "",
            ]
        )
        current_time = end_time

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).strip() + "\n")


def build_timeline_plan(
    video_clips: Iterable[dict],
    audio_tracks: Iterable[dict],
    subtitle_tracks: Iterable[dict],
    format_profile: str,
) -> dict:
    clip_items = sorted(video_clips, key=lambda item: float(item.get("start_sec", 0.0)))
    audio_items = sorted(audio_tracks, key=lambda item: float(item.get("start_sec", 0.0)))
    subtitle_items = list(subtitle_tracks)

    total_duration = 0.0
    for item in list(clip_items) + list(audio_items):
        start_sec = float(item.get("start_sec", 0.0))
        duration_sec = float(item.get("duration_sec") or 0.0)
        total_duration = max(total_duration, start_sec + max(duration_sec, 0.0))

    return {
        "format_profile": format_profile,
        "estimated_duration_sec": round(total_duration, 3),
        "video_clips": clip_items,
        "audio_tracks": audio_items,
        "subtitle_tracks": subtitle_items,
        "render_ready": bool(clip_items),
        "supported_render_scope": {
            "video_clips": "sequential concat only",
            "audio_tracks": "first track may be mixed as a looping bed",
            "subtitle_tracks": "first track may be burned into a copied render",
        },
        "notes": [
            "Timeline storage is non-destructive and additive.",
            "Rendering creates a new output and never mutates source assets.",
        ],
    }