import os
import shutil
import subprocess
from typing import Optional

from orchestrator.studio_media import resolve_format_profile


BIN_DIR = os.path.join(os.path.dirname(__file__), "bin")
FFMPEG_EXE = os.path.join(BIN_DIR, "ffmpeg.exe")


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


def _escape_subtitles_path(path: str) -> str:
    return path.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def render_video_clip(
    input_path: str,
    output_path: str,
    profile_name: str,
    trim_in_sec: float = 0.0,
    duration_sec: Optional[float] = None,
    subtitles_path: Optional[str] = None,
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

    scale_filter = (
        f"scale={profile['width']}:{profile['height']}:force_original_aspect_ratio=decrease,"
        f"pad={profile['width']}:{profile['height']}:(ow-iw)/2:(oh-ih)/2,fps={profile['fps']}"
    )
    filter_value = scale_filter
    if subtitles_path:
        filter_value = f"{scale_filter},subtitles='{_escape_subtitles_path(subtitles_path)}'"

    args.extend(
        [
            "-vf",
            filter_value,
            "-c:v",
            profile["video_codec"],
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            profile["video_bitrate"],
            "-c:a",
            profile["audio_codec"],
            "-b:a",
            profile["audio_bitrate"],
            "-ar",
            "48000",
            output_path,
        ]
    )
    return _run_ffmpeg(args)


def render_image_hold_clip(
    input_path: str,
    output_path: str,
    profile_name: str,
    duration_sec: float,
) -> tuple[bool, str]:
    profile = resolve_format_profile(profile_name)
    if profile["kind"] != "video":
        return False, f"Profile {profile_name} is not a video profile."

    duration_sec = max(float(duration_sec or 0.0), 1.0)
    filter_value = (
        f"scale={profile['width']}:{profile['height']}:force_original_aspect_ratio=decrease,"
        f"pad={profile['width']}:{profile['height']}:(ow-iw)/2:(oh-ih)/2,fps={profile['fps']}"
    )
    return _run_ffmpeg(
        [
            "-y",
            "-loop",
            "1",
            "-i",
            input_path,
            "-t",
            f"{duration_sec:.3f}",
            "-vf",
            filter_value,
            "-c:v",
            profile["video_codec"],
            "-pix_fmt",
            "yuv420p",
            "-an",
            output_path,
        ]
    )


def replace_audio_track(video_path: str, audio_path: str, output_path: str) -> tuple[bool, str]:
    return _run_ffmpeg(
        [
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            output_path,
        ]
    )
