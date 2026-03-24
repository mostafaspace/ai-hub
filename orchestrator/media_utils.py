import os
import subprocess
import shutil

BIN_DIR = os.path.join(os.path.dirname(__file__), "bin")
FFMPEG_EXE = os.path.join(BIN_DIR, "ffmpeg.exe")
FFPROBE_EXE = os.path.join(BIN_DIR, "ffprobe.exe")

def get_ffmpeg_cmd():
    return FFMPEG_EXE if os.path.exists(FFMPEG_EXE) else shutil.which("ffmpeg")

def get_ffprobe_cmd():
    return FFPROBE_EXE if os.path.exists(FFPROBE_EXE) else shutil.which("ffprobe")

def get_media_duration(path: str) -> float:
    """Returns duration in seconds using ffprobe or ffmpeg fallback."""
    ffprobe_cmd = get_ffprobe_cmd()
    if ffprobe_cmd:
        cmd = [
            ffprobe_cmd,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception:
            pass # Fallback to ffmpeg
            
    ffmpeg_cmd = get_ffmpeg_cmd()
    if not ffmpeg_cmd: return 0.0
    
    # Fallback: parse ffmpeg -i output
    try:
        cmd = [ffmpeg_cmd, "-i", path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        # Duration: 00:00:13.92, ...
        import re
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
        if match:
            h, m, s = map(float, match.groups())
            return h * 3600 + m * 60 + s
    except Exception as e:
        print(f"[Error] Failed to parse duration with ffmpeg: {e}")
    return 0.0

def loop_video_to_duration(video_path: str, target_duration: float, output_path: str) -> bool:
    """Loops video to reach target_duration using stream_loop."""
    ffmpeg_cmd = get_ffmpeg_cmd()
    if not ffmpeg_cmd: return False
    
    # We loop it infinitely and cut at target_duration
    # Added -an to strip any existing audio from the raw video to prevent muxing issues later
    cmd = [
        ffmpeg_cmd,
        "-stream_loop", "-1",
        "-i", video_path,
        "-t", str(target_duration),
        "-an", # No audio in the looped video
        "-c:v", "libx264", 
        "-pix_fmt", "yuv420p",
        "-y",
        output_path
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"[Error] Failed to loop video: {e}")
        return False


def loop_audio_to_duration(audio_path: str, target_duration: float, output_path: str) -> bool:
    """Loops audio to reach target_duration using stream_loop."""
    ffmpeg_cmd = get_ffmpeg_cmd()
    if not ffmpeg_cmd:
        return False

    cmd = [
        ffmpeg_cmd,
        "-stream_loop", "-1",
        "-i", audio_path,
        "-t", str(target_duration),
        "-c:a", "aac",
        "-b:a", "192k",
        "-y",
        output_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"[Error] Failed to loop audio: {e}")
        return False

def mix_audio_files(voice_path: str, music_path: str, output_path: str, music_volume: float = 0.2) -> bool:
    """Mixes voiceover and music with volume adjustment."""
    ffmpeg_cmd = get_ffmpeg_cmd()
    if not ffmpeg_cmd: return False
    
    # [0:a] is voice, [1:a] is music
    # Use duration=longest so short narration doesn't truncate the score bed.
    filter_complex = (
        f"[0:a]volume=1.2[v];"
        f"[1:a]volume={music_volume}[m];"
        "[v][m]amix=inputs=2:duration=longest:dropout_transition=2[a]"
    )
    cmd = [
        ffmpeg_cmd,
        "-i", voice_path,
        "-i", music_path,
        "-filter_complex", filter_complex,
        "-map", "[a]",
        "-c:a", "aac",
        "-b:a", "192k",
        "-y",
        output_path
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"[Error] Failed to mix audio: {e}")
        return False

def mux_video_and_audio(video_path: str, audio_path: str, output_path: str) -> bool:
    """
    Uses FFmpeg to mux a video file and an audio file together.
    Forcefully takes video from 1st input and audio from 2nd input.
    """
    ffmpeg_cmd = get_ffmpeg_cmd()
    if ffmpeg_cmd is None:
        print("[Error] FFmpeg not found.")
        return False
        
    # Command: ffmpeg -i video.mp4 -i audio.wav -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest output.mp4 -y
    cmd = [
        ffmpeg_cmd,
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",         # Take video from 1st input
        "-map", "1:a:0",         # Take audio from 2nd input
        "-c:v", "copy",          
        "-c:a", "aac",           
        "-b:a", "192k",          
        "-shortest",             
        "-y",                    
        output_path
    ]
    
    print(f"Muxing media: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except Exception as e:
        print(f"[Error] FFmpeg muxing failed: {e}")
        return False
