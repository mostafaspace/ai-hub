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
    """Returns duration in seconds using ffprobe."""
    cmd = [
        get_ffprobe_cmd(),
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"[Error] Failed to get duration for {path}: {e}")
        return 0.0

def loop_video_to_duration(video_path: str, target_duration: float, output_path: str) -> bool:
    """Loops video to reach target_duration using stream_loop."""
    ffmpeg_cmd = get_ffmpeg_cmd()
    if not ffmpeg_cmd: return False
    
    # We loop it infinitely and cut at target_duration
    cmd = [
        ffmpeg_cmd,
        "-stream_loop", "-1",
        "-i", video_path,
        "-t", str(target_duration),
        "-c:v", "libx264", # Re-encode to ensure smooth cuts/looping
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

def mix_audio_files(voice_path: str, music_path: str, output_path: str, music_volume: float = 0.2) -> bool:
    """Mixes voiceover and music with volume adjustment."""
    ffmpeg_cmd = get_ffmpeg_cmd()
    if not ffmpeg_cmd: return False
    
    # [0:a] is voice, [1:a] is music
    # amix duration=first means finish when the first stream (voice) ends
    filter_complex = f"[0:a]volume=1.2[v];[1:a]volume={music_volume}[m];[v][m]amix=inputs=2:duration=first[a]"
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
    The output video length will be the shortest of the two streams to prevent looping/silence.
    """
    ffmpeg_cmd = get_ffmpeg_cmd()
    if ffmpeg_cmd is None:
        print("[Error] FFmpeg not found locally or on system PATH. Cannot mux audio and video.")
        return False
        
    if not os.path.exists(video_path):
        print(f"[Error] Video input not found: {video_path}")
        return False
        
    if not os.path.exists(audio_path):
        print(f"[Error] Audio input not found: {audio_path}")
        return False
        
    # Command: ffmpeg -i video.mp4 -i audio.wav -c:v copy -c:a aac -shortest output.mp4 -y
    cmd = [
        ffmpeg_cmd,
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",          # Copy video stream without re-encoding
        "-c:a", "aac",           # Encode audio to AAC for wide mp4 compatibility
        "-b:a", "192k",          # Audio bitrate
        "-shortest",             # Finish encoding when the shortest input stream ends
        "-y",                    # Overwrite output file
        output_path
    ]
    
    print(f"Muxing media: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Muxing completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Error] FFmpeg muxing failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"[Error] Exception during FFmpeg execution: {e}")
        return False
