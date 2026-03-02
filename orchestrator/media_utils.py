import os
import subprocess
import shutil

BIN_DIR = os.path.join(os.path.dirname(__file__), "bin")
FFMPEG_EXE = os.path.join(BIN_DIR, "ffmpeg.exe")

def mux_video_and_audio(video_path: str, audio_path: str, output_path: str) -> bool:
    """
    Uses FFmpeg to mux a video file and an audio file together.
    The output video length will be the shortest of the two streams to prevent looping/silence.
    """
    # Check for local bin first, then system path
    ffmpeg_cmd = FFMPEG_EXE if os.path.exists(FFMPEG_EXE) else shutil.which("ffmpeg")
    
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
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"[Error] Exception during FFmpeg execution: {e}")
        return False
