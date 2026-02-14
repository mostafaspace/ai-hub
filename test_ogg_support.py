"""
Test script to verify OGG (WhatsApp Voice Note) support in Qwen3-TTS.
"""
import requests
import sys
import io
import os
import math
import struct
import wave
import wave
try:
    import imageio_ffmpeg
    from pydub import AudioSegment

    # Configure pydub to use local ffmpeg if available
    # Note: The instruction specified "d:/antigravity/ffmpeg.exe" but the provided
    # code snippet used os.path.abspath(os.path.join(os.path.dirname(__file__), "ffmpeg.exe")).
    # I will use the explicit path from the instruction.
    explicit_local_ffmpeg_path = "d:/antigravity/ffmpeg.exe"

    if os.path.exists(explicit_local_ffmpeg_path):
        ffmpeg_to_use = explicit_local_ffmpeg_path
    else:
        ffmpeg_to_use = imageio_ffmpeg.get_ffmpeg_exe()
    
    AudioSegment.converter = ffmpeg_to_use
    os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_to_use)
    
    print(f"Using ffmpeg: {AudioSegment.converter}")

except ImportError:
    print("Please install pydub and imageio_ffmpeg: pip install pydub imageio-ffmpeg")
    sys.exit(1)

# Configuration
DEVICE_IP = "127.0.0.1" # Using loopback for speed/reliability in local test
TTS_URL = f"http://{DEVICE_IP}:8000/v1/audio/voice_clone"

def create_test_ogg(filename="test_note.ogg"):
    """Create a dummy OGG file using pydub."""
    print(f"Creating dummy OGG file: {filename}...")
    
    # Create a simple sine wave tone
    sample_rate = 24000
    duration_sec = 2.0
    frequency = 440.0
    
    # Generate PCM data
    samples = []
    for i in range(int(sample_rate * duration_sec)):
        value = int(32767.0 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
        samples.append(value)
    
    # Pack to bytes (16-bit mono)
    audio_data = struct.pack('<' + 'h' * len(samples), *samples)
    
    # Generate a 1-second sine wave
    from pydub.generators import Sine
    sine_wave = Sine(1000).to_audio_segment(duration=1000)
    song = AudioSegment.from_mono_audiosegments(sine_wave, sine_wave)
    
    print("Exporting OGG...")
    song.export("test_note.ogg", format="ogg")
    
    # Verify file generation
    if not os.path.exists("test_note.ogg") or os.path.getsize("test_note.ogg") == 0:
        print("Warning: pydub export produced empty file. Trying direct ffmpeg generation...")
        subprocess.run([
            AudioSegment.converter, "-y", "-f", "lavfi", "-i", "sine=f=440:d=1", "-c:a", "libvorbis", "test_note.ogg"
        ], check=True)
        
    print(f"Generated test_note.ogg size: {os.path.getsize('test_note.ogg')} bytes")
    return filename

def test_ogg_upload():
    filename = "test_note.ogg"
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        create_test_ogg(filename)
        
    print(f"Uploading {filename} to {TTS_URL}...")
    
    try:
        with open(filename, "rb") as f:
            files = {
                "ref_audio": (filename, f, "audio/ogg")
            }
            data = {
                "text": "This is a test verifying WhatsApp voice note support.",
                "ref_text": "Just a simple tone for testing reference audio.",
                "language": "Auto",
                "response_format": "mp3"
            }
            
            response = requests.post(TTS_URL, files=files, data=data, timeout=120)
            
        if response.status_code == 200:
            print("[SUCCESS] Server accepted OGG file and returned audio!")
            output_file = "test_ogg_result.mp3"
            with open(output_file, "wb") as f_out:
                f_out.write(response.content)
            print(f"Saved result to {output_file} ({len(response.content)} bytes)")
            return True
        else:
            print(f"[FAILURE] Server returned {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

if __name__ == "__main__":
    if test_ogg_upload():
        print("OGG Support Verification: PASSED")
        sys.exit(0)
    else:
        print("OGG Support Verification: FAILED")
        sys.exit(1)
