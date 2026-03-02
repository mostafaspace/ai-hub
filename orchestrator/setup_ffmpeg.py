import os
import urllib.request
import zipfile
import shutil

BIN_DIR = os.path.join(os.path.dirname(__file__), "bin")
FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
ZIP_PATH = os.path.join(BIN_DIR, "ffmpeg.zip")
EXTRACT_PATH = os.path.join(BIN_DIR, "ffmpeg_extracted")

def setup_ffmpeg():
    if not os.path.exists(BIN_DIR):
        os.makedirs(BIN_DIR)

    ffmpeg_exe = os.path.join(BIN_DIR, "ffmpeg.exe")
    if os.path.exists(ffmpeg_exe):
        print(f"FFmpeg is already installed at {ffmpeg_exe}")
        return ffmpeg_exe

    print(f"Downloading FFmpeg from {FFMPEG_URL}...")
    try:
        urllib.request.urlretrieve(FFMPEG_URL, ZIP_PATH)
        print("Download complete. Extracting...")
        
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
            
        # The zip contains a top-level folder 'ffmpeg-master-latest-win64-gpl'
        # We need to find the bin/ffmpeg.exe inside it
        for root, dirs, files in os.walk(EXTRACT_PATH):
            if "ffmpeg.exe" in files:
                source_exe = os.path.join(root, "ffmpeg.exe")
                shutil.move(source_exe, ffmpeg_exe)
                print(f"Successfully moved ffmpeg.exe to {ffmpeg_exe}")
                return ffmpeg_exe
                
        # If we reach here, ffmpeg.exe was not found inside the archive
        raise RuntimeError("ffmpeg.exe not found inside the downloaded archive.")
                
    except Exception as e:
        print(f"Failed to setup FFmpeg: {e}")
        return None
    finally:
        # Cleanup
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
        if os.path.exists(EXTRACT_PATH):
            shutil.rmtree(EXTRACT_PATH)

if __name__ == "__main__":
    setup_ffmpeg()
