import os
import io
import uuid
import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

# Import media_utils for FFmpeg muxing
from orchestrator.media_utils import mux_video_and_audio

utils_router = APIRouter(prefix="/v1/utils", tags=["Utility Services"])

# Working directory for utility operations
TEMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "tmp_utils"))
os.makedirs(TEMP_DIR, exist_ok=True)


class MuxUrlRequest(BaseModel):
    video_url: str
    audio_url: str


@utils_router.post("/ffmpeg/mux/urls")
async def mux_from_urls(req: MuxUrlRequest):
    """
    Downloads a video and an audio file from the provided URLs, 
    muxes them using FFmpeg, and returns the final MP4 file.
    """
    task_id = str(uuid.uuid4())
    task_dir = os.path.join(TEMP_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    video_path = os.path.join(task_dir, "input_video.mp4")
    audio_path = os.path.join(task_dir, "input_audio.wav")
    output_path = os.path.join(task_dir, "output.mp4")

    # High timeout since media files can be large
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            # Download video
            print(f"[FFmpeg Utils] Downloading video from {req.video_url}...")
            v_resp = await client.get(req.video_url)
            if v_resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Failed to download video: {v_resp.status_code}")
            with open(video_path, 'wb') as f:
                f.write(v_resp.content)

            # Download audio
            print(f"[FFmpeg Utils] Downloading audio from {req.audio_url}...")
            a_resp = await client.get(req.audio_url)
            if a_resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Failed to download audio: {a_resp.status_code}")
            with open(audio_path, 'wb') as f:
                f.write(a_resp.content)
            
            # Mux
            print(f"[FFmpeg Utils] Muxing media for task {task_id}...")
            success = mux_video_and_audio(video_path, audio_path, output_path)
            
            if not success:
                raise HTTPException(status_code=500, detail="FFmpeg muxing failed.")
                
            return FileResponse(path=output_path, media_type="video/mp4", filename=f"muxed_{task_id}.mp4")

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error processing media: {str(e)}")


@utils_router.post("/ffmpeg/mux/files")
async def mux_from_files(
    video_file: UploadFile = File(...),
    audio_file: UploadFile = File(...)
):
    """
    Accepts direct file uploads for video and audio, 
    muxes them using FFmpeg, and returns the final MP4.
    """
    task_id = str(uuid.uuid4())
    task_dir = os.path.join(TEMP_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    # Simple extraction of extensions, assuming standard types if none provided
    v_ext = os.path.splitext(video_file.filename)[1] or ".mp4"
    a_ext = os.path.splitext(audio_file.filename)[1] or ".wav"

    video_path = os.path.join(task_dir, f"input_video{v_ext}")
    audio_path = os.path.join(task_dir, f"input_audio{a_ext}")
    output_path = os.path.join(task_dir, "output.mp4")

    try:
        # Write uploads to disk
        content_v = await video_file.read()
        with open(video_path, "wb") as f:
            f.write(content_v)
            
        content_a = await audio_file.read()
        with open(audio_path, "wb") as f:
            f.write(content_a)
            
        print(f"[FFmpeg Utils] Muxing media for task {task_id} from uploaded files...")
        success = mux_video_and_audio(video_path, audio_path, output_path)
        
        if not success:
            raise HTTPException(status_code=500, detail="FFmpeg muxing failed.")
            
        return FileResponse(path=output_path, media_type="video/mp4", filename=f"muxed_{task_id}.mp4")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing media files: {str(e)}")


class ImageOverlayRequest(BaseModel):
    image_url: str
    top_text: Optional[str] = None
    bottom_text: Optional[str] = None

@utils_router.post("/image/overlay/urls")
async def overlay_text_from_url(req: ImageOverlayRequest):
    """
    Downloads an image, overlays classic impact-style text (top and/or bottom), 
    and returns the processed image.
    """
    # Pillow basic text overly
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(req.image_url)
            if resp.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to fetch image.")
                
            img_bytes = resp.content
            
        # Process image
        return process_image_overlay(img_bytes, req.top_text, req.bottom_text)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

@utils_router.post("/image/overlay/file")
async def overlay_text_from_file(
    image_file: UploadFile = File(...),
    top_text: str = Form(None),
    bottom_text: str = Form(None)
):
    """
    Accepts an image file upload, overlays classic impact-style text (top and/or bottom), 
    and returns the processed image.
    """
    try:
        img_bytes = await image_file.read()
        return process_image_overlay(img_bytes, top_text, bottom_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")
        

def process_image_overlay(img_bytes: bytes, top_text: Optional[str], bottom_text: Optional[str]) -> Response:
    """Helper to draw text on image utilizing Pillow."""
    try:
        image = Image.open(io.BytesIO(img_bytes))
        draw = ImageDraw.Draw(image)
        
        # Calculate a dynamic font size based on image width
        image_w, image_h = image.size
        # Try to load a standard truetype font. 
        # On Windows 'impact.ttf' usually exists. Fallback to default if not.
        try:
            # Roughly scale font size to 10% of image height
            font_size = int(image_h * 0.10)
            font = ImageFont.truetype("impact.ttf", font_size)
        except IOError:
            # Fallback if impact.ttf isn't found
            font = ImageFont.load_default()
            font_size = 20 # fixed small size for default
        
        # Helper function to draw text with a black outline
        def draw_text_with_outline(text, position):
            if not text:
                return
            text = text.upper()
            
            # Using textbbox to get dimensions of the text
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            # Center the text horizontally
            x = (image_w - text_w) / 2
            y = position
            
            # Stroke / Outline
            outline_color = "black"
            fill_color = "white"
            stroke_width = max(2, int(font_size * 0.05))
            
            # Draw stroke by drawing text slightly offset in all directions
            for adj_x in range(-stroke_width, stroke_width+1):
                for adj_y in range(-stroke_width, stroke_width+1):
                    draw.text((x + adj_x, y + adj_y), text, font=font, fill=outline_color)
            
            # Draw main text
            draw.text((x, y), text, font=font, fill=fill_color)

        # Draw top text
        if top_text:
            draw_text_with_outline(top_text, 10) # 10px margin from top
            
        # Draw bottom text
        if bottom_text:
            # bottom margin approx font_size + 20
            draw_text_with_outline(bottom_text, image_h - font_size - 20)

        # Save to buffer
        img_io = io.BytesIO()
        # Preserve original format or default to JPEG
        img_format = image.format if image.format else "JPEG"
        image.save(img_io, format=img_format)
        img_io.seek(0)
        
        # Determine mime type
        media_type = f"image/{img_format.lower()}"
        if media_type == "image/jpg": media_type = "image/jpeg"
        
        return Response(content=img_io.getvalue(), media_type=media_type)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
