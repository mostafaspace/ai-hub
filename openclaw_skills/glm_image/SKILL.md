---
name: glm_image
description: API documentation for the GLM-Image Vision server, enabling text-to-image and image-to-image generation.
---

# GLM-Image Vision API Skill

This skill provides the API specification for the GLM-Image Vision server. Agents can use this information to construct HTTP requests to generate or edit images.

## Server Details

-   **Base URL**: `http://192.168.1.26:8003`
-   **OpenAPI Spec**: `openapi.yaml` (in this directory)

> [!IMPORTANT]
> **First Request**: The model (~32GB) downloads on first use. Subsequent startups use the cached model.
> **Timeouts**: Image generation takes **2-10 minutes** depending on GPU. Set HTTP timeout to at least 600s.
> **VRAM**: Requires 80GB+ VRAM, or uses CPU offload on smaller GPUs (slower inference).

## Capabilities

1.  **Text-to-Image**: Generate high-fidelity images from text prompts.
2.  **Image-to-Image**: Edit, style-transfer, or transform existing images with text guidance.

## API Endpoints

### 1. Generate Image (Text-to-Image)
**POST** `/v1/images/generations`

Generates an image from a text prompt (OpenAI-compatible).

**JSON Body:**
```json
{
  "prompt": "A beautiful sunset over a mountain lake, photorealistic",
  "size": "1024x1024",
  "n": 1,
  "response_format": "url",
  "num_inference_steps": 50,
  "guidance_scale": 1.5
}
```
-   `prompt`: Text description of the image to generate. Enclose text to render in quotation marks.
-   `size`: `WIDTHxHEIGHT` — dimensions must be divisible by 32.
-   `response_format`: `url` (returns a download URL) or `b64_json` (returns base64-encoded PNG).
-   `num_inference_steps`: Higher = better quality, slower. Default: 50.
-   `guidance_scale`: Recommended: 1.5.

**Response:**
```json
{
  "created": 1234567890,
  "data": [
    {
      "url": "http://192.168.1.26:8003/files/abc123.png",
      "revised_prompt": "A beautiful sunset..."
    }
  ]
}
```

### 2. Edit Image (Image-to-Image)
**POST** `/v1/images/edits`

**Content-Type**: `multipart/form-data`

Edits or transforms an input image based on a text prompt.

**Fields:**
-   `image`: The input image file (PNG/JPG).
-   `prompt`: Text instruction for how to modify the image.
-   `size`: Output size (`WIDTHxHEIGHT`). Defaults to input image size.
-   `response_format`: `url` (default) or `b64_json`.
-   `num_inference_steps`: Default 50.
-   `guidance_scale`: Default 1.5.

### 3. Health Check
**GET** `/health`

Returns server status and whether the model is currently loaded.

### 4. List Models
**GET** `/v1/models`

Returns model metadata and supported capabilities.

### 5. Manual Unload
**POST** `/v1/internal/unload`

Forces the model to unload from VRAM immediately.

## Example Workflow (Agent)

To generate an image:

1.  Construct the JSON payload with `prompt` and desired `size`.
2.  Send a **POST** request to `http://192.168.1.26:8003/v1/images/generations`.
3.  Parse the response JSON to get the image `url` or `b64_json`.
4.  If using `url`, download the image from the returned URL.

## Troubleshooting

### Timeout / Slow Response
-   **Normal**: GLM-Image generation takes 2-10 minutes per image. Set timeout to 600s+.
-   **First run**: Model download (~32GB) can take 30-60 minutes.

### Out of Memory
-   CPU offload is enabled by default. If still OOM, try reducing image size (e.g., `512x512`).
