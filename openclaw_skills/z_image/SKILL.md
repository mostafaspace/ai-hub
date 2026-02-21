---
name: z_image
description: API documentation for the Z-Image Vision server, enabling text-to-image pipelines.
---

# Z-Image Vision API Skill

This skill provides the API specification for the Z-Image Vision server. Agents can use this information to construct HTTP requests to generate images.

## Server Details

-   **Base URL**: `http://192.168.1.26:8003`
-   **OpenAPI Spec**: `openapi.yaml` (in this directory)

> [!IMPORTANT]
> **Lazy Loading**: The model loads on the first image request (not at startup). The first generation takes **2–5 minutes** (model load + inference). Subsequent requests are much faster (~30–60s).
> **Timeouts**: Set HTTP timeout to at least **600 seconds** (10 minutes).
> **VRAM**: Optimized for **Multi-GPU**. CPU offload disabled for maximum speed.

## Pre-flight & Progress Checking

- **Pre-flight:** Call `GET /health` or `GET /v1/internal/status` to check if the model is alive. 
- **Progress Polling:** The `POST /v1/images/generations` request is strictly **synchronous**. It holds the connection open until the image finishes. 
    - *However*, you can run a parallel thread to poll `GET /v1/internal/status`. It will return `{"status": "generating"}` while the server is busy.
    - Do **NOT** assume the POST request failed just because it takes a long time. Set your timeout to `600s` and WAIT for the JSON response.

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
  "negative_prompt": "blurry, low quality",
  "size": "1024x1024",
  "n": 1,
  "response_format": "b64_json",
  "num_inference_steps": 50,
  "guidance_scale": 4.0,
  "cfg_normalization": false
}
```
-   `prompt`: Text description of the image to generate. Enclose text to render in quotation marks.
-   `size`: `WIDTHxHEIGHT` — dimensions must be divisible by 32.
-   `response_format`: `b64_json` (recommended — returns base64-encoded PNG inline) or `url` (returns a download URL).
-   `num_inference_steps`: Higher = better quality, slower. Default: 50.
-   `guidance_scale`: Recommended: 4.0.
-   `cfg_normalization`: Set to False for standard generation.
-   `negative_prompt`: Things to avoid in the generation.

**Response:**
```json
{
  "created": 1234567890,
  "data": [
    {
      "b64_json": "<base64-encoded PNG data>",
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
**POST** or **GET** `/v1/internal/unload`

Forces the model to unload from VRAM immediately. Both HTTP methods are supported.

## Example Workflow (Agent)

To generate an image:

1.  **Pre-flight**: Send **GET** to `http://192.168.1.26:8003/health` to check if the model is currently loaded.
2.  Construct the JSON payload with `prompt` and desired `size`.
3.  Send a **POST** request to `http://192.168.1.26:8003/v1/images/generations` with **timeout=600**.
4.  Parse the response JSON to get the image `b64_json` or `url`.
5.  If using `url`, download the image from the returned URL.

## Error Handling

-   **500 (Model loading failed)**: The model failed to load. Wait 10 seconds and try again. The API will retry model loading on the next request.
-   **Timeout**: Normal for first request. The model (~32GB) takes 2–5 minutes to load on first use.

## Troubleshooting

### Timeout / Slow Response
-   **Normal**: Z-Image generation takes 1-5 minutes per image. Set timeout to 600s+.
-   **First run**: Model download (~16GB) can take 30-60 minutes.

### Out of Memory
-   CPU offload is enabled by default. If still OOM, try reducing image size (e.g., `512x512`).
