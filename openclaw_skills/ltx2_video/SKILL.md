---
name: ltx2_video
description: Agent-facing API guide for the LTX-2.3 video server. Covers async text-to-video and image-to-video generation, polling, outputs, and prompting guidance for cleaner local results.
---

# LTX-2.3 Video Generation API

## Server Details

- **Base URL**: `http://192.168.1.26:8004`
- **OpenAPI Spec**: `openapi.yaml`

> [!IMPORTANT]
> Use the async endpoints. Sync endpoints are legacy and can hold the connection open for minutes.

## Endpoints

| Purpose | Method | Endpoint | Content-Type |
|---|---|---|---|
| Async text-to-video | `POST` | `/v1/video/async_t2v` | `application/json` |
| Async image-to-video | `POST` | `/v1/video/async_i2v` | `multipart/form-data` |
| Poll task | `GET` | `/v1/video/tasks/{task_id}` | - |
| Download output | `GET` | `/outputs/{filename}` | - |
| Health | `GET` | `/health` | - |
| Manual unload | `POST` or `GET` | `/v1/internal/unload` | - |

## Async Text-to-Video

### Submit
**POST** `/v1/video/async_t2v`

```json
{
  "prompt": "A bright cinematic daylight shot of a lone explorer standing on a snowy mountain summit as wind moves the coat slightly and the camera slowly pushes in.",
  "height": 576,
  "width": 1024,
  "num_frames": 25,
  "frame_rate": 24.0,
  "num_inference_steps": 20,
  "seed": 42,
  "negative_prompt": "blurry, noisy, grainy, low detail, dark, flicker, jitter, deformed face, warped hands, extra limbs, text, watermark, AI artifacts",
  "enhance_prompt": true,
  "cfg_scale_video": 3.0,
  "stg_scale_video": 1.0,
  "cfg_scale_audio": 7.0,
  "modality_scale": 3.0
}
```

### Poll
**GET** `/v1/video/tasks/{task_id}`

**Processing**
```json
{"status": "processing"}
```

**Completed**
```json
{"status": "completed", "url": "http://192.168.1.26:8004/outputs/ltx2_<task_id>.mp4"}
```

**Failed**
```json
{"status": "failed", "error": "error message"}
```

## Async Image-to-Video

### Submit
**POST** `/v1/video/async_i2v`

Multipart fields:
- `prompt`
- `image`
- `height` default `512`
- `width` default `768`
- `num_frames` default `121`
- `frame_rate` default `24.0`
- `num_inference_steps` default `20`
- `seed` default `42`
- `negative_prompt`
- `enhance_prompt` default `true`
- `cfg_scale_video` default `3.0`
- `stg_scale_video` default `1.0`

### Poll and download
Same pattern as async text-to-video.

## Current Defaults

| Parameter | Default |
|---|---|
| `frame_rate` | `24.0` |
| `num_inference_steps` | `20` |
| `cfg_scale_video` | `3.0` |
| `stg_scale_video` | `1.0` |
| `cfg_scale_audio` | `7.0` |
| `modality_scale` | `3.0` |

## Prompting Guide

- Prefer bright daylight or well-lit scenes for the cleanest local output.
- Keep to one main subject and one simple camera move.
- For image-to-video, describe only subtle motion that should happen after the input image.
- Avoid overloaded fantasy scenes, many moving objects, or rapid scene changes.
- Short clips stay cleaner. `17` to `25` frames is a good local range for high-clarity tests.
- Use reference-image I2V when quality matters more than novelty.

## Recommended Quality Recipes

### Clean local I2V

- `1024x576`
- `24 fps`
- `25 frames`
- `16-20` inference steps
- `cfg_scale_video=3.0`
- `stg_scale_video=1.0`
- simple prompt

### Fast sanity T2V

- `768x512`
- `24 fps`
- `17 frames`
- `8-12` inference steps

## Windows CLI Submission

> [!IMPORTANT]
> For JSON requests on Windows, write the body to a temporary `payload.json` file and use `curl -d @payload.json`. Do not inline JSON in the command because quoting often breaks at the first space.

Example:

```powershell
@'
{
  "prompt": "A bright cinematic daylight shot of a lone explorer standing on a snowy mountain summit as wind moves the coat slightly and the camera slowly pushes in.",
  "height": 576,
  "width": 1024,
  "num_frames": 25,
  "frame_rate": 24.0,
  "num_inference_steps": 20,
  "seed": 42
}
'@ | Set-Content payload.json

curl.exe -X POST "http://192.168.1.26:8004/v1/video/async_t2v" ^
  -H "Content-Type: application/json" ^
  -d @payload.json
```

## Health and Unload

- `GET /health` returns `{"status":"running","device":"cuda","vram_loaded":true|false}`
- `POST /v1/internal/unload` unloads the model so VRAM can be reclaimed
