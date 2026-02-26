---
name: ltx2_video
description: API documentation for the LTX-2 Video server. Supports text-to-video (T2V) and image-to-video (I2V) generation with audio.
---

# LTX-2 Video Generation API

This skill provides the API specification for the LTX-2 Video server. Agents can use this to generate videos from text prompts or input images.

## Server Details

- **Base URL**: `http://192.168.1.26:8004`

> [!IMPORTANT]
> **Timeout Setting**: The first request may take **60–120 seconds** to load the model. Set your HTTP client timeout accordingly.
> **Cold Start**: The model auto-unloads after idle. First request after idle triggers a reload (~90s).

## API Endpoints

| Mode | Method | Endpoint | Content-Type | Notes |
|---|---|---|---|---|
| Async T2V (recommended) | POST | `/v1/video/async_t2v` | `application/json` | Returns `task_id` immediately |
| Async I2V (recommended) | POST | `/v1/video/async_i2v` | `multipart/form-data` | Returns `task_id` immediately |
| Poll Task Status | GET | `/v1/video/tasks/{task_id}` | — | Returns `processing`, `completed`, or `failed` |
| Download Video | GET | `/outputs/{filename}` | — | Serve generated MP4 |
| Sync T2V (legacy) | POST | `/v1/video/t2v` | `application/json` | Blocks until done (~2 min) |
| Sync I2V (legacy) | POST | `/v1/video/i2v` | `multipart/form-data` | Blocks until done (~2 min) |
| Health Check | GET | `/health` | — | |

> [!IMPORTANT]
> **Use the async endpoints.** Sync endpoints hold the HTTP connection for 1–2 minutes and will timeout most HTTP clients.

---

## Async Text-to-Video (Recommended)

### Step 1: Submit
**POST** `/v1/video/async_t2v`

```json
{
  "prompt": "A cinematic wide shot of a sunset over the ocean...",
  "height": 512,
  "width": 768,
  "num_frames": 121,
  "frame_rate": 24.0,
  "num_inference_steps": 40,
  "seed": 42
}
```

**Response:**
```json
{"task_id": "abc123", "status": "processing", "message": "Task queued. Poll GET /v1/video/tasks/{task_id}"}
```

### Step 2: Poll
**GET** `/v1/video/tasks/{task_id}`

Poll every 5–10 seconds until `status` changes.

**While processing:**
```json
{"status": "processing"}
```

**On success:**
```json
{"status": "completed", "url": "http://192.168.1.26:8004/outputs/ltx2_abc123.mp4"}
```

**On failure:**
```json
{"status": "failed", "error": "error message"}
```

### Step 3: Download
Fetch the video from the `url` returned in the completed task.

---

## Async Image-to-Video (Recommended)

### Step 1: Submit
**POST** `/v1/video/async_i2v`  
**Content-Type**: `multipart/form-data`

**Fields:**
- `prompt` (String): Describes the desired motion and scene continuation.
- `image` (File): The input image to animate.
- `height` (Integer, default 512)
- `width` (Integer, default 768)
- `num_frames` (Integer, default 121)
- `frame_rate` (Float, default 25.0)
- `num_inference_steps` (Integer, default 40)
- `seed` (Integer, default 42)

**Response:**
```json
{"task_id": "def456", "status": "processing", "message": "Task queued. Poll GET /v1/video/tasks/{task_id}"}
```

### Step 2–3: Poll + Download
Same as T2V above.

---

## WhatsApp Compatibility

> [!TIP]
> The output MP4 is **already WhatsApp-compatible**: H.264 video, AAC audio, yuv420p pixel format, MP4 container. No re-encoding needed — send the downloaded file directly.

## Example Agent Workflow: WhatsApp Image → Video

When a user sends an image via WhatsApp with a message like *"animate this"*:

1. **Download the image** from the WhatsApp media URL to get the raw bytes.
2. **Write a prompt** describing the desired animation (use the prompting guide below).
3. **Submit to I2V**:
   ```
   POST http://192.168.1.26:8004/v1/video/async_i2v
   Content-Type: multipart/form-data
   Fields: prompt="...", image=<downloaded bytes>
   ```
4. **Poll** `GET /v1/video/tasks/{task_id}` every 5–10 seconds.
5. **Download** the MP4 from the `url` in the completed response.
6. **Send** the MP4 file back to the WhatsApp conversation — it plays inline without conversion.

---

## Parameter Reference

| Parameter | T2V | I2V | Default | Notes |
|---|---|---|---|---|
| `prompt` | ✅ | ✅ | — | Required. See prompting guide below |
| `height` | ✅ | ✅ | 512 | Must be multiple of 64 |
| `width` | ✅ | ✅ | 768 | Must be multiple of 64 |
| `num_frames` | ✅ | ✅ | 121 | 49≈2s, 97≈4s, 121≈5s at 24fps |
| `frame_rate` | ✅ | ✅ | 25.0 | fps of output video |
| `num_inference_steps` | ✅ | ✅ | 40 | 20 = faster, 40 = better quality |
| `seed` | ✅ | ✅ | 42 | For reproducibility |
| `cfg_scale_video` | ✅ | ✅ | 3.0 | Video guidance strength |
| `stg_scale_video` | ✅ | ✅ | 1.0 | Spatiotemporal guidance |
| `cfg_scale_audio` | ✅ | — | 7.0 | Audio guidance strength |
| `modality_scale` | ✅ | — | 3.0 | Audio/video cross-modal guidance |

---

## Prompting Guide

Good prompts dramatically improve quality. Write as **one flowing paragraph** in **present tense**.

### Key Aspects to Include (in order)

1. **Establish the shot** — Camera angle, scale, film genre/style
   > `Cinematic medium shot`, `Handheld documentary style`, `Static POV from inside the oven`

2. **Set the scene** — Lighting, color palette, textures, atmosphere
   > `Warm golden light`, `neon glow`, `flickering candles`, `fog and rain`

3. **Describe the action** — A natural sequence from beginning to end
   > `The camera slowly pushes in... then the subject turns to face the viewer...`

4. **Define characters** — Age, clothing, hair, emotions via physical cues
   > `A woman in her 30s with short brown hair, eyes wide with focus`

5. **Identify camera movement** — When it shifts and to what
   > `Camera pans right to reveal...`, `Slow dolly back until...`, `Cranes up to show...`

6. **Describe audio** — Ambient sounds, dialogue (in quotes), voice style
   > `Ambient live music`, `"That's it... Dad's lost it."` (quiet whisper)

### For Best Results

- ✅ Write as **one flowing paragraph** (4–8 sentences)
- ✅ Use **present tense** verbs
- ✅ Match detail to shot scale — closeups need more precision than wide shots
- ✅ Specify camera's **relationship to subject**, not absolute moves
- ❌ Don't use lists or bullet points in the prompt
- ❌ Don't describe impossible continuous action (model window is 2–5s)
- ❌ Don't use vague style modifiers ("cool", "interesting") without specifics

### Style Categories

- **Animation:** `stop-motion`, `2D/3D animation`, `claymation`, `hand-drawn`
- **Stylized:** `comic book`, `cyberpunk`, `8-bit pixel`, `surreal`, `minimalist`, `painterly`
- **Cinematic:** `period drama`, `film noir`, `fantasy`, `epic space opera`, `thriller`, `documentary`

### Visual Details Vocabulary

| Category | Examples |
|---|---|
| Lighting | `flickering candles`, `neon glow`, `natural sunlight`, `dramatic shadows` |
| Textures | `rough stone`, `smooth metal`, `worn fabric`, `glossy surfaces` |
| Color palette | `vibrant`, `muted`, `monochromatic`, `high contrast` |
| Atmosphere | `fog`, `rain`, `dust`, `particles`, `smoke` |

### Camera Language

| Motion | Keywords |
|---|---|
| Translation | `follows`, `tracks`, `pushes in`, `pulls back`, `dollies back` |
| Rotation | `pans across`, `circles around`, `tilts upward`, `over-the-shoulder` |
| Characteristics | `handheld movement`, `lens flares`, `film grain`, `jittery stop-motion` |
| Scale | `expansive`, `epic`, `intimate`, `claustrophobic` |
| Pacing | `slow motion`, `time-lapse`, `rapid cuts`, `lingering shot`, `fade-in` |

### Example Prompts

**Action/Cinematic:**
```
Cinematic action packed shot. The man says silently: "We need to run." The camera zooms in 
on his mouth then immediately screams: "NOW!". The camera zooms back out, he turns around 
and starts running away, the camera tracks his run in handheld style. The camera cranes up 
and shows him run into the distance down the street at a busy New York night.
```

**Character/Comedy:**
```
INT. OVEN – DAY. Static camera from inside the oven, looking outward through the slightly 
fogged glass door. Warm golden light glows around freshly baked cookies. The baker's face 
fills the frame, eyes wide with focus, his breath fogging the glass as he leans in. Baker 
(whispering dramatically): "Today… I achieve perfection." Coworker (mouth full): "Nope. 
You forgot the sugar." Pixar style acting and timing.
```

### Timeout

- **First Request Slow**: The API loads models on demand. The first request can take 1–2 minutes.
