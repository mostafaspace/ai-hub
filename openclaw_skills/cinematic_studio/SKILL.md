---
name: cinematic_studio
description: Build high-end cinematic project assets through the Studio task system, including series intros and immersive music-driven videos.
---

# Cinematic Studio Skill

The **Cinematic Studio** endpoints turn the existing AI-Hub stack into a creator workflow for polished visual packages.

Instead of manually calling image generation, i2v, music, narration, and FFmpeg assembly step by step, the Orchestrator handles the full sequence as one Studio task and attaches every artifact back to the project.

## Server Details

- **Base URL**: `http://192.168.1.26:9000`
- **OpenAPI Spec**: `openapi.yaml`

## Best Uses

- TV or streaming-style title sequences
- Immersive 3-minute mood videos
- Music visualizers
- Branded intro packages
- Cinematic concept reels

## Async Pattern

1. Create or reuse a Studio project.
2. Submit a cinematic task.
3. Poll `GET /v1/studio/tasks/{task_id}` until it completes.
4. Use the returned `final_video_url` plus the attached storyboard, music, and plan assets.

## Endpoints

### Unified Cinematic Director
**POST** `/v1/studio/projects/{project_id}/cinematic-productions`

This should be the default OpenClaw route.

- If the request includes `source_image_urls`, the server routes toward image-guided generation and prefers `i2v`.
- If the request has only concept text, the server routes toward prompt-guided generation and builds the shot plan internally, preferring `t2v`.

```json
{
  "production_type": "auto",
  "title": "Empire of Ashes",
  "concept": "after the sun is shattered, rival bloodlines wage war above the clouds for the last ember",
  "duration_sec": 45,
  "scene_count": 6,
  "source_image_urls": [],
  "use_project_image_assets": false,
  "use_i2v": true,
  "output_profile": "cinematic_wide"
}
```

### Series Intro Builder
**POST** `/v1/studio/projects/{project_id}/series-intros`

```json
{
  "title": "Glass Kingdom",
  "concept": "a dynasty unraveling on a floating city",
  "genre": "prestige sci-fi drama",
  "style_notes": "cinematic lighting, premium lensing, rich atmosphere, netflix-level polish",
  "duration_sec": 60,
  "scene_count": 8,
  "voice": "Vivian",
  "use_i2v": true,
  "output_profile": "cinematic_wide"
}
```

What you get:
- Saved shot plan JSON
- Storyboard frames
- Animated or fallback hold-shot clips
- Narration track
- Music score
- Final cinematic cut

### Immersive Video Builder
**POST** `/v1/studio/projects/{project_id}/immersive-videos`

```json
{
  "concept": "an atmospheric journey through a rain-soaked neon district",
  "style_notes": "immersive cinematic visuals, moody atmosphere, premium color, slow camera motion",
  "duration_sec": 180,
  "scene_count": 12,
  "use_i2v": true,
  "output_profile": "cinematic_wide"
}
```

What you get:
- Saved shot plan JSON
- Storyboard frames
- Animated or fallback hold-shot clips
- Music-driven final cut

## Guidance For OpenClaw

- Prefer the unified cinematic director route first.
- Prefer `cinematic_wide` for polished landscape outputs.
- Use `shot_plan` when the user already knows the exact beats or framing.
- Keep `scene_count` moderate. Fewer stronger shots usually look better than too many weak ones.
- Poll the Studio task until completion and then surface the `final_video_url` plus any attached plan and storyboard assets.
