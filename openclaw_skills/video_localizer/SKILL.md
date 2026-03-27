---
name: video_localizer
description: Localize videos into additional languages with dubbed audio and optional burned subtitles.
---

# Video Localizer Skill

Use this skill when the user needs multilingual delivery from one source video.

## Server Details

- **Base URL**: `http://192.168.1.26:9000`

## What It Produces

- Translated narration text
- Dubbed audio track
- Dubbed video
- Subtitle file
- Optional burned-subtitle video

## Required Flow

1. Create or reuse a Studio project.
2. Submit a job to `POST /v1/studio/projects/{project_id}/localization-runs`.
3. Poll `GET /v1/studio/tasks/{task_id}` until completion.
4. Return the localized package for each language.

## Request Body

- `media_url`: Source video URL
- `target_languages`: Example: `["Arabic", "French", "Spanish"]`
- `voice`: Voice to use for the dubbed narration
- `burn_subtitles`: Whether to produce burned-caption exports
- `subtitle_profile`: Example: `discord_clip`
