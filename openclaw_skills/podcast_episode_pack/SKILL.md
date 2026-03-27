---
name: podcast_episode_pack
description: Build a podcast or narrated talking-head production pack from notes, outlines, articles, URLs, or brochure text.
---

# Podcast Episode Pack Skill

Use this skill when the user wants recurring audio or narrated video content from written material.

## Server Details

- **Base URL**: `http://192.168.1.26:9000`

## What It Produces

- Episode script
- Show notes
- Voice auditions
- Final narrated audio
- Cover art
- Optional teaser video

## Required Flow

1. Create or reuse a Studio project.
2. Submit a job to `POST /v1/studio/projects/{project_id}/episode-packs`.
3. Poll `GET /v1/studio/tasks/{task_id}` until completion.
4. Return the audio, script, cover image, audition samples, and teaser asset.

## Request Body

- `source_text`: Direct notes, outline, article text, or script source
- `source_url`: Optional URL to ingest text from
- `brochure_url`: Optional uploaded brochure or text asset URL
- `title`: Optional episode title
- `voice`: Final narration voice
- `audition_voices`: Optional voices to compare before delivery
- `language`: Optional language hint for narration
- `cover_prompt`: Optional override for cover art generation
- `generate_teaser_video`: Whether to create a teaser clip
