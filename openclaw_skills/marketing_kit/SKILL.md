---
name: marketing_kit
description: Turn a product brief, URL, or brochure into hero visuals, comparison creative, hooks, teaser assets, and voiceover ads.
---

# Marketing Kit Skill

Use this skill when the user wants fast product marketing assets from a brief.

## Server Details

- **Base URL**: `http://192.168.1.26:9000`

## What It Produces

- Marketing brief
- Hero image
- Comparison visual
- Voiceover ad
- Sellable hooks
- Optional teaser video

## Required Flow

1. Create or reuse a Studio project.
2. Submit a job to `POST /v1/studio/projects/{project_id}/marketing-kits`.
3. Poll `GET /v1/studio/tasks/{task_id}` until completion.
4. Return the image, audio, text, and teaser assets.

## Request Body

- `source_text`: Direct product brief or feature list
- `source_url`: Optional URL to ingest
- `brochure_url`: Optional uploaded brochure or source asset
- `product_name`: Optional override for the product name
- `target_audiences`: Optional list of audience hints
- `voice`: Voice for generated ad copy
- `generate_teaser_video`: Whether to create a teaser video
