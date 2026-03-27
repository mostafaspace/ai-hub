---
name: shortform_factory
description: Create monetizable short-form clip packs from long videos, webinars, podcasts, demos, or founder recordings.
---

# Short-Form Factory Skill

Use this skill when a user wants to turn one long piece of media into a package of sellable short-form assets.

## Server Details

- **Base URL**: `http://192.168.1.26:9000`

## Best Use Cases

- Webinar to social campaign
- Podcast to clips
- Founder rant to daily posts
- Product demo to platform-ready cutdowns

## Required Flow

1. Create or reuse a Studio project.
2. Submit a job to `POST /v1/studio/projects/{project_id}/short-form-packs`.
3. Poll `GET /v1/studio/tasks/{task_id}` until completion.
4. Return the generated clip URLs, subtitle URLs, thumbnails, and platform variants.

## Request Body

- `media_url`: Source video or audio URL
- `target_count`: Desired clip count, up to 20
- `clip_duration_sec`: Optional clip duration override
- `output_profiles`: Example: `["youtube_short", "tiktok_vertical", "discord_clip"]`
- `burn_captions`: Whether to burn subtitles into rendered clips
- `include_contact_sheet`: Whether to include a contact sheet preview

## Example

```json
{
  "media_url": "http://192.168.1.26:9000/outputs/practical/uploads/generic/.../webinar.mp4",
  "target_count": 8,
  "output_profiles": ["youtube_short", "tiktok_vertical", "discord_clip"],
  "burn_captions": true,
  "include_contact_sheet": true
}
```
