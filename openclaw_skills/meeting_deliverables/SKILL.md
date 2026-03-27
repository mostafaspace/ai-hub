---
name: meeting_deliverables
description: Convert a meeting recording into transcript, summary, action items, follow-up email, briefing audio, and a stakeholder clip.
---

# Meeting Deliverables Skill

Use this skill when the user wants a meeting recording turned into practical follow-up assets.

## Server Details

- **Base URL**: `http://192.168.1.26:9000`

## What It Produces

- Transcript
- Summary
- Action items
- Follow-up email draft
- Audio briefing
- Stakeholder-ready video clip

## Required Flow

1. Create or reuse a Studio project.
2. Submit a job to `POST /v1/studio/projects/{project_id}/meeting-deliverables`.
3. Poll `GET /v1/studio/tasks/{task_id}` until completion.
4. Deliver the URLs plus the extracted action items.

## Request Body

- `media_url`: Meeting recording URL
- `recipient_name`: Optional email recipient or team name
- `sender_name`: Optional signer for the email draft
- `briefing_voice`: Voice for the stakeholder audio briefing
- `stakeholder_clip_duration_sec`: Length of the generated recap clip
