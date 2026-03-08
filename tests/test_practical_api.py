import os
import sys

from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import orchestrator.server as orch
import orchestrator.studio_api as studio
import orchestrator.studio_media as studio_media


client = TestClient(orch.app)


def test_studio_project_workspace_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(studio, "PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setattr(studio, "PACKS_DIR", str(tmp_path / "packs"))
    monkeypatch.setattr(studio, "OUTPUTS_DIR", str(tmp_path / "outputs"))
    os.makedirs(studio.PROJECTS_DIR, exist_ok=True)
    os.makedirs(studio.PACKS_DIR, exist_ok=True)
    os.makedirs(studio.OUTPUTS_DIR, exist_ok=True)

    create_resp = client.post(
        "/v1/studio/projects",
        json={
            "name": "Launch Kit",
            "description": "Workspace for a launch campaign",
            "default_profile": "youtube_short",
            "tags": ["launch", "video"],
        },
    )
    assert create_resp.status_code == 200
    project = create_resp.json()
    assert project["name"] == "Launch Kit"
    assert project["timeline"]["video_clips"] == []

    project_id = project["project_id"]
    timeline_resp = client.put(
        f"/v1/studio/projects/{project_id}/timeline",
        json={
            "video_clips": [
                {
                    "asset_url": "http://example.com/clip.mp4",
                    "label": "Intro",
                    "start_sec": 0,
                    "duration_sec": 5,
                }
            ],
            "audio_tracks": [
                {
                    "asset_url": "http://example.com/music.mp3",
                    "label": "Bed",
                    "start_sec": 0,
                    "duration_sec": 8,
                    "volume": 0.7,
                }
            ],
            "subtitle_tracks": [
                {
                    "subtitle_url": "http://example.com/captions.srt",
                    "label": "Main"
                }
            ],
            "notes": "Keep this lightweight and additive.",
        },
    )
    assert timeline_resp.status_code == 200

    plan_resp = client.post(f"/v1/studio/projects/{project_id}/timeline/plan", json={})
    assert plan_resp.status_code == 200
    plan = plan_resp.json()["plan"]
    assert plan["format_profile"] == "youtube_short"
    assert plan["estimated_duration_sec"] == 8.0
    assert plan["supported_render_scope"] == "manifest_only"


def test_character_pack_and_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(studio, "PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setattr(studio, "PACKS_DIR", str(tmp_path / "packs"))
    monkeypatch.setattr(studio, "OUTPUTS_DIR", str(tmp_path / "outputs"))
    os.makedirs(studio.PROJECTS_DIR, exist_ok=True)
    os.makedirs(studio.PACKS_DIR, exist_ok=True)
    os.makedirs(studio.OUTPUTS_DIR, exist_ok=True)

    profiles_resp = client.get("/v1/studio/format-profiles")
    assert profiles_resp.status_code == 200
    profiles = profiles_resp.json()["profiles"]
    assert "youtube_short" in profiles
    assert profiles["podcast_mp3"]["kind"] == "audio"

    pack_resp = client.post(
        "/v1/studio/character-packs",
        json={
            "name": "Hero Pack",
            "voice": "Vivian",
            "prompt_prefix": "cinematic hero portrait",
            "prompt_suffix": "high detail, dramatic lighting",
            "negative_prompt": "blurry",
        },
    )
    assert pack_resp.status_code == 200
    pack = pack_resp.json()
    assert pack["voice"] == "Vivian"

    get_resp = client.get(f"/v1/studio/character-packs/{pack['pack_id']}")
    assert get_resp.status_code == 200
    assert get_resp.json()["negative_prompt"] == "blurry"

    combined = studio._apply_character_pack("standing on a rooftop", pack)
    assert "cinematic hero portrait" in combined
    assert "standing on a rooftop" in combined


def test_studio_task_lifecycle_without_touching_live_routes(monkeypatch):
    original_tasks = dict(studio.STUDIO_TASKS)
    studio.STUDIO_TASKS.clear()

    def fake_start_background_task(task_id, runner):
        studio._complete_task(task_id, {"task_id": task_id, "status": "completed", "samples": []})

    monkeypatch.setattr(studio, "start_background_task", fake_start_background_task)

    response = client.post(
        "/v1/studio/voice-auditions",
        json={
            "text": "Testing audition mode",
            "voices": ["Vivian", "Ethan"],
        },
    )
    assert response.status_code == 200
    task_id = response.json()["task_id"]

    task_resp = client.get(f"/v1/studio/tasks/{task_id}")
    assert task_resp.status_code == 200
    task = task_resp.json()
    assert task["status"] == "completed"
    assert task["result"]["samples"] == []

    studio.STUDIO_TASKS.clear()
    studio.STUDIO_TASKS.update(original_tasks)


def test_approximate_subtitles_are_split_and_timestamped(tmp_path):
    subtitle_path = tmp_path / "captions.srt"
    studio_media.write_approximate_srt(
        "Hello world. This is a longer line that should still become readable subtitles.",
        6.0,
        str(subtitle_path),
    )
    content = subtitle_path.read_text(encoding="utf-8")
    assert "00:00:00,000 -->" in content
    assert "Hello world." in content
    assert content.count("-->" ) >= 2


def test_studio_task_list_endpoint(monkeypatch):
    original_tasks = dict(studio.STUDIO_TASKS)
    studio.STUDIO_TASKS.clear()
    studio.STUDIO_TASKS['task-a'] = {
        'task_id': 'task-a',
        'kind': 'voice_auditions',
        'status': 'completed',
        'created_at': 1.0,
        'updated_at': 2.0,
        'project_id': None,
        'result': {'samples': []},
        'error': None,
    }

    response = client.get('/v1/studio/tasks')
    assert response.status_code == 200
    payload = response.json()
    assert payload['tasks'][0]['task_id'] == 'task-a'

    studio.STUDIO_TASKS.clear()
    studio.STUDIO_TASKS.update(original_tasks)