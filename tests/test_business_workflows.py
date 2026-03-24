import asyncio
import json
import os
import sys

from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import orchestrator.business_api as business
import orchestrator.server as orch
import orchestrator.studio_api as studio


client = TestClient(orch.app)


def configure_studio_dirs(tmp_path, monkeypatch):
    root_dir = tmp_path / "orchestrator"
    state_dir = root_dir / "studio_state"
    outputs_root = root_dir / "director_outputs"
    monkeypatch.setattr(studio, "ROOT_DIR", str(root_dir))
    monkeypatch.setattr(studio, "STATE_DIR", str(state_dir))
    monkeypatch.setattr(studio, "TASKS_DIR", str(state_dir / "tasks"))
    monkeypatch.setattr(studio, "PROJECTS_DIR", str(root_dir / "studio_projects"))
    monkeypatch.setattr(studio, "PACKS_DIR", str(root_dir / "studio_character_packs"))
    monkeypatch.setattr(studio, "OUTPUTS_DIR", str(outputs_root / "practical"))
    monkeypatch.setattr(studio, "UPLOADS_DIR", str(outputs_root / "practical" / "uploads"))
    monkeypatch.setattr(studio, "EXPORTS_DIR", str(outputs_root / "practical" / "exports"))
    monkeypatch.setattr(studio, "IMPORTS_DIR", str(outputs_root / "practical" / "imports"))
    monkeypatch.setattr(studio, "REGISTRY_PATH", str(root_dir / "models.yaml"))
    os.makedirs(root_dir, exist_ok=True)
    with open(studio.REGISTRY_PATH, "w", encoding="utf-8") as handle:
        json.dump({"backends": {}}, handle)
    studio._ensure_dirs()
    studio.STUDIO_TASKS.clear()
    studio.RUNNING_TASKS.clear()


def create_project():
    response = client.post("/v1/studio/projects", json={"name": "Revenue Ops", "default_profile": "youtube_short"})
    assert response.status_code == 200
    return response.json()["project_id"]


def touch_file(path: str, content: bytes = b"data"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(content)


def seed_task(task_id: str):
    studio.STUDIO_TASKS[task_id] = {
        "task_id": task_id,
        "status": "processing",
        "project_id": None,
        "events": [],
        "cancel_requested": False,
    }


def test_business_endpoints_queue_tasks(tmp_path, monkeypatch):
    configure_studio_dirs(tmp_path, monkeypatch)
    project_id = create_project()

    def fake_start_background_task(task_id):
        task = studio.STUDIO_TASKS[task_id]
        task["status"] = "completed"
        task["result"] = {"task_id": task_id, "status": "completed"}
        task["finished_at"] = studio._now()
        studio._persist_task(task)

    monkeypatch.setattr(studio, "start_background_task", fake_start_background_task)

    cases = [
        (f"/v1/studio/projects/{project_id}/short-form-packs", {"media_url": "http://example.com/demo.mp4"}),
        (f"/v1/studio/projects/{project_id}/episode-packs", {"source_text": "Launch notes for our weekly show."}),
        (f"/v1/studio/projects/{project_id}/localization-runs", {"media_url": "http://example.com/demo.mp4"}),
        (f"/v1/studio/projects/{project_id}/meeting-deliverables", {"media_url": "http://example.com/meeting.mp4"}),
        (f"/v1/studio/projects/{project_id}/marketing-kits", {"source_text": "A product that automates clip creation."}),
    ]

    for url, payload in cases:
        response = client.post(url, json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "completed"


def test_short_form_runner_builds_clip_pack(tmp_path, monkeypatch):
    configure_studio_dirs(tmp_path, monkeypatch)
    project_id = create_project()
    seed_task("task-short")

    source_path = str(tmp_path / "source.mp4")
    touch_file(source_path)
    monkeypatch.setattr(business, "ffmpeg_available", lambda: True)
    async def fake_prepare_media(task_id, media_url):
        return {
            "source_path": source_path,
            "audio_path": source_path,
            "duration_seconds": 90.0,
            "source_ext": ".mp4",
            "is_video": True,
        }

    async def fake_transcribe(audio_path, prompt=""):
        return "Hook one. Hook two. Hook three. Hook four. Hook five. Hook six."

    monkeypatch.setattr(business, "_prepare_media", fake_prepare_media)
    monkeypatch.setattr(business, "_transcribe_audio", fake_transcribe)
    monkeypatch.setattr(
        business,
        "render_video_clip",
        lambda input_path, output_path, profile_name, trim_in_sec=0.0, duration_sec=None, subtitles_path=None: (touch_file(output_path), (True, "ok"))[1],
    )
    monkeypatch.setattr(
        business,
        "extract_thumbnail",
        lambda input_path, output_path, timestamp_sec=0.0: (touch_file(output_path), (True, "ok"))[1],
    )
    monkeypatch.setattr(
        business,
        "create_contact_sheet",
        lambda input_path, output_path, columns=4, rows=2, thumb_width=320: (touch_file(output_path), (True, "ok"))[1],
    )

    result = asyncio.run(
        business._run_short_form_pack_task(
            "task-short",
            {
                "project_id": project_id,
                "media_url": "http://example.com/demo.mp4",
                "target_count": 3,
                "output_profiles": ["youtube_short", "discord_clip"],
                "burn_captions": True,
                "include_contact_sheet": True,
                "label": "short-pack",
            },
        )
    )

    assert len(result["clips"]) == 3
    assert result["clips"][0]["variants"][0]["profile"] == "youtube_short"
    project = studio._load_project(project_id)
    assert any(asset["label"] == "short-pack-clip-1" for asset in project["assets"])


def test_episode_pack_runner_creates_audio_art_and_teaser(tmp_path, monkeypatch):
    configure_studio_dirs(tmp_path, monkeypatch)
    project_id = create_project()
    seed_task("task-episode")

    async def fake_fetch_text(*args, **kwargs):
        return "This is our founder story and product vision."

    async def fake_writer(*args, **kwargs):
        return {
            "title": "Founder Briefing",
            "script": "Welcome to the show. We are launching a new offer today.",
            "show_notes": "Launch overview and positioning.",
            "teaser_script": "We are launching a new offer today.",
            "cover_prompt": "premium podcast cover art",
        }

    async def fake_tts(text, voice, output_path, language=None, instruct=None):
        touch_file(output_path)
        return studio._relative_output_url(output_path)

    async def fake_image(task_id, prompt, output_name="hero.png"):
        return f"http://127.0.0.1:9000/outputs/practical/{output_name}"

    async def fake_director(image_prompt, voiceover_text, voice):
        return {"task_id": "dir-1", "output_url": "http://127.0.0.1:9000/outputs/practical/teaser.mp4"}

    monkeypatch.setattr(business, "_fetch_source_text", fake_fetch_text)
    monkeypatch.setattr(business, "_writer_json", fake_writer)
    monkeypatch.setattr(business, "_synthesize_voice", fake_tts)
    monkeypatch.setattr(business, "_render_image_from_prompt", fake_image)
    monkeypatch.setattr(business, "_run_director_and_wait", fake_director)

    result = asyncio.run(
        business._run_episode_pack_task(
            "task-episode",
            {
                "project_id": project_id,
                "source_text": "launch notes",
                "voice": "Vivian",
                "generate_teaser_video": True,
                "label": "episode-pack",
            },
        )
    )

    assert result["title"] == "Founder Briefing"
    assert len(result["voice_auditions"]) >= 1
    assert result["cover_image_url"].endswith("episode-cover.png")


def test_localization_runner_creates_dub_assets(tmp_path, monkeypatch):
    configure_studio_dirs(tmp_path, monkeypatch)
    project_id = create_project()
    seed_task("task-localize")

    source_path = str(tmp_path / "dub-source.mp4")
    touch_file(source_path)
    monkeypatch.setattr(business, "ffmpeg_available", lambda: True)
    async def fake_prepare_media(task_id, media_url):
        return {
            "source_path": source_path,
            "audio_path": source_path,
            "duration_seconds": 42.0,
            "source_ext": ".mp4",
            "is_video": True,
        }

    async def fake_transcribe(audio_path, prompt=""):
        return "Welcome to our new demo."

    async def fake_writer(prompt, audio_path=None, max_tokens=512):
        return "مرحبا بكم في العرض الجديد."

    async def fake_tts(text, voice, output_path, language=None, instruct=None):
        touch_file(output_path)
        return studio._relative_output_url(output_path)

    monkeypatch.setattr(business, "_prepare_media", fake_prepare_media)
    monkeypatch.setattr(business, "_transcribe_audio", fake_transcribe)
    monkeypatch.setattr(business, "_call_writer", fake_writer)
    monkeypatch.setattr(business, "_synthesize_voice", fake_tts)
    monkeypatch.setattr(
        business,
        "replace_audio_track",
        lambda video_path, audio_path, output_path: (touch_file(output_path), (True, "ok"))[1],
    )
    monkeypatch.setattr(
        business,
        "render_video_clip",
        lambda input_path, output_path, profile_name, trim_in_sec=0.0, duration_sec=None, subtitles_path=None: (touch_file(output_path), (True, "ok"))[1],
    )

    result = asyncio.run(
        business._run_localization_task(
            "task-localize",
            {
                "project_id": project_id,
                "media_url": "http://example.com/demo.mp4",
                "target_languages": ["Arabic"],
                "label": "localization",
            },
        )
    )

    assert result["localizations"][0]["language"] == "Arabic"
    assert result["localizations"][0]["dubbed_video_url"].endswith(".mp4")


def test_meeting_runner_creates_docs_audio_and_clip(tmp_path, monkeypatch):
    configure_studio_dirs(tmp_path, monkeypatch)
    project_id = create_project()
    seed_task("task-meeting")

    source_path = str(tmp_path / "meeting.mp4")
    touch_file(source_path)
    monkeypatch.setattr(business, "ffmpeg_available", lambda: True)
    async def fake_prepare_media(task_id, media_url):
        return {
            "source_path": source_path,
            "audio_path": source_path,
            "duration_seconds": 55.0,
            "source_ext": ".mp4",
            "is_video": True,
        }

    async def fake_transcribe(audio_path, prompt=""):
        return "We reviewed the launch plan. Marketing owns the landing page. Sales will update outreach."

    async def fake_writer(*args, **kwargs):
        return {
            "summary": "The meeting aligned launch ownership.",
            "action_items": ["Marketing updates the landing page.", "Sales updates outreach messaging."],
            "follow_up_email": "Hi team,\n\nWe aligned on launch ownership.\n\nBest,\nOpenClaw",
            "briefing_script": "Here is your briefing. The meeting aligned launch ownership.",
        }

    async def fake_tts(text, voice, output_path, language=None, instruct=None):
        touch_file(output_path)
        return studio._relative_output_url(output_path)

    monkeypatch.setattr(business, "_prepare_media", fake_prepare_media)
    monkeypatch.setattr(business, "_transcribe_audio", fake_transcribe)
    monkeypatch.setattr(business, "_writer_json", fake_writer)
    monkeypatch.setattr(business, "_synthesize_voice", fake_tts)
    monkeypatch.setattr(
        business,
        "render_video_clip",
        lambda input_path, output_path, profile_name, trim_in_sec=0.0, duration_sec=None, subtitles_path=None: (touch_file(output_path), (True, "ok"))[1],
    )

    result = asyncio.run(
        business._run_meeting_deliverables_task(
            "task-meeting",
            {"project_id": project_id, "media_url": "http://example.com/meeting.mp4", "label": "meeting-pack"},
        )
    )

    assert result["briefing_audio_url"].endswith(".mp3")
    assert len(result["action_items"]) == 2
    assert result["stakeholder_clip_url"].endswith(".mp4")


def test_marketing_runner_creates_campaign_assets(tmp_path, monkeypatch):
    configure_studio_dirs(tmp_path, monkeypatch)
    project_id = create_project()
    seed_task("task-marketing")

    async def fake_fetch_text(*args, **kwargs):
        return "A product that turns long videos into sales-ready short clips."

    async def fake_writer(*args, **kwargs):
        return {
            "product_name": "ClipForge",
            "hero_headline": "Turn Long Videos Into Revenue Clips",
            "benefit_bullets": ["Auto clip extraction", "Faster repurposing", "Agency-ready outputs"],
            "hooks": ["Turn every webinar into a campaign.", "Repurpose faster without hiring more editors."],
            "voice_script": "ClipForge turns long videos into revenue clips fast.",
            "hero_prompt": "premium product hero image",
            "comparison_prompt": "clean comparison visual",
        }

    async def fake_image(task_id, prompt, output_name="hero.png"):
        return f"http://127.0.0.1:9000/outputs/practical/{output_name}"

    async def fake_tts(text, voice, output_path, language=None, instruct=None):
        touch_file(output_path)
        return studio._relative_output_url(output_path)

    async def fake_director(image_prompt, voiceover_text, voice):
        return {"task_id": "dir-2", "output_url": "http://127.0.0.1:9000/outputs/practical/ad.mp4"}

    monkeypatch.setattr(business, "_fetch_source_text", fake_fetch_text)
    monkeypatch.setattr(business, "_writer_json", fake_writer)
    monkeypatch.setattr(business, "_render_image_from_prompt", fake_image)
    monkeypatch.setattr(business, "_synthesize_voice", fake_tts)
    monkeypatch.setattr(business, "_run_director_and_wait", fake_director)

    result = asyncio.run(
        business._run_marketing_kit_task(
            "task-marketing",
            {"project_id": project_id, "source_text": "marketing brief", "label": "marketing-kit"},
        )
    )

    assert result["product_name"] == "ClipForge"
    assert len(result["hooks"]) == 2
    assert result["voiceover_ad_url"].endswith(".mp3")
