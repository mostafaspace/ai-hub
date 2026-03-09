import io
import json
import os
import sys
import zipfile

from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import orchestrator.server as orch
import orchestrator.studio_api as studio
import orchestrator.studio_media as studio_media


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


def test_studio_project_workspace_flow_and_uploads(tmp_path, monkeypatch):
    configure_studio_dirs(tmp_path, monkeypatch)

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
    project_id = project["project_id"]

    upload_resp = client.post(
        "/v1/studio/uploads",
        data={"project_id": project_id, "kind": "image", "label": "cover-art"},
        files={"file": ("cover.png", b"pngdata", "image/png")},
    )
    assert upload_resp.status_code == 200
    upload_payload = upload_resp.json()
    assert upload_payload["asset"]["label"] == "cover-art"
    assert "/outputs/practical/uploads/" in upload_payload["upload"]["url"]

    timeline_resp = client.put(
        f"/v1/studio/projects/{project_id}/timeline",
        json={
            "video_clips": [
                {
                    "asset_url": upload_payload["upload"]["url"],
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
                    "label": "Main",
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
    assert plan["render_ready"] is True
    assert plan["supported_render_scope"]["video_clips"] == "sequential concat only"

    assets_resp = client.get(f"/v1/studio/projects/{project_id}/assets")
    assert assets_resp.status_code == 200
    assert assets_resp.json()["assets"][0]["kind"] == "image"


def test_character_pack_photo_attachment(tmp_path, monkeypatch):
    configure_studio_dirs(tmp_path, monkeypatch)

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

    photo_resp = client.post(
        f"/v1/studio/character-packs/{pack['pack_id']}/photo",
        data={"label": "reference-headshot"},
        files={"file": ("headshot.jpg", b"jpegdata", "image/jpeg")},
    )
    assert photo_resp.status_code == 200
    photo_payload = photo_resp.json()
    assert "/outputs/practical/uploads/" in photo_payload["photo_url"]

    get_resp = client.get(f"/v1/studio/character-packs/{pack['pack_id']}")
    assert get_resp.status_code == 200
    pack_data = get_resp.json()
    assert pack_data["photo_url"] == photo_payload["photo_url"]
    assert photo_payload["photo_url"] in pack_data["reference_asset_urls"]

    combined = studio._apply_character_pack("standing on a rooftop", pack_data)
    assert "cinematic hero portrait" in combined
    assert "standing on a rooftop" in combined


def test_task_persistence_cancel_resume_and_observability(tmp_path, monkeypatch):
    configure_studio_dirs(tmp_path, monkeypatch)

    def fake_start_background_task(task_id):
        task = studio.STUDIO_TASKS[task_id]
        task["status"] = "completed"
        task["result"] = {"task_id": task_id, "status": "completed"}
        task["finished_at"] = studio._now()
        studio._persist_task(task)

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
    assert task_resp.json()["status"] == "completed"
    assert os.path.exists(studio._task_path(task_id))

    task = studio.STUDIO_TASKS[task_id]
    task["status"] = "processing"
    studio._persist_task(task)
    studio._load_persisted_tasks()
    assert studio.STUDIO_TASKS[task_id]["status"] == "interrupted"

    resume_resp = client.post(f"/v1/studio/tasks/{task_id}/resume")
    assert resume_resp.status_code == 200
    assert resume_resp.json()["attempt"] == 2

    pending_task = studio._create_task("captions", "captions", {"media_url": "http://example.com/demo.mp4"}, None, None, "captions")
    cancel_resp = client.post(f"/v1/studio/tasks/{pending_task['task_id']}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"

    observability_resp = client.get("/v1/studio/observability")
    assert observability_resp.status_code == 200
    metrics = observability_resp.json()
    assert metrics["tasks_total"] >= 2
    assert "voice_auditions" in metrics["tasks_by_kind"]


def test_project_import_and_project_level_jobs(tmp_path, monkeypatch):
    configure_studio_dirs(tmp_path, monkeypatch)

    create_resp = client.post(
        "/v1/studio/projects",
        json={"name": "Daily Show", "default_profile": "youtube_short"},
    )
    project_id = create_resp.json()["project_id"]

    def fake_start_background_task(task_id):
        task = studio.STUDIO_TASKS[task_id]
        kind = task["kind"]
        task["status"] = "completed"
        if kind == "timeline_render":
            task["result"] = {"output_url": "http://127.0.0.1:9000/outputs/practical/fake/timeline.mp4"}
        elif kind == "director_attach":
            task["result"] = {"output_url": "http://127.0.0.1:9000/outputs/practical/fake/director.mp4"}
        else:
            task["result"] = {"export_url": "http://127.0.0.1:9000/outputs/practical/fake/export.zip"}
        task["finished_at"] = studio._now()
        studio._persist_task(task)

    monkeypatch.setattr(studio, "start_background_task", fake_start_background_task)

    timeline_task = client.post(f"/v1/studio/projects/{project_id}/timeline/render", json={})
    assert timeline_task.status_code == 200

    director_task = client.post(
        f"/v1/studio/projects/{project_id}/director-runs",
        json={"image_prompt": "city skyline at dusk", "voiceover_text": "Tonight on the show", "voice": "Vivian"},
    )
    assert director_task.status_code == 200

    export_task = client.post(f"/v1/studio/projects/{project_id}/export", json={})
    assert export_task.status_code == 200

    project_tasks_resp = client.get(f"/v1/studio/projects/{project_id}/tasks")
    assert project_tasks_resp.status_code == 200
    assert len(project_tasks_resp.json()["tasks"]) == 3

    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "project.json",
            json.dumps(
                {
                    "project_id": "proj-old",
                    "name": "Imported Project",
                    "description": "archived",
                    "default_profile": "youtube_short",
                    "notes": "",
                    "tags": [],
                    "assets": [
                        {
                            "asset_id": "asset-1",
                            "kind": "image",
                            "label": "poster",
                            "url": "http://example.com/poster.png",
                            "metadata": {},
                            "created_at": 1.0,
                        }
                    ],
                    "timeline": {"video_clips": [], "audio_tracks": [], "subtitle_tracks": [], "notes": "", "last_render_plan": None},
                    "created_at": 1.0,
                    "updated_at": 1.0,
                }
            ),
        )
        archive.writestr("manifest.json", json.dumps({"bundled_assets": [{"asset_id": "asset-1", "bundle_path": "assets/poster.png"}]}))
        archive.writestr("assets/poster.png", b"posterdata")
    archive_bytes.seek(0)

    import_resp = client.post(
        "/v1/studio/projects/import",
        files={"file": ("project-export.zip", archive_bytes.read(), "application/zip")},
    )
    assert import_resp.status_code == 200
    payload = import_resp.json()
    assert payload["imported_assets"] == 1
    assert payload["project"]["project_id"] != "proj-old"
    assert "/outputs/practical/imports/" in payload["project"]["assets"][0]["url"]


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
    assert content.count("-->") >= 2