import asyncio
import json
import os
import sys

from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import orchestrator.cinematic_api as cinematic
import orchestrator.comfy_client as comfy_client
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
    response = client.post("/v1/studio/projects", json={"name": "Cinematic Lab", "default_profile": "cinematic_wide"})
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


def test_cinematic_endpoints_queue_tasks(tmp_path, monkeypatch):
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
        (
            f"/v1/studio/projects/{project_id}/cinematic-productions",
            {"concept": "a war for the last ember", "production_type": "auto"},
        ),
        (
            f"/v1/studio/projects/{project_id}/series-intros",
            {"title": "Glass Kingdom", "concept": "a dynasty unraveling on a floating city"},
        ),
        (
            f"/v1/studio/projects/{project_id}/immersive-videos",
            {"concept": "an atmospheric journey through a rain-soaked neon district"},
        ),
    ]

    for url, payload in cases:
        response = client.post(url, json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "completed"


def test_series_intro_runner_creates_plan_audio_and_final_video(tmp_path, monkeypatch):
    configure_studio_dirs(tmp_path, monkeypatch)
    project_id = create_project()
    seed_task("task-intro")
    monkeypatch.setattr(cinematic, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(studio, "_require_backend", lambda kind: "http://127.0.0.1:8014")
    monkeypatch.setattr(comfy_client, "workflow_template_ready", lambda name: True)

    observed = {}

    async def fake_build_shot_assets(task_id, project_id, label, shots, profile_name, use_i2v, quality_preset="premium", video_backend_name="video"):
        observed["use_i2v"] = use_i2v
        observed["video_backend_name"] = video_backend_name
        items = []
        for index, shot in enumerate(shots, start=1):
            clip_path = studio._output_path(task_id, f"scene-{index}.mp4")
            touch_file(clip_path)
            storyboard_path = studio._output_path(task_id, f"scene-{index}.png")
            touch_file(storyboard_path)
            items.append(
                {
                    "index": index,
                    "label": shot.label or f"scene-{index}",
                    "storyboard_url": studio._relative_output_url(storyboard_path),
                    "clip_url": studio._relative_output_url(clip_path),
                    "clip_path": clip_path,
                    "animation_mode": "i2v",
                    "duration_sec": shot.duration_sec,
                }
            )
        return items

    async def fake_synthesize_voice(text, voice, output_path):
        touch_file(output_path)
        return studio._relative_output_url(output_path)

    async def fake_music(task_id, prompt, duration_sec, output_name):
        output_path = studio._output_path(task_id, output_name)
        touch_file(output_path)
        return {"path": output_path, "url": studio._relative_output_url(output_path)}

    async def fake_visual(task_id, clip_paths):
        output_path = studio._output_path(task_id, "visual-sequence.mp4")
        touch_file(output_path)
        return {"path": output_path, "url": studio._relative_output_url(output_path), "duration_sec": 24.0}

    async def fake_mix(task_id, narration_path, music_path, music_volume=0.28):
        output_path = studio._output_path(task_id, "mixed-audio.m4a")
        touch_file(output_path)
        return output_path

    async def fake_finalize(task_id, profile_name, visual_path, audio_path):
        output_path = studio._output_path(task_id, "final-cinematic-cut.mp4")
        touch_file(output_path)
        return {"path": output_path, "url": studio._relative_output_url(output_path), "duration_sec": 60.0}

    monkeypatch.setattr(cinematic, "_build_shot_assets", fake_build_shot_assets)
    monkeypatch.setattr(cinematic, "_synthesize_voice", fake_synthesize_voice)
    monkeypatch.setattr(cinematic, "_generate_music_track", fake_music)
    monkeypatch.setattr(cinematic, "_assemble_visual_sequence", fake_visual)
    monkeypatch.setattr(cinematic, "_mix_or_select_audio", fake_mix)
    monkeypatch.setattr(cinematic, "_finalize_video", fake_finalize)

    result = asyncio.run(
        cinematic._run_series_intro_task(
            "task-intro",
            {
                "project_id": project_id,
                "title": "Glass Kingdom",
                "concept": "a dynasty unraveling on a floating city",
                "scene_count": 2,
                "duration_sec": 24.0,
                "label": "series-intro",
            },
        )
    )

    assert result["title"] == "Glass Kingdom"
    assert result["motion_strategy"] == "t2v_first"
    assert observed["use_i2v"] is False
    assert observed["video_backend_name"] == "video_premium"
    assert result["final_video_url"].endswith("final-cinematic-cut.mp4")
    assert len(result["shots"]) == 10
    project = studio._load_project(project_id)
    assert any(asset["label"] == "series-intro" for asset in project["assets"])


def test_immersive_runner_falls_back_to_t2v_clips(tmp_path, monkeypatch):
    configure_studio_dirs(tmp_path, monkeypatch)
    project_id = create_project()
    seed_task("task-immersive")
    monkeypatch.setattr(cinematic, "ffmpeg_available", lambda: True)

    async def fake_storyboard(task_id, shot, index, profile_name, quality_preset="premium"):
        image_path = studio._output_path(task_id, f"movement-{index}.png")
        touch_file(image_path)
        return {"path": image_path, "url": studio._relative_output_url(image_path)}

    async def fake_i2v(task_id, shot, index, image_path, profile_name, quality_preset="premium"):
        raise RuntimeError("simulated i2v failure")

    async def fake_t2v(task_id, shot, index, profile_name, quality_preset="premium", backend_name="video"):
        clip_path = studio._output_path(task_id, f"movement-{index}.mp4")
        touch_file(clip_path)
        return {"path": clip_path, "url": studio._relative_output_url(clip_path), "mode": "t2v"}

    async def fake_visual(task_id, clip_paths):
        output_path = studio._output_path(task_id, "visual-sequence.mp4")
        touch_file(output_path)
        return {"path": output_path, "url": studio._relative_output_url(output_path), "duration_sec": 18.0}

    async def fake_music(task_id, prompt, duration_sec, output_name):
        output_path = studio._output_path(task_id, output_name)
        touch_file(output_path)
        return {"path": output_path, "url": studio._relative_output_url(output_path)}

    async def fake_finalize(task_id, profile_name, visual_path, audio_path):
        output_path = studio._output_path(task_id, "final-cinematic-cut.mp4")
        touch_file(output_path)
        return {"path": output_path, "url": studio._relative_output_url(output_path), "duration_sec": 18.0}

    monkeypatch.setattr(cinematic, "_generate_storyboard_image", fake_storyboard)
    monkeypatch.setattr(cinematic, "_generate_i2v_clip", fake_i2v)
    monkeypatch.setattr(cinematic, "_generate_t2v_clip", fake_t2v)
    monkeypatch.setattr(cinematic, "_assemble_visual_sequence", fake_visual)
    monkeypatch.setattr(cinematic, "_generate_music_track", fake_music)
    monkeypatch.setattr(cinematic, "_finalize_video", fake_finalize)

    result = asyncio.run(
        cinematic._run_immersive_video_task(
            "task-immersive",
            {
                "project_id": project_id,
                "concept": "an atmospheric journey through a rain-soaked neon district",
                "scene_count": 1,
                "duration_sec": 18.0,
                "label": "immersive-video",
            },
        )
    )

    assert result["concept"].startswith("an atmospheric journey")
    assert result["shots"][0]["animation_mode"] == "t2v"
    assert result["final_video_url"].endswith("final-cinematic-cut.mp4")


def test_cinematic_director_routes_prompt_only_requests_to_prompt_guided(tmp_path, monkeypatch):
    configure_studio_dirs(tmp_path, monkeypatch)
    project_id = create_project()
    seed_task("task-router-prompt")
    monkeypatch.setattr(cinematic, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(studio, "_require_backend", lambda kind: "http://127.0.0.1:8014")
    monkeypatch.setattr(comfy_client, "workflow_template_ready", lambda name: True)

    observed = {}

    async def fake_build_shot_assets(task_id, project_id, label, shots, profile_name, use_i2v, quality_preset="premium", video_backend_name="video"):
        observed["video_backend_name"] = video_backend_name
        clip_path = studio._output_path(task_id, "prompt-shot.mp4")
        still_path = studio._output_path(task_id, "prompt-shot.png")
        touch_file(clip_path)
        touch_file(still_path)
        return [
            {
                "index": 1,
                "label": "prompt-shot",
                "storyboard_url": studio._relative_output_url(still_path),
                "clip_url": studio._relative_output_url(clip_path),
                "clip_path": clip_path,
                "animation_mode": "t2v",
                "duration_sec": 3.0,
            }
        ]

    async def fake_image_guided(*args, **kwargs):
        raise AssertionError("image-guided path should not be used for prompt-only requests")

    async def fake_synthesize_voice(text, voice, output_path):
        touch_file(output_path)
        return studio._relative_output_url(output_path)

    async def fake_music(task_id, prompt, duration_sec, output_name):
        output_path = studio._output_path(task_id, output_name)
        touch_file(output_path)
        return {"path": output_path, "url": studio._relative_output_url(output_path)}

    async def fake_visual(task_id, clip_paths):
        output_path = studio._output_path(task_id, "visual-sequence.mp4")
        touch_file(output_path)
        return {"path": output_path, "url": studio._relative_output_url(output_path), "duration_sec": 12.0}

    async def fake_mix(task_id, narration_path, music_path, music_volume=0.28):
        output_path = studio._output_path(task_id, "mixed-audio.m4a")
        touch_file(output_path)
        return output_path

    async def fake_finalize(task_id, profile_name, visual_path, audio_path):
        output_path = studio._output_path(task_id, "final-cinematic-cut.mp4")
        touch_file(output_path)
        return {"path": output_path, "url": studio._relative_output_url(output_path), "duration_sec": 12.0}

    monkeypatch.setattr(cinematic, "_build_shot_assets", fake_build_shot_assets)
    monkeypatch.setattr(cinematic, "_build_image_guided_shot_assets", fake_image_guided)
    monkeypatch.setattr(cinematic, "_synthesize_voice", fake_synthesize_voice)
    monkeypatch.setattr(cinematic, "_generate_music_track", fake_music)
    monkeypatch.setattr(cinematic, "_assemble_visual_sequence", fake_visual)
    monkeypatch.setattr(cinematic, "_mix_or_select_audio", fake_mix)
    monkeypatch.setattr(cinematic, "_finalize_video", fake_finalize)

    result = asyncio.run(
        cinematic._run_cinematic_director_task(
            "task-router-prompt",
            {
                "project_id": project_id,
                "concept": "a city collapses under a broken sun",
                "production_type": "auto",
                "duration_sec": 12.0,
                "scene_count": 4,
                "use_i2v": False,
                "label": "cinematic-director",
            },
        )
    )

    assert result["routing_mode"] == "prompt_guided"
    assert result["production_type"] == "trailer"
    assert result["video_backend_name"] == "video_premium"
    assert observed["video_backend_name"] == "video_premium"
    assert result["shots"][0]["animation_mode"] == "t2v"


def test_cinematic_director_routes_image_requests_to_image_guided(tmp_path, monkeypatch):
    configure_studio_dirs(tmp_path, monkeypatch)
    project_id = create_project()
    seed_task("task-router-image")
    monkeypatch.setattr(cinematic, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(studio, "_require_backend", lambda kind: "http://127.0.0.1:8014")
    monkeypatch.setattr(comfy_client, "workflow_template_ready", lambda name: True)

    async def fake_prompt_guided(*args, **kwargs):
        raise AssertionError("prompt-guided path should not be used when source images are provided")

    observed = {}

    async def fake_image_guided(task_id, project_id, label, shots, profile_name, source_image_urls, use_i2v, quality_preset="premium", video_backend_name="video"):
        observed["use_i2v"] = use_i2v
        observed["video_backend_name"] = video_backend_name
        clip_path = studio._output_path(task_id, "guided-shot.mp4")
        still_path = studio._output_path(task_id, "guided-shot.png")
        touch_file(clip_path)
        touch_file(still_path)
        return [
            {
                "index": 1,
                "label": "guided-shot",
                "storyboard_url": studio._relative_output_url(still_path),
                "clip_url": studio._relative_output_url(clip_path),
                "clip_path": clip_path,
                "animation_mode": "t2v",
                "duration_sec": 3.0,
            }
        ]

    async def fake_synthesize_voice(text, voice, output_path):
        touch_file(output_path)
        return studio._relative_output_url(output_path)

    async def fake_music(task_id, prompt, duration_sec, output_name):
        output_path = studio._output_path(task_id, output_name)
        touch_file(output_path)
        return {"path": output_path, "url": studio._relative_output_url(output_path)}

    async def fake_visual(task_id, clip_paths):
        output_path = studio._output_path(task_id, "visual-sequence.mp4")
        touch_file(output_path)
        return {"path": output_path, "url": studio._relative_output_url(output_path), "duration_sec": 12.0}

    async def fake_mix(task_id, narration_path, music_path, music_volume=0.28):
        output_path = studio._output_path(task_id, "mixed-audio.m4a")
        touch_file(output_path)
        return output_path

    async def fake_finalize(task_id, profile_name, visual_path, audio_path):
        output_path = studio._output_path(task_id, "final-cinematic-cut.mp4")
        touch_file(output_path)
        return {"path": output_path, "url": studio._relative_output_url(output_path), "duration_sec": 12.0}

    monkeypatch.setattr(cinematic, "_build_shot_assets", fake_prompt_guided)
    monkeypatch.setattr(cinematic, "_build_image_guided_shot_assets", fake_image_guided)
    monkeypatch.setattr(cinematic, "_synthesize_voice", fake_synthesize_voice)
    monkeypatch.setattr(cinematic, "_generate_music_track", fake_music)
    monkeypatch.setattr(cinematic, "_assemble_visual_sequence", fake_visual)
    monkeypatch.setattr(cinematic, "_mix_or_select_audio", fake_mix)
    monkeypatch.setattr(cinematic, "_finalize_video", fake_finalize)

    result = asyncio.run(
        cinematic._run_cinematic_director_task(
            "task-router-image",
            {
                "project_id": project_id,
                "concept": "a kingdom at war in the clouds",
                "production_type": "auto",
                "title": "Cloudfall",
                "source_image_urls": ["http://127.0.0.1:9000/outputs/reference/frame1.png"],
                "duration_sec": 12.0,
                "scene_count": 4,
                "use_i2v": True,
                "label": "cinematic-director",
            },
        )
    )

    assert result["routing_mode"] == "image_guided"
    assert result["motion_strategy"] == "t2v_first"
    assert result["video_backend_name"] == "video_premium"
    assert result["source_image_count"] == 1
    assert observed["use_i2v"] is False
    assert observed["video_backend_name"] == "video_premium"
    assert result["shots"][0]["animation_mode"] == "t2v"


def test_series_intro_quality_gate_rejects_hold_fallback(tmp_path, monkeypatch):
    configure_studio_dirs(tmp_path, monkeypatch)
    project_id = create_project()
    seed_task("task-intro-hold")
    monkeypatch.setattr(cinematic, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(studio, "_require_backend", lambda kind: "http://127.0.0.1:8014")
    monkeypatch.setattr(comfy_client, "workflow_template_ready", lambda name: True)

    async def fake_build_shot_assets(task_id, project_id, label, shots, profile_name, use_i2v, quality_preset="premium", video_backend_name="video"):
        clip_path = studio._output_path(task_id, "scene-1.mp4")
        still_path = studio._output_path(task_id, "scene-1.png")
        touch_file(clip_path)
        touch_file(still_path)
        return [
            {
                "index": 1,
                "label": "scene-1",
                "storyboard_url": studio._relative_output_url(still_path),
                "clip_url": studio._relative_output_url(clip_path),
                "clip_path": clip_path,
                "animation_mode": "hold",
                "duration_sec": 6.0,
            }
        ]

    monkeypatch.setattr(cinematic, "_build_shot_assets", fake_build_shot_assets)

    try:
        asyncio.run(
            cinematic._run_series_intro_task(
                "task-intro-hold",
                {
                    "project_id": project_id,
                    "title": "Glass Kingdom",
                    "concept": "a dynasty unraveling on a floating city",
                    "label": "series-intro",
                },
            )
        )
    except RuntimeError as exc:
        assert "quality floor" in str(exc)
    else:
        raise AssertionError("expected the cinematic quality gate to reject hold-only intro output")


def test_default_intro_prompts_avoid_embedded_title_text():
    req = cinematic.SeriesIntroRequest(
        title="Empire of Glass",
        concept="after the sun shatters, rival bloodlines wage a cold holy war above a storm-wrapped ocean of clouds",
    )

    shots = cinematic._default_intro_shots(req)

    assert shots
    assert all("title reveal shot" not in shot.image_prompt.lower() for shot in shots)
    assert any("no text" in shot.image_prompt.lower() for shot in shots)
    assert "later title overlay" in shots[-1].image_prompt.lower()


def test_video_generation_candidates_raise_native_resolution_before_fallback():
    assert cinematic._video_generation_candidates("cinematic_wide", "premium") == [
        (1280, 704),
        (1152, 640),
        (1024, 576),
    ]
    assert cinematic._video_generation_candidates("cinematic_wide", "ultra") == [
        (1536, 896),
        (1280, 704),
        (1152, 640),
        (1024, 576),
    ]


def test_should_enhance_i2v_source_for_premium_and_ultra():
    assert cinematic._should_enhance_i2v_source("premium") is True
    assert cinematic._should_enhance_i2v_source("ultra") is True
    assert cinematic._should_enhance_i2v_source("standard") is False


def test_i2v_conditioning_strength_relaxes_cinematic_source_lock():
    assert cinematic._i2v_conditioning_strength("ultra") == 0.72
    assert cinematic._i2v_conditioning_strength("premium") == 0.82
    assert cinematic._i2v_conditioning_strength("standard") == 1.0


def test_prefer_t2v_motion_for_premium_series_intro_and_trailer():
    assert cinematic._prefer_t2v_motion(True, "series_intro", "premium") is True
    assert cinematic._prefer_t2v_motion(True, "trailer", "ultra") is True
    assert cinematic._prefer_t2v_motion(True, "immersive_video", "premium") is False
    assert cinematic._prefer_t2v_motion(False, "series_intro", "premium") is True


def test_premium_video_backend_ready_rejects_local_ltx_for_premium_cinematic(monkeypatch):
    monkeypatch.setattr(studio, "_require_backend", lambda kind: "http://127.0.0.1:8004")

    assert cinematic._video_backend_is_premium_ready("immersive_video", "premium") is True
    assert cinematic._video_backend_is_premium_ready("series_intro", "standard") is True
    assert cinematic._video_backend_is_premium_ready("series_intro", "premium") is False


def test_series_intro_runner_rejects_premium_render_on_local_ltx(tmp_path, monkeypatch):
    configure_studio_dirs(tmp_path, monkeypatch)
    project_id = create_project()
    seed_task("task-intro-blocked")
    monkeypatch.setattr(cinematic, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(studio, "_require_backend", lambda kind: "http://127.0.0.1:8004")

    try:
        asyncio.run(
            cinematic._run_series_intro_task(
                "task-intro-blocked",
                {
                    "project_id": project_id,
                    "title": "Glass Kingdom",
                    "concept": "a dynasty unraveling on a floating city",
                    "label": "series-intro",
                },
            )
        )
    except RuntimeError as exc:
        assert "Premium cinematic render blocked" in str(exc)
    else:
        raise AssertionError("expected premium series intro to be blocked on the local LTX backend")


def test_ultra_detail_post_pass_only_runs_for_ultra():
    assert cinematic._should_apply_detail_post_pass("ultra") is True
    assert cinematic._should_apply_detail_post_pass("premium") is False


def test_i2v_generation_candidates_avoid_1536_retry_dead_end():
    assert cinematic._i2v_generation_candidates("cinematic_wide", "ultra") == [
        (1280, 704),
        (1152, 640),
        (1024, 576),
    ]


def test_premium_clip_frame_count_uses_longer_native_motion():
    assert cinematic._clip_frame_count(5.0, 24.0, "premium") == 121


def test_long_cinematic_shots_expand_instead_of_looping():
    shots = [
        cinematic.CinematicShot(
            label="scene-1",
            image_prompt="storm throne",
            motion_prompt="slow reveal",
            duration_sec=6.0,
        )
    ]

    expanded = cinematic._expand_shots_for_motion_budget(shots, "premium")

    assert len(expanded) == 2
    assert expanded[0].label == "scene-1-a"
    assert expanded[1].label == "scene-1-b"
    assert round(expanded[0].duration_sec, 2) == 3.0
    assert round(expanded[1].duration_sec, 2) == 3.0


def test_finalize_video_extends_short_audio_to_visual_duration(tmp_path, monkeypatch):
    configure_studio_dirs(tmp_path, monkeypatch)
    visual_path = studio._output_path("task-finalize", "visual-sequence.mp4")
    audio_path = studio._output_path("task-finalize", "mixed-audio.m4a")
    touch_file(visual_path)
    touch_file(audio_path)

    calls = {"looped_audio": None, "mux_audio": None}

    def fake_detect_duration(path):
        return 8.0

    def fake_get_media_duration(path):
        return 2.8 if path.endswith("mixed-audio.m4a") else 8.0

    def fake_loop_audio(audio_in, target_duration, output_path):
        calls["looped_audio"] = output_path
        touch_file(output_path)
        return True

    def fake_mux(video_in, audio_in, output_path):
        calls["mux_audio"] = audio_in
        touch_file(output_path)
        return True

    monkeypatch.setattr(cinematic, "detect_duration", fake_detect_duration)
    monkeypatch.setattr(cinematic, "get_media_duration", fake_get_media_duration)
    monkeypatch.setattr(cinematic, "loop_audio_to_duration", fake_loop_audio)
    monkeypatch.setattr(cinematic, "mux_video_and_audio", fake_mux)

    result = asyncio.run(cinematic._finalize_video("task-finalize", "cinematic_wide", visual_path, audio_path))

    assert result["duration_sec"] == 8.0
    assert calls["looped_audio"] is not None
    assert calls["mux_audio"] == calls["looped_audio"]
