import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import orchestrator.comfy_client as comfy_client


def test_render_workflow_applies_manifest_placeholders(tmp_path, monkeypatch):
    workflow_dir = tmp_path / "comfy_workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(comfy_client, "COMFY_WORKFLOWS_DIR", str(workflow_dir))

    workflow = {
        "10": {"inputs": {"text": "placeholder"}},
        "20": {"inputs": {"width": 640, "height": 360}},
        "30": {"inputs": {"steps": 20, "seed": 1}},
    }
    manifest = {
        "placeholders": {
            "prompt": ["10", "inputs", "text"],
            "width": ["20", "inputs", "width"],
            "height": ["20", "inputs", "height"],
            "steps": ["30", "inputs", "steps"],
        }
    }

    (workflow_dir / "premium_ltx23_t2v.workflow_api.json").write_text(json.dumps(workflow), encoding="utf-8")
    (workflow_dir / "premium_ltx23_t2v.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    rendered, loaded_manifest = comfy_client.render_workflow(
        "premium_ltx23_t2v",
        {"prompt": "glass kingdom", "width": 1280, "height": 704, "steps": 28},
    )

    assert loaded_manifest["placeholders"]["prompt"] == ["10", "inputs", "text"]
    assert rendered["10"]["inputs"]["text"] == "glass kingdom"
    assert rendered["20"]["inputs"]["width"] == 1280
    assert rendered["20"]["inputs"]["height"] == 704
    assert rendered["30"]["inputs"]["steps"] == 28


def test_extract_outputs_prefers_manifest_order():
    history = {
        "outputs": {
            "99": {
                "images": [{"filename": "fallback.png", "subfolder": "", "type": "output"}],
            },
            "200": {
                "videos": [{"filename": "premium.mp4", "subfolder": "video", "type": "output"}],
            },
        }
    }
    manifest = {"output_node_ids": ["200"], "output_keys": ["videos", "images"]}

    files = comfy_client._extract_outputs(history, manifest)

    assert files[0]["filename"] == "premium.mp4"
    assert files[0]["kind"] == "videos"
    assert files[1]["filename"] == "fallback.png"


def test_build_view_url_encodes_file_info():
    url = comfy_client.build_view_url(
        "http://127.0.0.1:8188",
        {"filename": "premium.mp4", "subfolder": "video/out", "type": "output"},
    )

    assert url.startswith("http://127.0.0.1:8188/view?")
    assert "filename=premium.mp4" in url
    assert "subfolder=video%2Fout" in url
