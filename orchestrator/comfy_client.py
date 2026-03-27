import copy
import asyncio
import json
import os
import time
import uuid
from typing import Any, Optional
from urllib.parse import urlencode

import httpx


ROOT_DIR = os.path.dirname(__file__)
COMFY_WORKFLOWS_DIR = os.path.join(ROOT_DIR, "comfy_workflows")


def _template_paths(template_name: str) -> tuple[str, str]:
    workflow_path = os.path.join(COMFY_WORKFLOWS_DIR, f"{template_name}.workflow_api.json")
    manifest_path = os.path.join(COMFY_WORKFLOWS_DIR, f"{template_name}.manifest.json")
    return workflow_path, manifest_path


def workflow_template_ready(template_name: str) -> bool:
    workflow_path, manifest_path = _template_paths(template_name)
    return os.path.exists(workflow_path) and os.path.exists(manifest_path)


def load_workflow_template(template_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    workflow_path, manifest_path = _template_paths(template_name)
    if not os.path.exists(workflow_path):
        raise FileNotFoundError(f"Missing ComfyUI workflow template: {workflow_path}")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing ComfyUI workflow manifest: {manifest_path}")

    with open(workflow_path, "r", encoding="utf-8") as handle:
        workflow = json.load(handle)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return workflow, manifest


def _normalize_path(path: list[Any]) -> list[Any]:
    return [str(part) if isinstance(part, int) else part for part in path]


def _set_path_value(payload: Any, path: list[Any], value: Any) -> None:
    current = payload
    normalized = _normalize_path(path)
    for part in normalized[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    last = normalized[-1]
    if isinstance(current, list):
        current[int(last)] = value
    else:
        current[last] = value


def render_workflow(template_name: str, values: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    workflow, manifest = load_workflow_template(template_name)
    rendered = copy.deepcopy(workflow)
    placeholders = dict(manifest.get("placeholders") or {})
    required = set(manifest.get("required_placeholders") or placeholders.keys())

    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"Missing ComfyUI workflow placeholders: {', '.join(sorted(missing))}")

    for key, path in placeholders.items():
        if key not in values:
            continue
        _set_path_value(rendered, path, values[key])
    return rendered, manifest


def build_view_url(base_url: str, file_info: dict[str, Any]) -> str:
    params = {
        "filename": file_info.get("filename", ""),
        "subfolder": file_info.get("subfolder", ""),
        "type": file_info.get("type", "output"),
    }
    return f"{base_url.rstrip('/')}/view?{urlencode(params)}"


def _extract_outputs(history_entry: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = history_entry.get("outputs") or {}
    preferred_nodes = [str(node_id) for node_id in (manifest.get("output_node_ids") or [])]
    ordered_nodes = preferred_nodes + [node_id for node_id in outputs.keys() if node_id not in preferred_nodes]
    file_sets = manifest.get("output_keys") or ["videos", "gifs", "images"]

    collected: list[dict[str, Any]] = []
    for node_id in ordered_nodes:
        node_outputs = outputs.get(str(node_id)) or {}
        for key in file_sets:
            items = node_outputs.get(key) or []
            for item in items:
                enriched = dict(item)
                enriched.setdefault("kind", key)
                enriched.setdefault("node_id", str(node_id))
                collected.append(enriched)
    return collected


async def run_workflow(
    base_url: str,
    template_name: str,
    values: dict[str, Any],
    output_path: Optional[str] = None,
    poll_interval_sec: float = 2.0,
    timeout_sec: float = 1800.0,
) -> dict[str, Any]:
    workflow, manifest = render_workflow(template_name, values)
    client_id = uuid.uuid4().hex
    started_at = time.time()
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/prompt",
            json={"prompt": workflow, "client_id": client_id},
        )
        response.raise_for_status()
        payload = response.json() or {}
        prompt_id = payload.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return a prompt_id: {payload}")

        history_entry = None
        last_history = None
        while time.time() - started_at <= timeout_sec:
            history_response = await client.get(f"{base_url.rstrip('/')}/history/{prompt_id}")
            history_response.raise_for_status()
            history_payload = history_response.json() or {}
            last_history = history_payload
            history_entry = history_payload.get(prompt_id) or history_payload.get(str(prompt_id))
            if history_entry:
                status = ((history_entry.get("status") or {}).get("status_str") or "").lower()
                outputs = history_entry.get("outputs") or {}
                if outputs:
                    break
                if status in {"error", "failed"}:
                    raise RuntimeError(f"ComfyUI workflow failed: {history_entry}")
            await client.get(f"{base_url.rstrip('/')}/queue")
            await asyncio.sleep(max(float(poll_interval_sec), 0.5))

        if not history_entry:
            raise RuntimeError(f"ComfyUI workflow did not complete before timeout: {last_history}")

        files = _extract_outputs(history_entry, manifest)
        if not files:
            raise RuntimeError(f"ComfyUI workflow completed without usable outputs: {history_entry}")

        first_file = files[0]
        view_url = build_view_url(base_url, first_file)
        if output_path:
            download_response = await client.get(view_url)
            download_response.raise_for_status()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as handle:
                handle.write(download_response.content)

        return {
            "prompt_id": prompt_id,
            "history": history_entry,
            "files": files,
            "primary_file": first_file,
            "primary_url": view_url,
            "downloaded_path": output_path,
        }
