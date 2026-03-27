# ComfyUI Premium Video Workflows

This directory is intentionally separate from the existing local video backend.

For premium cinematic `series_intro` and `trailer` jobs, the orchestrator now
looks for a ComfyUI backend named `video_premium` and an exported API-format
workflow template here.

Required files for the default premium path:

- `premium_ltx23_t2v.workflow_api.json`
- `premium_ltx23_t2v.manifest.json`

How to prepare them:

1. Open ComfyUI with the official `ComfyUI-LTXVideo` nodes installed.
2. Load or build the T2V workflow you want to use for premium cinematic shots.
3. Save the workflow using `Save (API Format)`.
4. Put that JSON here as `premium_ltx23_t2v.workflow_api.json`.
5. Create `premium_ltx23_t2v.manifest.json` using the example file in this folder.

The manifest tells the orchestrator which node inputs should be patched at
runtime and which output nodes to download from.

Until both files exist and `video_premium` is configured in `models.yaml`,
premium cinematic renders will fail fast instead of quietly using draft-grade
video output.
