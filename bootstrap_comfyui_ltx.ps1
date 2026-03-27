$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ComfyRoot = Join-Path $Root "ComfyUI"
$ComfyVenv = Join-Path $ComfyRoot ".venv312"
$ComfyPython = Join-Path $ComfyVenv "Scripts\python.exe"
$ComfyPip = Join-Path $ComfyVenv "Scripts\pip.exe"
$CustomNodes = Join-Path $ComfyRoot "custom_nodes"
$LtxNodeRoot = Join-Path $CustomNodes "ComfyUI-LTXVideo"
$Py312 = Join-Path $Root ".uv-python\cpython-3.12.11-windows-x86_64-none\python.exe"

if (-not (Test-Path $ComfyRoot)) {
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git $ComfyRoot
}

if (-not (Test-Path $ComfyVenv)) {
    if (Test-Path $Py312) {
        & $Py312 -m venv $ComfyVenv
    } else {
        py -3.12 -m venv $ComfyVenv
    }
}

& $ComfyPython -m pip install --upgrade pip
& $ComfyPip install --force-reinstall torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
& $ComfyPip install -r (Join-Path $ComfyRoot "requirements.txt")

if (-not (Test-Path $CustomNodes)) {
    New-Item -ItemType Directory -Path $CustomNodes | Out-Null
}

if (-not (Test-Path $LtxNodeRoot)) {
    git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git $LtxNodeRoot
}

$LtxReq = Join-Path $LtxNodeRoot "requirements.txt"
if (Test-Path $LtxReq) {
    & $ComfyPip install -r $LtxReq
}

Write-Host "ComfyUI bootstrap complete."
Write-Host "Using ComfyUI venv: $ComfyVenv"
Write-Host "Next steps:"
Write-Host "1. Launch ComfyUI once and install/load the desired LTX workflow."
Write-Host "2. Export the workflow in API format to orchestrator\\comfy_workflows\\premium_ltx23_t2v.workflow_api.json"
Write-Host "3. Copy premium_ltx23_t2v.manifest.example.json to premium_ltx23_t2v.manifest.json and update node ids."
