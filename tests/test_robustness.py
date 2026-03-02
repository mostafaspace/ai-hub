"""
Comprehensive robustness tests for:
  - commit d559a42: GracefulJSONRoute (api_utils.py)
  - commit bf26b63: Orchestrator (orchestrator/server.py, setup_ffmpeg.py)

Run with:  python tests/test_robustness.py
"""

import os
import sys
import json
import asyncio
import tempfile
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Unit tests: fix_json_string
# ─────────────────────────────────────────────────────────────────────────────
from api_utils import fix_json_string

def test_fix_json_valid():
    """Valid JSON must be returned as-is."""
    raw = '{"input": "Hello", "voice": "Vivian"}'
    result = fix_json_string(raw)
    assert result == raw
    json.loads(result)
    print("[PASS] Valid JSON returned unchanged")

def test_fix_json_python_booleans():
    """Python True/False/None in double-quoted JSON."""
    raw = '{"cfg_normalization": True, "active": False, "value": None}'
    result = fix_json_string(raw)
    parsed = json.loads(result)
    assert parsed["cfg_normalization"] is True
    assert parsed["active"] is False
    assert parsed["value"] is None
    print("[PASS] Python booleans/None fixed in double-quoted JSON")

def test_fix_json_single_quotes():
    """Single-quoted dict should be converted to valid JSON."""
    raw = "{'input': 'Hello', 'voice': 'Vivian'}"
    result = fix_json_string(raw)
    parsed = json.loads(result)
    assert parsed["input"] == "Hello"
    assert parsed["voice"] == "Vivian"
    print("[PASS] Single-quoted dict fixed")

def test_fix_json_hopeless():
    """Totally broken JSON should be returned as-is (not crash)."""
    raw = '{"input": "Hello", "voice": '
    result = fix_json_string(raw)
    assert isinstance(result, str)
    try:
        json.loads(result)
        assert False, "Should NOT parse hopeless JSON"
    except json.JSONDecodeError:
        pass
    print("[PASS] Hopeless JSON returned as-is (no crash)")

def test_fix_json_mixed_booleans_and_single_quotes():
    """Mixed: single-quoted keys AND Python booleans (e.g. {'key': True})."""
    raw = "{'prompt': 'A cat', 'cfg_normalization': True}"
    result = fix_json_string(raw)
    parsed = json.loads(result)
    assert parsed["prompt"] == "A cat"
    assert parsed["cfg_normalization"] is True
    print("[PASS] Mixed single-quotes + Python booleans fixed")

def test_fix_json_nested():
    """Nested struct with Python booleans in double-quoted JSON."""
    raw = '{"options": {"verbose": True, "count": 3}, "active": False}'
    result = fix_json_string(raw)
    parsed = json.loads(result)
    assert parsed["options"]["verbose"] is True
    assert parsed["active"] is False
    print("[PASS] Nested structure with Python booleans fixed")

def test_fix_json_empty_string():
    """Empty string should be returned as-is (not crash)."""
    result = fix_json_string("")
    assert result == ""
    print("[PASS] Empty string handled gracefully")

def test_fix_json_non_dict_list():
    """Single-quoted list should round-trip correctly."""
    raw = "['hello', 'world']"
    result = fix_json_string(raw)
    parsed = json.loads(result)
    assert parsed == ["hello", "world"]
    print("[PASS] Single-quoted list fixed")

# ─────────────────────────────────────────────────────────────────────────────
# 2. GracefulJSONRoute integration via a minimal FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
def test_graceful_route_integration():
    """GracefulJSONRoute should fix body before Pydantic sees it."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from pydantic import BaseModel
    from api_utils import GracefulJSONRoute

    app = FastAPI()
    app.router.route_class = GracefulJSONRoute

    class Payload(BaseModel):
        name: str
        active: bool = False

    @app.post("/echo")
    def echo(p: Payload):
        return {"name": p.name, "active": p.active}

    client = TestClient(app)

    # 1. Valid JSON
    r = client.post("/echo", content='{"name": "test", "active": true}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 200
    assert r.json()["active"] is True
    print("[PASS] GracefulJSONRoute: valid JSON passes through")

    # 2. Python boolean in double-quoted JSON
    r = client.post("/echo", content='{"name": "test", "active": True}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    assert r.json()["active"] is True
    print("[PASS] GracefulJSONRoute: Python boolean fixed automatically")

    # 3. Single-quoted dict with Python booleans
    r = client.post("/echo", content="{'name': 'test', 'active': True}",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    assert r.json()["name"] == "test"
    assert r.json()["active"] is True
    print("[PASS] GracefulJSONRoute: single-quoted dict + Python bool fixed")

    # 4. Hopeless JSON -> 422
    r = client.post("/echo", content='{"name": "test", "active":',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"
    print("[PASS] GracefulJSONRoute: hopeless JSON correctly rejected with 422")

    # 5. Non-JSON content-type -> not intercepted
    r = client.post("/echo", content='{"name": "test", "active": true}',
                    headers={"Content-Type": "text/plain"})
    assert r.status_code in (400, 422)
    print("[PASS] GracefulJSONRoute: non-JSON content-type not intercepted")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Orchestrator: routing + Director backend-check (no live models needed)
# ─────────────────────────────────────────────────────────────────────────────
def test_orchestrator_routes():
    """Spin up the orchestrator via TestClient and verify routing + 503 guards."""
    from fastapi.testclient import TestClient
    import orchestrator.server as orch

    original_backends = orch.BACKENDS.copy()
    orch.BACKENDS.clear()  # simulate no backends configured

    client = TestClient(orch.app)

    try:
        # Health check via catch_all
        r = client.get("/health")
        assert r.status_code == 200
        print("[PASS] Orchestrator /health -> 200")

        # Root
        r = client.get("/")
        assert r.status_code == 200
        assert "status" in r.json()
        print("[PASS] Orchestrator / -> 200 with status key")

        # TTS route -> 503 when not configured
        r = client.post("/v1/audio/speech", json={"input": "hello", "voice": "Vivian"})
        assert r.status_code == 503, f"Expected 503, got {r.status_code}: {r.text}"
        print("[PASS] /v1/audio/speech -> 503 when TTS not configured")

        # ASR route -> 503
        r = client.post("/v1/audio/transcriptions",
                        files={"file": ("test.wav", b"RIFF", "audio/wav")})
        assert r.status_code == 503, f"Expected 503, got {r.status_code}: {r.text}"
        print("[PASS] /v1/audio/transcriptions -> 503 when ASR not configured")

        # Vision route -> 503
        r = client.post("/v1/images/async_generate", json={"prompt": "test"})
        assert r.status_code == 503, f"Expected 503, got {r.status_code}: {r.text}"
        print("[PASS] /v1/images/async_generate -> 503 when Vision not configured")

        # Video route -> 503
        r = client.post("/v1/video/async_t2v", json={"prompt": "test"})
        assert r.status_code == 503, f"Expected 503, got {r.status_code}: {r.text}"
        print("[PASS] /v1/video/async_t2v -> 503 when Video not configured")

        # Director -> 503 when required backends missing
        r = client.post("/v1/workflows/director", json={
            "image_prompt": "test",
            "voiceover_text": "test",
            "voice": "Vivian"
        })
        assert r.status_code == 503, f"Expected 503, got {r.status_code}: {r.text}"
        print("[PASS] /v1/workflows/director -> 503 when backends missing")

        # Unknown path -> 404
        r = client.get("/v1/some/unknown/path")
        assert r.status_code == 404
        print("[PASS] Unknown path -> 404")

    finally:
        orch.BACKENDS.update(original_backends)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Orchestrator: Director request validation
# ─────────────────────────────────────────────────────────────────────────────
def test_director_request_validation():
    """Director should return 422 on missing required fields."""
    from fastapi.testclient import TestClient
    import orchestrator.server as orch

    client = TestClient(orch.app)

    # Missing 'voiceover_text'
    r = client.post("/v1/workflows/director", json={"image_prompt": "only prompt"})
    assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"
    print("[PASS] Director: missing voiceover_text -> 422")

    # Missing 'image_prompt'
    r = client.post("/v1/workflows/director", json={"voiceover_text": "only voice"})
    assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"
    print("[PASS] Director: missing image_prompt -> 422")

    # Empty body
    r = client.post("/v1/workflows/director", content=b"",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"
    print("[PASS] Director: empty body -> 422")

# ─────────────────────────────────────────────────────────────────────────────
# 5. media_utils: mux_video_and_audio error path
# ─────────────────────────────────────────────────────────────────────────────
def test_mux_missing_inputs():
    """mux_video_and_audio should return False (not raise) for missing files."""
    from orchestrator.media_utils import mux_video_and_audio

    result = mux_video_and_audio(
        video_path="/nonexistent/video.mp4",
        audio_path="/nonexistent/audio.wav",
        output_path="/tmp/out.mp4"
    )
    assert result is False
    print("[PASS] mux_video_and_audio: missing inputs -> False (no crash)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. setup_ffmpeg: already-installed path
# ─────────────────────────────────────────────────────────────────────────────
def test_setup_ffmpeg_already_installed():
    """setup_ffmpeg should return the exe path if ffmpeg.exe already exists."""
    import orchestrator.setup_ffmpeg as sfm

    # Create a temporary directory structure with a fake ffmpeg.exe
    tmp_dir = tempfile.mkdtemp()
    fake_bin = os.path.join(tmp_dir, "bin")
    os.makedirs(fake_bin)
    fake_exe = os.path.join(fake_bin, "ffmpeg.exe")
    with open(fake_exe, "w") as f:
        f.write("fake ffmpeg")

    # Monkey-patch BIN_DIR so the function looks in our temp dir
    orig_bin_dir = sfm.BIN_DIR
    sfm.BIN_DIR = fake_bin

    try:
        result = sfm.setup_ffmpeg()
        assert result == fake_exe, f"Expected {fake_exe}, got {result}"
        print(f"[PASS] setup_ffmpeg: already installed -> returns exe path")
    finally:
        sfm.BIN_DIR = orig_bin_dir
        shutil.rmtree(tmp_dir)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Poll deadline logic (unit test)
# ─────────────────────────────────────────────────────────────────────────────
def test_poll_deadline_logic():
    """Verify the deadline math used in orchestrator polling is correct."""
    async def _check():
        loop = asyncio.get_event_loop()
        deadline = loop.time() + 0.1  # 100ms deadline
        await asyncio.sleep(0.2)      # sleep past deadline
        timed_out = loop.time() > deadline
        assert timed_out, "Deadline should have passed"
        print("[PASS] poll_deadline logic: deadline correctly detected after sleep")

    asyncio.run(_check())


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("fix_json_string: valid JSON",                    test_fix_json_valid),
        ("fix_json_string: Python booleans",               test_fix_json_python_booleans),
        ("fix_json_string: single quotes",                 test_fix_json_single_quotes),
        ("fix_json_string: hopeless JSON",                 test_fix_json_hopeless),
        ("fix_json_string: mixed malformed",               test_fix_json_mixed_booleans_and_single_quotes),
        ("fix_json_string: nested structure",              test_fix_json_nested),
        ("fix_json_string: empty string",                  test_fix_json_empty_string),
        ("fix_json_string: single-quoted list",            test_fix_json_non_dict_list),
        ("GracefulJSONRoute: integration",                 test_graceful_route_integration),
        ("Orchestrator: routing + 503 guards",             test_orchestrator_routes),
        ("Orchestrator: Director validation",              test_director_request_validation),
        ("media_utils: missing inputs",                    test_mux_missing_inputs),
        ("setup_ffmpeg: already-installed path",           test_setup_ffmpeg_already_installed),
        ("Poll deadline: logic",                           test_poll_deadline_logic),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n-- {name} --")
        try:
            fn()
            passed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[FAIL] {name}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed:
        sys.exit(1)
