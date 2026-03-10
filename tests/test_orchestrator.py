import asyncio
import os
import sys

import httpx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


async def test_orchestrator():
    orchestrator_url = f"http://127.0.0.1:{config.ORCHESTRATOR_PORT}"
    print(f"Testing Orchestrator at {orchestrator_url}...")

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            # 1. Test explicit health endpoint
            resp = await client.get(f"{orchestrator_url}/health")
            if resp.status_code == 200:
                print(f"[PASS] Orchestrator is reachable. Registry: {resp.json().get('registry')}")
            else:
                print(f"[FAIL] Orchestrator Health Check Failed: {resp.status_code}")
                return

            # 2. Test aggregated model listing
            print("Testing hub model listing...")
            resp = await client.get(f"{orchestrator_url}/v1/models")
            if resp.status_code == 200:
                print(f"[PASS] Hub model listing responded. Models: {len(resp.json().get('data', []))}")
            else:
                print(f"[FAIL] Expected 200 but got {resp.status_code} - {resp.text}")

            # 3. Test TTS voices route
            print("Testing TTS voices route...")
            resp = await client.get(f"{orchestrator_url}/v1/audio/voices")
            if resp.status_code in [200, 502]:
                print(f"[PASS] Voices route matched. Status: {resp.status_code}")
            else:
                print(f"[FAIL] Expected 200/502 but got {resp.status_code} - {resp.text}")

            # 4. Test music stats route
            print("Testing music stats route...")
            resp = await client.get(f"{orchestrator_url}/v1/stats")
            if resp.status_code in [200, 502]:
                print(f"[PASS] Stats route matched. Status: {resp.status_code}")
            else:
                print(f"[FAIL] Expected 200/502 but got {resp.status_code} - {resp.text}")

            # 5. Test TTS routing
            print("Testing TTS proxy route...")
            tts_payload = {"input": "Testing the AI-Hub architecture!", "voice": "Vivian"}
            resp = await client.post(f"{orchestrator_url}/v1/audio/speech", json=tts_payload)
            if resp.status_code in [200, 502, 422]:
                print(f"[PASS] Proxy routed to TTS. Status: {resp.status_code}")
            else:
                print(f"[FAIL] Expected 200/502/422 but got {resp.status_code} - {resp.text}")

            # 6. Test music routing
            print("Testing Music proxy route...")
            music_payload = {"prompt": "A short cinematic piano cue.", "audio_duration": 10, "thinking": True}
            resp = await client.post(f"{orchestrator_url}/v1/audio/async_generations", json=music_payload)
            if resp.status_code in [200, 500, 502, 422]:
                print(f"[PASS] Proxy routed to Music. Status: {resp.status_code}")
            else:
                print(f"[FAIL] Expected 200/500/502/422 but got {resp.status_code} - {resp.text}")

            # 7. Test vision routing
            print("Testing Vision proxy route...")
            vision_payload = {"prompt": "A magical orchestrator", "cfg_normalization": True}
            resp = await client.post(f"{orchestrator_url}/v1/images/async_generate", json=vision_payload)
            if resp.status_code in [200, 502, 422]:
                print(f"[PASS] Proxy routed to Vision. Status: {resp.status_code}")
            else:
                print(f"[FAIL] Expected 200/502/422 but got {resp.status_code} - {resp.text}")

            # 8. Test video routing
            print("Testing Video proxy route...")
            video_payload = {
                "prompt": "A quick camera move over a futuristic skyline.",
                "height": 256,
                "width": 256,
                "num_frames": 17,
                "frame_rate": 8.0,
                "num_inference_steps": 2,
                "seed": 42,
            }
            resp = await client.post(f"{orchestrator_url}/v1/video/async_t2v", json=video_payload)
            if resp.status_code in [200, 500, 502, 422]:
                print(f"[PASS] Proxy routed to Video. Status: {resp.status_code}")
            else:
                print(f"[FAIL] Expected 200/500/502/422 but got {resp.status_code} - {resp.text}")

            # 9. Test Director workflow
            print("Testing Director workflow...")
            worker_payload = {
                "image_prompt": "A cyberpunk city at night with neon lights and flying cars.",
                "voice": "Vivian",
                "voiceover_text": "Welcome to Neo-Tokyo. The air here tastes like battery acid and broken dreams.",
            }
            resp = await client.post(f"{orchestrator_url}/v1/workflows/director", json=worker_payload, timeout=600.0)
            if resp.status_code in [200, 502]:
                print(f"[PASS] Director workflow route responded as expected. Status: {resp.status_code}")
                if resp.status_code == 200:
                    print(f"  Final output created at: {resp.json().get('output_url')}")
            else:
                print(f"[FAIL] Director Workflow failed: {resp.status_code} - {resp.text}")

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[ERROR] Connection failed context: {repr(e)}")


if __name__ == "__main__":
    asyncio.run(test_orchestrator())
