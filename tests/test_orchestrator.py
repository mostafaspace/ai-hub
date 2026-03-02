import httpx
import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

async def test_orchestrator():
    orchestrator_url = f"http://127.0.0.1:{config.ORCHESTRATOR_PORT}"
    print(f"Testing Orchestrator at {orchestrator_url}...")
    
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            # 1. Test Health/Root endpoint
            resp = await client.get(f"{orchestrator_url}/")
            if resp.status_code == 200:
                print(f"[PASS] Orchestrator is reachable. Registry: {resp.json().get('registry')}")
            else:
                print(f"[FAIL] Orchestrator Health Check Failed: {resp.status_code}")
                return
                
            # 2. Test TTS Routing (Assuming TTS is running, but even if not, we should get 502 Bad Gateway instead of 404 Not Found)
            print("Testing TTS proxy route...")
            tts_payload = {"input": "Testing the AI-Hub architecture!", "voice": "Vivian"}
            resp = await client.post(f"{orchestrator_url}/v1/audio/speech", json=tts_payload)
            if resp.status_code in [200, 502, 422]: 
                # 200 = Success, 502 = Proxy logic worked but backend is down, 422 = Backend rejected. All mean proxy matched.
                print(f"[PASS] Proxy routed to TTS. Status: {resp.status_code}")
            else:
                print(f"[FAIL] Expected 200/502/422 but got {resp.status_code} - {resp.text}")
                
            # 3. Test Vision Routing
            print("Testing Vision proxy route...")
            vision_payload = {"prompt": "A magical orchestrator", "cfg_normalization": True}
            resp = await client.post(f"{orchestrator_url}/v1/images/async_generate", json=vision_payload)
            if resp.status_code in [200, 502, 422]:
                print(f"[PASS] Proxy routed to Vision. Status: {resp.status_code}")
            else:
                print(f"[FAIL] Expected 200/502/422 but got {resp.status_code} - {resp.text}")

            # 4. Test Director Workflow
            print("Testing Director workflow...")
            worker_payload = {
                "image_prompt": "A cyberpunk city at night with neon lights and flying cars.",
                "voice": "Vivian",
                "voiceover_text": "Welcome to Neo-Tokyo. The air here tastes like battery acid and broken dreams.",
            }
            # This endpoint is extremely heavy (takes minutes), so we'll just check if it returns a 200 after processing.
            resp = await client.post(f"{orchestrator_url}/v1/workflows/director", json=worker_payload, timeout=600.0)
            if resp.status_code == 200:
                print(f"[PASS] Director Workflow succeeded! Final output created at: {resp.json().get('output_url')}")
            else:
                print(f"[FAIL] Director Workflow failed: {resp.status_code} - {resp.text}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ERROR] Connection failed context: {repr(e)}")

if __name__ == "__main__":
    asyncio.run(test_orchestrator())
