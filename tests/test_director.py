import httpx
import asyncio
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

async def test_director():
    orchestrator_url = f"http://127.0.0.1:{config.ORCHESTRATOR_PORT}"
    
    # We use a massive timeout here because we are sequentially generating audio, then image, then video.
    # Video generation alone can take 1-3 minutes or more on some hardware.
    timeout = httpx.Timeout(900.0, connect=60.0)
    
    payload = {
        "image_prompt": "A peaceful futuristic city at sunset, 4k, masterpiece, highly detailed",
        "voiceover_text": "Welcome to the city of tomorrow, where dreams touch the sky.",
        "voice": "Vivian"
    }

    print(f"============================================================")
    print(f"  Testing The Content Director Macro-Skill")
    print(f"  URL: {orchestrator_url}/v1/workflows/director")
    print(f"  Payload: {payload}")
    print(f"  (This will take several minutes...)")
    print(f"============================================================")
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(f"{orchestrator_url}/v1/workflows/director", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                print(f"\n[PASS] Workflow completed successfully!")
                print(f"Status: {data.get('status')}")
                print(f"Task ID: {data.get('task_id')}")
                print(f"\n---> Final Video Saved to: {data.get('final_video_path')} <---")
            else:
                print(f"\n[FAIL] Workflow failed with status: {resp.status_code}")
                print(resp.text)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n[ERROR] Request failed: {repr(e)}")

if __name__ == "__main__":
    asyncio.run(test_director())
