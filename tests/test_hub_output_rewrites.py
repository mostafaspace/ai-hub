import asyncio
import os
import sys

import httpx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from orchestrator import server


class FakeHTTPClient:
    def __init__(self):
        self.calls = []

    async def get(self, url, headers=None):
        self.calls.append(("get", url, headers))
        request = httpx.Request("GET", url, headers=headers)

        if url == "http://vision.local/v1/images/tasks/task-image":
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "data": [{"url": "http://vision.local/outputs/gen_image.png"}],
                },
                headers={"content-type": "application/json"},
                request=request,
            )

        if url == "http://video.local/v1/video/tasks/task-video":
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "url": "http://video.local/outputs/clip.mp4",
                },
                headers={"content-type": "application/json"},
                request=request,
            )

        if url == "http://tts.local/v1/models":
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": "qwen-tts-base", "object": "model"}]},
                headers={"content-type": "application/json"},
                request=request,
            )

        if url == "http://music.local/v1/models":
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": "ace-step-1.5", "object": "model"}]},
                headers={"content-type": "application/json"},
                request=request,
            )

        if url == "http://vision.local/v1/models":
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": "z-image", "object": "model"}]},
                headers={"content-type": "application/json"},
                request=request,
            )

        if url == "http://video.local/v1/models":
            return httpx.Response(404, json={"detail": "Not found"}, headers={"content-type": "application/json"}, request=request)

        raise AssertionError(f"Unexpected GET URL: {url}")

    def build_request(self, method, url, headers=None, content=None):
        self.calls.append(("build_request", method, url, headers))
        return httpx.Request(method, url, headers=headers, content=content)

    async def send(self, request, stream=False):
        self.calls.append(("send", request.method, str(request.url), stream))
        url = str(request.url)

        if url == "http://vision.local/outputs/gen_image.png":
            return httpx.Response(
                200,
                content=b"image-bytes",
                headers={"content-type": "image/png"},
                request=request,
            )

        if url == "http://video.local/outputs/clip.mp4":
            return httpx.Response(
                200,
                content=b"video-bytes",
                headers={"content-type": "video/mp4"},
                request=request,
            )

        if url == "http://tts.local/v1/audio/voices":
            return httpx.Response(
                200,
                json={"voices": [{"voice_id": "Vivian", "name": "Vivian"}]},
                headers={"content-type": "application/json"},
                request=request,
            )

        if url == "http://music.local/v1/stats":
            return httpx.Response(
                200,
                json={"queue_size": 1, "jobs": {"queued": 1}},
                headers={"content-type": "application/json"},
                request=request,
            )

        raise AssertionError(f"Unexpected proxy URL: {request.url}")


async def main():
    original_backends = dict(server.BACKENDS)
    original_client = server.http_client
    fake_client = FakeHTTPClient()

    server.BACKENDS = {
        "tts": "http://tts.local",
        "music": "http://music.local",
        "vision": "http://vision.local",
        "video": "http://video.local",
    }
    server.http_client = fake_client

    try:
        expected_base = server._orchestrator_base_url()
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get("/v1/models")
            assert resp.status_code == 200, resp.text
            model_rows = resp.json()["data"]
            assert {item["id"] for item in model_rows} == {"qwen-tts-base", "ace-step-1.5", "z-image"}
            assert {item["source_backend"] for item in model_rows} == {"tts", "music", "vision"}
            print("[PASS] Hub model listing aggregates backend model catalogs.")

            resp = await client.get("/v1/audio/voices")
            assert resp.status_code == 200, resp.text
            assert resp.json()["voices"][0]["voice_id"] == "Vivian"
            print("[PASS] Voice listing is available through the hub.")

            resp = await client.get("/v1/stats")
            assert resp.status_code == 200, resp.text
            assert resp.json()["queue_size"] == 1
            print("[PASS] Music stats are available through the hub.")

            resp = await client.get("/v1/images/tasks/task-image")
            assert resp.status_code == 200, resp.text
            assert resp.json()["data"][0]["url"] == f"{expected_base}/v1/images/outputs/gen_image.png"
            print("[PASS] Vision task URLs are rewritten to hub output paths.")

            resp = await client.get("/v1/video/tasks/task-video")
            assert resp.status_code == 200, resp.text
            assert resp.json()["url"] == f"{expected_base}/v1/video/outputs/clip.mp4"
            print("[PASS] Video task URLs are rewritten to hub output paths.")

            resp = await client.get("/v1/images/outputs/gen_image.png")
            assert resp.status_code == 200, resp.text
            assert resp.headers["content-type"].startswith("image/png")
            assert resp.content == b"image-bytes"
            print("[PASS] Vision output downloads are proxied through the hub.")

            resp = await client.get("/v1/video/outputs/clip.mp4")
            assert resp.status_code == 200, resp.text
            assert resp.headers["content-type"].startswith("video/mp4")
            assert resp.content == b"video-bytes"
            print("[PASS] Video output downloads are proxied through the hub.")
    finally:
        server.BACKENDS = original_backends
        server.http_client = original_client


if __name__ == "__main__":
    asyncio.run(main())
