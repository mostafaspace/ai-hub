import asyncio
import websockets
import json

async def test_stream():
    uri = "ws://127.0.0.1:8000/v1/audio/stream"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket.")
            
            payload = {
                "input": "This is a streaming test to ensure no gibberish.",
                "voice": "Vivian",
                "language": "English"
            }
            await websocket.send(json.dumps(payload))
            print("Sent payload, waiting for audio chunks...")
            
            with open("streamed_output.pcm", "wb") as f:
                while True:
                    try:
                        chunk = await websocket.recv()
                        if isinstance(chunk, bytes):
                            f.write(chunk)
                            print(f"Received chunk of {len(chunk)} bytes")
                        else:
                            print(f"Received text message: {chunk}")
                            if "error" in chunk.lower():
                                break
                    except websockets.exceptions.ConnectionClosed:
                        print("Connection closed cleanly.")
                        break
            print("Finished writing streamed_output.pcm")
            
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    asyncio.run(test_stream())
