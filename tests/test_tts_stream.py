import asyncio
import websockets
import json
import time
import wave
import sys
import os

# Qwen3-TTS default WebSocket streaming endpoint wrapper 
URI = "ws://127.0.0.1:8000/v1/audio/stream"
OUTPUT_FILE = "stream_output.wav"
SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2 # 16-bit PCM (2 bytes)

async def test_streaming_tts(text_prompt: str, voice: str = "Vivian"):
    print(f"Connecting to {URI}...")
    
    try:
        async with websockets.connect(URI) as websocket:
            print("WebSocket connected!")
            
            # 1. Send Configuration Payload
            payload = {
                "input": text_prompt,
                "voice": voice,
                "language": "Auto"
            }
            
            start_time = time.time()
            ttfb_time = None
            
            print(f"Sending payload: {json.dumps(payload)}")
            await websocket.send(json.dumps(payload))
            
            # 2. Setup WAV file to aggressively append to
            print(f"Opening {OUTPUT_FILE} for writing...")
            chunks_received = 0
            total_bytes = 0
            
            with wave.open(OUTPUT_FILE, 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(SAMPLE_WIDTH)
                wav_file.setframerate(SAMPLE_RATE)
                
                # 3. Listen for Binary PCM Audio Chunks
                try:
                    while True:
                        # Receive binary data stream
                        message = await websocket.recv()
                        
                        if isinstance(message, bytes):
                            # Record Time-To-First-Byte on the very first chunk
                            if chunks_received == 0:
                                ttfb_time = time.time()
                                print(f"[SUCCESS] TTFB (Time-To-First-Byte): {ttfb_time - start_time:.2f} seconds")
                                print("Streaming audio chunks to disk...")
                            
                            # Append chunk directly to WAV file
                            wav_file.writeframes(message)
                            
                            chunks_received += 1
                            total_bytes += len(message)
                            sys.stdout.write(f"\rRecieved chunk {chunks_received} ({total_bytes / 1024:.1f} KB)")
                            sys.stdout.flush()
                        else:
                            print(f"\nReceived text message (Error?): {message}")
                            
                except websockets.exceptions.ConnectionClosedOK:
                    print("\n\n[OK] Stream completed normally by server.")
                except websockets.exceptions.ConnectionClosedError as e:
                    print(f"\n\n[ERROR] Stream closed abruptly: {e}")
            
            end_time = time.time()
            print(f"\nSummary:")
            print(f"- Total Time      : {end_time - start_time:.2f}s")
            print(f"- Total Audio     : {total_bytes / 1024:.1f} KB")
            print(f"- Chunks Recv.    : {chunks_received}")
            print(f"- File Saved to   : {os.path.abspath(OUTPUT_FILE)}")
            
    except ConnectionRefusedError:
         print(f"[ERROR] Connection Refused. Is the Qwen3-TTS server running on port 8000?")
         
if __name__ == "__main__":
    prompt = "Welcome to the real-time WebSocket streaming audio test! This is a long sentence designed to prove that the chunks are generated sequentially while the PyTorch inference is still running in the background. If this works, the latency will be incredibly low!"
    asyncio.run(test_streaming_tts(prompt))
