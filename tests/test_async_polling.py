import threading
import requests
import time
import json

BASE_URL = 'http://127.0.0.1:8003'

def trigger_generation():
    print(f'[Thread 1] Starting long generation request...')
    payload = {
        'prompt': 'A turtle wearing a top hat on a rocket, highly detailed',
        'n': 1,
        'size': '512x512',
        'response_format': 'url',
        'num_inference_steps': 30,
        'guidance_scale': 4.0
    }
    try:
        start_time = time.time()
        response = requests.post(f'{BASE_URL}/v1/images/generations', json=payload, timeout=600)
        dur = time.time() - start_time
        print(f'[Thread 1] Generation finished in {dur:.2f}s! Status: {response.status_code}')
        if response.status_code == 200:
            print(f'[Thread 1] URL: {response.json()["data"][0]["url"]}')
    except Exception as e:
        print(f'[Thread 1] Generation Error: {e}')

def poll_status():
    print(f'[Thread 2] Waiting 5 seconds before polling...')
    time.sleep(5)
    
    for i in range(5):
        try:
            print(f'[Thread 2] Polling /v1/internal/status (Attempt {i+1})...')
            start = time.time()
            res = requests.get(f'{BASE_URL}/v1/internal/status', timeout=5)
            dur = time.time() - start
            print(f'[Thread 2] Status Response ({dur*1000:.1f}ms): {res.json()}')
            
            # Additional test: check the invalid POST endpoint handler
            print(f'[Thread 2] Polling /v1/images/generations with GET...')
            res_bad = requests.get(f'{BASE_URL}/v1/images/generations', timeout=5)
            print(f'[Thread 2] Bad GET Response: {res_bad.json()}')
            
        except Exception as e:
            print(f'[Thread 2] Polling Error: {e}')
        
        time.sleep(15) # Poll every 15s

t1 = threading.Thread(target=trigger_generation)
t2 = threading.Thread(target=poll_status)

print("Starting multithreaded API test...")
t1.start()
t2.start()

t1.join()
t2.join()
print("Test Complete.")
