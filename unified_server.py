
"""
Antigravity AI - Unified Server Launcher

Starts all AI servers (TTS, Music, ASR) in a single window as subprocesses.
Aggregates logs with prefixes and handles graceful shutdown.

Usage:
    python unified_server.py
"""

import subprocess
import sys
import threading
import queue
import time
import signal
import os
import ctypes

# Configuration
SERVERS = [
    {
        "name": "TTS",
        "cwd": "Qwen3-TTS",
        "cmd": ["python", "server.py"],
        "color": "\033[96m",  # Cyan
    },
    {
        "name": "MUSIC",
        "cwd": "ACE-Step-1.5",
        "cmd": ["uv", "run", "acestep-api", "--host", "0.0.0.0", "--port", "8001"],
        "color": "\033[95m",  # Magenta
    },
    {
        "name": "ASR",
        "cwd": "Qwen3-ASR",
        "cmd": ["python", "server.py"],
        "color": "\033[93m",  # Yellow
    },
]

# Global state
processes = []
stop_event = threading.Event()
log_queue = queue.Queue()

def enable_windows_ansi():
    """Enable ANSI escape sequences for Windows console."""
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

def stream_reader(process, server_name, color):
    """Reads stdout/stderr from a process and pushes to log queue."""
    for line in iter(process.stdout.readline, b''):
        if stop_event.is_set():
            break
        # Decode line
        try:
            line_str = line.decode('utf-8').rstrip()
        except UnicodeDecodeError:
            line_str = line.decode('cp1252', errors='replace').rstrip()
            
        if line_str:
            log_queue.put(f"{color}[{server_name}]\033[0m {line_str}")
            
    # Process ended
    if not stop_event.is_set():
        log_queue.put(f"\033[91m[{server_name}] Process exited with code {process.poll()}\033[0m")

def start_servers():
    """Start all servers as subprocesses."""
    for config in SERVERS:
        print(f"Starting {config['name']} server...")
        env = os.environ.copy()
        env["HF_HOME"] = r"D:\hf_models" # Ensure explicit HF_HOME
        env["PYTHONUNBUFFERED"] = "1"    # Force unbuffered output
        
        try:
            p = subprocess.Popen(
                config["cmd"],
                cwd=config["cwd"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Merge stderr into stdout
                env=env,
                bufsize=1, # Line buffered
            )
            processes.append(p)
            
            # Start monitoring thread
            t = threading.Thread(
                target=stream_reader, 
                args=(p, config["name"], config["color"]),
                daemon=True
            )
            t.start()
            
        except Exception as e:
            print(f"Failed to start {config['name']}: {e}")

def stop_servers(signum=None, frame=None):
    """Stop all servers gracefully."""
    if stop_event.is_set():
        return
        
    print("\nStopping all servers...")
    stop_event.set()
    
    for p in processes:
        if p.poll() is None: # If still running
            print(f"Terminating process {p.pid}...")
            p.terminate()
            
    # Give them a moment to shut down
    time.sleep(2)
    
    # Force kill if needed
    for p in processes:
        if p.poll() is None:
            print(f"Killing process {p.pid}...")
            p.kill()
            
    print("All servers stopped.")
    sys.exit(0)

def main():
    if sys.platform == "win32":
        enable_windows_ansi()
        # Handle Ctrl+C on Windows
        signal.signal(signal.SIGINT, stop_servers)
        signal.signal(signal.SIGTERM, stop_servers)
    
    print("=" * 60)
    print("      ANTIGRAVITY AI - Unified Server Launcher")
    print("      Press Ctrl+C to stop all servers")
    print("=" * 60)
    
    start_servers()
    
    # Main loop just prints logs
    try:
        while not stop_event.is_set():
            try:
                msg = log_queue.get(timeout=0.2)
                print(msg)
            except queue.Empty:
                # Check if all processes are dead
                if not processes or all(p.poll() is not None for p in processes):
                    print("All server processes have exited.")
                    stop_servers()
    except KeyboardInterrupt:
        stop_servers()

if __name__ == "__main__":
    main()
