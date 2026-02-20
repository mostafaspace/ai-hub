
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
import config


def kill_port_occupants():
    """Kill any existing processes occupying our configured ports."""
    ports = [config.TTS_PORT, config.MUSIC_PORT, config.ASR_PORT, config.VISION_PORT]
    pids_to_kill = set()

    for port in ports:
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                # Match lines like:  TCP  0.0.0.0:8000  0.0.0.0:0  LISTENING  12345
                parts = line.split()
                if len(parts) >= 5 and f":{port}" in parts[1] and parts[3] == "LISTENING":
                    pid = int(parts[4])
                    if pid > 0:
                        pids_to_kill.add((port, pid))
        except Exception:
            pass

    if not pids_to_kill:
        return

    print(f"Found {len(pids_to_kill)} process(es) occupying server ports:")
    killed = set()
    for port, pid in pids_to_kill:
        if pid in killed:
            continue
        print(f"  Killing PID {pid} on port {port}...")
        try:
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True, timeout=5
            )
            killed.add(pid)
        except Exception as e:
            print(f"  Warning: Could not kill PID {pid}: {e}")

    # Brief pause for OS to release the sockets
    if killed:
        time.sleep(1)


SERVERS = [
    {
        "name": "TTS",
        "cwd": "Qwen3-TTS", 
        "cmd": ["python", "server.py"], # It will pick up config from within its own code now too
        "color": "\033[96m",  # Cyan
    },
    {
        "name": "MUSIC",
        "cwd": "ACE-Step-1.5",
        "cmd": ["uv", "run", "python", "-u", "-m", "acestep.api_server", "--host", config.HOST, "--port", str(config.MUSIC_PORT)],
        "color": "\033[95m",  # Magenta
    },
    {
        "name": "ASR",
        "cwd": "Qwen3-ASR",
        "cmd": ["python", "server.py"],
        "color": "\033[93m",  # Yellow
    },
    {
        "name": "VISION",
        "cwd": "Z-Image",
        "cmd": ["python", "vision_server.py"],
        "color": "\033[92m",  # Green
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
        print(f"DEBUG: {server_name} stream ended. returncode: {process.poll()}")
        log_queue.put(f"\033[91m[{server_name}] Process exited with code {process.poll()}\033[0m")

def start_servers():
    """Start all servers as subprocesses."""
    kill_port_occupants()
    for server_conf in SERVERS:
        print(f"Starting {server_conf['name']} server...")
        env = os.environ.copy()
        env["HF_HOME"] = config.HF_HOME # Ensure explicit HF_HOME
        env["PYTHONUNBUFFERED"] = "1"    # Force unbuffered output
        
        # Force single GPU (device 0) for Music server to prevent tensor device mismatch
        if server_conf["name"] == "MUSIC":
            env["CUDA_VISIBLE_DEVICES"] = "0"

        try:
            p = subprocess.Popen(
                server_conf["cmd"],
                cwd=server_conf["cwd"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Merge stderr into stdout
                env=env,
                bufsize=0, # Unbuffered (binary)
            )
            processes.append(p)
            
            # Start monitoring thread
            t = threading.Thread(
                target=stream_reader, 
                args=(p, server_conf["name"], server_conf["color"]),
                daemon=True
            )
            t.start()
            
        except Exception as e:
            print(f"Failed to start {server_conf['name']}: {e}")

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
