import subprocess
import sys
import os

print("Minimal Launcher Starting...")
orchestrator_path = os.path.join("orchestrator", "server.py")
print(f"Launching {orchestrator_path}...")

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

try:
    p = subprocess.Popen(
        [sys.executable, "-u", orchestrator_path],
        cwd=".",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    print(f"Process started with PID {p.pid}")
    
    # Read first 10 lines
    for i in range(20):
        line = p.stdout.readline()
        if not line:
            break
        print(f"[ORCH] {line.strip()}")
        
except Exception as e:
    print(f"Failed: {e}")

print("Minimal Launcher Done.")
