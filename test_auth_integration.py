import subprocess
import time
import os
import sys

print("🚀 Starting Tracker Server...", flush=True)
env = os.environ.copy()
env["PYTHONPATH"] = "."
server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "tracker.app:app", "--host", "0.0.0.0", "--port", "8000"],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

time.sleep(4)  # Wait for uvicorn to start

print("🤖 Starting Mock Worker (Force Fail Validation)...", flush=True)
worker = subprocess.Popen(
    [sys.executable, "mock_worker.py", "--force-fail-validation", "--steps", "5", "--local-epochs", "1", "--heartbeat-sec", "1.0"],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

try:
    stdout, stderr = worker.communicate(timeout=45)
    print("=== WORKER OUTPUT ===")
    print(stdout)
    print("=====================")
except subprocess.TimeoutExpired:
    print("Worker timed out. Killing...")
    worker.kill()
    out, _ = worker.communicate()
    print("=== WORKER OUTPUT (TIMED OUT) ===")
    print(out)
    print("=====================")

print("Stopping server...", flush=True)
server.kill()
server_out, _ = server.communicate()

print("=== SERVER METRICS & BAN LOGS ===")
# We want to see the ban and requests
lines = [line for line in server_out.splitlines() if "failed" in line.lower() or "ban" in line.lower() or "submit" in line.lower() or "register" in line.lower() or "signup" in line.lower()]
for line in lines[-25:]:
    print(line)
