import subprocess
import time
import requests
import os
import sys

print("Starting tracking server for authorized tests...")
env = os.environ.copy()
env["PYTHONPATH"] = "."
server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "tracker.app:app", "--host", "0.0.0.0", "--port", "8000"],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

time.sleep(4)  # Wait for uvicorn to start fully

URL = "http://localhost:8000/register"
VALID_TOKEN = "dc8e790d-e6fc-4b1b-8b0a-714d8ef1b9e1"
INVALID_TOKEN = "fake-token"

def test_auth(token, expected_status):
    print(f"Testing with token: {token}")
    try:
        response = requests.post(
            URL, 
            json={"gpu_vram_mb": 8000, "cpu_count": 4, "host_label": "test-node", "token": token},
            timeout=5
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        if response.status_code == expected_status:
            print("=> TEST PASSED\n")
        else:
            print(f"=> TEST FAILED (Expected {expected_status}, got {response.status_code})\n")
    except Exception as e:
        print(f"=> REQUEST FAILED: {e}\n")

try:
    print("--- Test 1: Invalid Token ---")
    test_auth(INVALID_TOKEN, 403)
    
    print("--- Test 2: Valid Token ---")
    test_auth(VALID_TOKEN, 200)

finally:
    print("Tests finished. Stopping server...")
    server.kill()
    server.wait()
    print("Server stopped.")
