import requests
import time

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
            print(f"=> TEST FAILED (Expected {expected_status})\n")
    except Exception as e:
        print(f"=> REQUEST FAILED: {e}\n")

if __name__ == "__main__":
    print("Waiting 2 secs for server to start...")
    time.sleep(2)
    print("--- Test 1: Invalid Token ---")
    test_auth(INVALID_TOKEN, 403)
    
    print("--- Test 2: Valid Token ---")
    test_auth(VALID_TOKEN, 200)
