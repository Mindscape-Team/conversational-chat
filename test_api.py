import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://127.0.0.1:8000"

def test_health():
    
    response = requests.get(f"{BASE_URL}/health")
    print("\n=== Health Check ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_root():
    
    response = requests.get(BASE_URL)
    print("\n=== Root Endpoint ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_chat_session():
    
    # Generate a unique user ID
    user_id = f"test_user_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    print(f"\n=== Starting Chat Session for {user_id} ===")
    
    # 1. Start Session
    print("\n1. Starting Session...")
    response = requests.post(f"{BASE_URL}/start_session", params={"user_id": user_id})
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code != 200:
        print("Failed to start session")
        return False
    
    session_id = response.json()["session_id"]
    
    # 2. Send Messages
    test_messages = [
        "Hi, I'm feeling a bit anxious today",
        "I've been having trouble sleeping",
        "What can I do to feel better?",
        "Thank you for your help"
    ]
    
    print("\n2. Sending Messages...")
    for message in test_messages:
        print(f"\nSending message: {message}")
        response = requests.post(
            f"{BASE_URL}/send_message",
            json={"user_id": user_id, "message": message}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Chatbot: {response.json()['response']}")
        
        if response.status_code != 200:
            print("Failed to send message")
            return False
    
    # 3. End Session
    print("\n3. Ending Session...")
    response = requests.post(f"{BASE_URL}/end_session", params={"user_id": user_id})
    print(f"Status Code: {response.status_code}")
    print(f"Session Summary: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def run_all_tests():
    
    print("=== Starting API Tests ===")
    
    tests = {
        "Health Check": test_health,
        "Root Endpoint": test_root,
        "Chat Session": test_chat_session
    }
    
    results = {}
    for test_name, test_func in tests.items():
        print(f"\nRunning {test_name}...")
        try:
            success = test_func()
            results[test_name] = "✅ Passed" if success else "❌ Failed"
        except Exception as e:
            print(f"Error: {str(e)}")
            results[test_name] = "❌ Error"
    
    print("\n=== Test Results ===")
    for test_name, result in results.items():
        print(f"{test_name}: {result}")

if __name__ == "__main__":
    run_all_tests() 