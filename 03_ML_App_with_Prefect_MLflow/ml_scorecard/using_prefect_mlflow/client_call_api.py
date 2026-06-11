# client_call_api.py
# 
#

import requests

# URL của API FastAPI
API_URL = "http://127.0.0.1:8000/score"

# Payload mẫu — KHÔNG có customer_id
payload = {
    "customer_id": 9999999, 
    "recency_days": 28,
    "frequency": 11,
    "monetary": 433.5,
    "avg_amount": 39.6,
    "qty_mean": 1.0,
    "channels": 3,
    "tenure_days": 320
}

try:
    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        print("=== Scoring Result ===")
        print(f"Result: {result}")
    else:
        print("❌ Error:", response.status_code)
        print("Message:", response.text)

except requests.exceptions.RequestException as e:
    print("❌ Request failed:", e)
