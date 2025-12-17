import requests

# URL của API FastAPI bạn đang chạy
API_URL = "http://127.0.0.1:8000/score"

# Payload mẫu – thay bằng dữ liệu thật của khách hàng
payload = {
    "customer_id": 999999,
    "recency_days": 12,
    "frequency": 20,
    "monetary": 250.5,
    "avg_amount": 62.6,
    "qty_mean": 1.3,
    "channels": 2,
    "tenure_days": 320
}

# Gửi POST request
response = requests.post(API_URL, json=payload)

# In kết quả
if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
