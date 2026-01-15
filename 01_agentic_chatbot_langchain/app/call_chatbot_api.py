
'''
#######################################################
import requests

# ✅ Phải dùng HTTPS

HOST = "https://agentic-chatbot-api--ts1krju.mangosea-bfac89e0.southeastasia.azurecontainerapps.io/" #"http://localhost:8001/" #"http://127.0.0.1:8001/" # "https://agentic-chatbot-api.mangosea-bfac89e0.southeastasia.azurecontainerapps.io:8000/"
url = HOST + "chat"

# ✅ Payload đúng format
payload = {
    "userid": "nhan.ngothanh12",
    "question": "cho tôi thông tin về vay mua nhà ở vib"
}

try:
    # ✅ Có timeout để tránh treo
    response = requests.post(url, json=payload, timeout=60)
    print(f"User Question: {payload.get('question')}")
    print(f"Response: {response}")

    if response.status_code == 200:
        data = response.json()
        print("✅ Kết quả trả về:")
        print(f"User Question: {payload.get('question')}")
        print(f"UserID : {data.get('userid')}")
        print(f"Question: {data.get('question')}")
        print(f"Answer  : {data.get('answer')}")
    else:
        print(f"❌ Lỗi {response.status_code}: {response.text}")

except requests.exceptions.RequestException as e:
    print("🚫 Lỗi kết nối:", e)
'''

#######################################################

import requests

HOST = "https://agentic-chatbot-api--ts1krju.mangosea-bfac89e0.southeastasia.azurecontainerapps.io/"
URL = HOST + "chat"

USER_ID = "nhan.ngothanh12"

print("🤖 Chat client started")
print("👉 Nhập câu hỏi (nhấn 'q' để thoát)\n")

while True:
    question = input("🧑 Bạn: ").strip()

    if question.lower() == "q":
        print("👋 Thoát chương trình.")
        break

    if not question:
        print("⚠️ Câu hỏi trống, vui lòng nhập lại.")
        continue

    payload = {
        "userid": USER_ID,
        "question": question
    }

    try:
        response = requests.post(
            URL,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            print("🤖 Bot:")
            print(data.get("answer", "⚠️ Không có câu trả lời"))
            print("-" * 60)
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")

    except requests.exceptions.RequestException as e:
        print("🚫 Lỗi kết nối:", e)
