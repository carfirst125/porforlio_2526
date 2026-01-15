from prefect import task, flow
import requests
from requests.exceptions import RequestException


# ==============================
# TASK CALL API WITH RETRY
# ==============================
@task(name="Task - call scoring api",
    retries=3,                    # thử lại tối đa 3 lần
    retry_delay_seconds=2,        # chờ 2 giây giữa mỗi lần retry
)
def call_scoring_api(payload: dict):

    url = "http://localhost:8000/score"   # FastAPI scoring endpoint

    try:
        response = requests.post(url, json=payload, timeout=15)

        if response.status_code != 200:
            raise Exception(f"API returned {response.status_code}: {response.text}")

        return response.json()

    except RequestException as e:
        # Prefect sẽ tự retry task này
        raise Exception(f"Request failed: {str(e)}")


# ==============================
# MAIN FLOW
# ==============================
@flow(name="Call Customer Demand forecast API")
def scoring_flow():

    sample_payload = {
        "customer_id": 999999,
        "recency_days": 120,
        "frequency": 4,
        "monetary": 250.5,
        "avg_amount": 62.6,
        "qty_mean": 1.3,
        "channels": 2,
        "tenure_days": 320
    }

    result = call_scoring_api(sample_payload)
    print("Final scoring result:", result)


if __name__ == "__main__":
    scoring_flow()
