##############################################
# uvicorn fastapi_prefect:app --reload
# 
# Description:
#     Fastapi using prefect, importing prefect_predict.py
# 

from fastapi import FastAPI
from pydantic import BaseModel
from prefect import flow
from prefect.deployments import run_deployment
from prefect_predict import scoring_flow

app = FastAPI()


class Customer(BaseModel):
    customer_id: int
    recency_days: float
    frequency: float
    monetary: float
    avg_amount: float
    qty_mean: float
    channels: float
    tenure_days: float

@app.post("/score")
async def score_customer(payload: Customer):

    # Gọi Prefect flow để thực thi
    score_value = scoring_flow(payload.dict())

    return {
        "customer_id": payload.customer_id,
        "score": score_value,
        "status": "success"
    }
