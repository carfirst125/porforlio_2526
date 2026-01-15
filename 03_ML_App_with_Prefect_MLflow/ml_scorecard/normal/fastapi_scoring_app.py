###############################################3
# uvicorn fastapi_scoring_app:app --reload
# 

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

class CustomerFeatures(BaseModel):
    customer_id: int
    recency_days: float
    frequency: float
    monetary: float
    avg_amount: float
    qty_mean: float
    channels: float
    tenure_days: float

app = FastAPI()
art = joblib.load("./model/scoring_artifacts.pkl")
preprocessor = art["preprocessor"]
pca = art["pca"]
numeric_feats = art["numeric_features"]
categorical_feats = art["categorical_features"]
model = art["xgb"]

@app.post("/score")
def score(payload: CustomerFeatures):
    df = pd.DataFrame([payload.dict()])
    X = df[numeric_feats]
    X_trans = preprocessor.transform(X)
    X_num = X_trans[:, :len(numeric_feats)]
    X_final = pca.transform(X_num)
    proba = model.predict_proba(X_final)[:, 1][0]
    return {"customer_id": payload.customer_id, "score": float(proba)}
