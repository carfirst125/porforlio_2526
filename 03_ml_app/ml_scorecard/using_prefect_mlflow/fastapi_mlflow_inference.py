# fastapi_mlflow_inference.py
#
# uvicorn fastapi_mlflow_inference:app --reload --port 8000
#
#
import mlflow
import mlflow.pyfunc
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# =====================================================
# CONFIG
# =====================================================
MLFLOW_TRACKING_URI = "file:./mlruns"
MODEL_NAME = "customer_conversion_model"
MODEL_STAGE = "Production"
ARTIFACT_PATH = "./model/scoring_artifacts.pkl"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# =====================================================
# LOAD MODEL & ARTIFACTS (ON STARTUP)
# =====================================================
print("🔄 Loading MLflow Production model...")
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
)

print("🔄 Loading preprocessing artifacts...")
artifacts = joblib.load(ARTIFACT_PATH)

preprocessor = artifacts["preprocessor"]
pca = artifacts["pca"]
numeric_features = artifacts["numeric_features"]
categorical_features = artifacts["categorical_features"]

print("✅ Model & artifacts loaded successfully")

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(title="Customer Conversion Scoring API")


# =====================================================
# INPUT SCHEMA
# =====================================================
class CustomerFeatures(BaseModel):
    recency_days: float
    frequency: float
    monetary: float
    avg_amount: float
    qty_mean: float
    channels: float
    tenure_days: float


# =====================================================
# PREDICTION ENDPOINT
# =====================================================
@app.post("/score")
def predict(customer: CustomerFeatures):
    # Convert input to DataFrame
    df = pd.DataFrame([customer.dict()])

    # ---- Preprocess ----
    X_trans = preprocessor.transform(df)

    n_num = len(numeric_features)
    X_num = X_trans[:, :n_num]

    if pca is not None:
        X_num_pca = pca.transform(X_num)
    else:
        X_num_pca = X_num

    if len(categorical_features) > 0:
        X_cat = X_trans[:, n_num:]
        X_final = np.hstack([X_num_pca, X_cat])
    else:
        X_final = X_num_pca

    # ---- Predict ----
    proba = model.predict(X_final)[0]
    prediction = int(proba >= 0.5)

    return {
        "probability": float(proba),
        "prediction": prediction
    }


# =====================================================
# HEALTH CHECK
# =====================================================
@app.get("/health")
def health():
    return {"status": "ok"}
