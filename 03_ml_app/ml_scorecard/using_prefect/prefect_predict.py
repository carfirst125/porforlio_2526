################################################
# prefect_predict.py
#    imported and function call by fastapi_prefect
#

from prefect import task, flow
import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict


ARTIFACT_PATH = "./model/scoring_artifacts.pkl"


# ============================
# TASK LOAD ARTIFACTS (cache)
# ============================
@task(persist_result=True)
def load_artifacts():

    if not os.path.exists(ARTIFACT_PATH):
        raise FileNotFoundError(f"Artifact not found: {ARTIFACT_PATH}")

    art = joblib.load(ARTIFACT_PATH)
    return {
        "preprocessor": art["preprocessor"],
        "pca": art["pca"],
        "numeric_feats": art["numeric_features"],
        "categorical_feats": art["categorical_features"],
        "model": art[art.get("best_model", "xgb")],
        "model_name": art.get("best_model", "xgb")
    }


# ============================
# TASK PREPROCESS (RETRY)
# ============================
@task(retries=3, retry_delay_seconds=2)
def preprocess_task(payload: Dict, artifacts: Dict):

    df = pd.DataFrame([payload])
    numeric_feats = artifacts["numeric_feats"]
    categorical_feats = artifacts["categorical_feats"]

    X = df[numeric_feats + categorical_feats]

    X_trans = artifacts["preprocessor"].transform(X)
    return X_trans


# ============================
# TASK PCA (RETRY)
# ============================
@task(retries=3, retry_delay_seconds=2)
def pca_task(X_trans, artifacts: Dict):

    n_num = len(artifacts["numeric_feats"])
    X_num = X_trans[:, :n_num]

    pca = artifacts["pca"]
    if pca and n_num > 0:
        X_num_pca = pca.transform(X_num)
    else:
        X_num_pca = X_num

    if len(artifacts["categorical_feats"]) > 0:
        X_cat = X_trans[:, n_num:]
        return np.hstack([X_num_pca, X_cat])

    return X_num_pca


# ============================
# TASK MODEL SCORING (RETRY)
# ============================
@task(retries=3, retry_delay_seconds=2)
def score_task(X_final, artifacts: Dict):
    model = artifacts["model"]
    proba = model.predict_proba(X_final)[:, 1][0]
    return float(proba)


# ============================
# MAIN FLOW
# ============================
@flow(name="Customer Demand Forecasting")
def scoring_flow(payload: dict):

    artifacts = load_artifacts()

    X_trans = preprocess_task(payload, artifacts)
    X_final = pca_task(X_trans, artifacts)
    score = score_task(X_final, artifacts)

    return {
        "customer_id": payload["customer_id"],
        "score": score,
        "model": artifacts["model_name"]
    }
