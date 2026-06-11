
import os
import json
import xgboost as xgb
import pandas as pd


MODEL_FILE = "xgb_model.json"


def model_fn(model_dir):
    """
    Load XGBoost model from model.tar.gz
    """
    model_path = os.path.join(model_dir, MODEL_FILE)

    booster = xgb.Booster()
    booster.load_model(model_path)

    return booster


def input_fn(request_body, content_type):
    """
    Parse input JSON
    Expected format:
    {
        "area": 120,
        "bedrooms": 3,
        "bathrooms": 2
    }
    """
    if content_type == "application/json":
        data = json.loads(request_body)

        # single record → DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)

        return df

    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """
    Run prediction using XGBoost Booster
    """
    dmatrix = xgb.DMatrix(input_data)
    predictions = model.predict(dmatrix)
    return predictions


def output_fn(prediction, accept):
    """
    Format output
    """
    if accept == "application/json":
        return json.dumps({
            "prediction": prediction.tolist()
        }), accept

    raise ValueError(f"Unsupported accept type: {accept}")


######################################################
'''

import os
import json
import joblib
import pandas as pd

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, content_type):
    if content_type == "application/json":
        data = json.loads(request_body)
        return pd.DataFrame([data])
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps({
            "prediction": prediction.tolist()
        }), accept
    raise ValueError(f"Unsupported accept: {accept}")
'''