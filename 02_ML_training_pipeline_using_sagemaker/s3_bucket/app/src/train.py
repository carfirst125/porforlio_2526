# src/train.py
import argparse
import os
import pandas as pd
import xgboost as xgb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")

    args = parser.parse_args()

    # Input file: /opt/ml/input/data/train/normalized_data.csv
    input_file = os.path.join(args.train, "normalized_data.csv")

    df = pd.read_csv(input_file)

    target = "MedHouseVal"
    X = df.drop(columns=[target])
    y = df[target]

    dtrain = xgb.DMatrix(X, label=y)

    params = {
        "max_depth": 6,
        "eta": 0.1,
        "objective": "reg:squarederror",
        "eval_metric": "rmse"
    }

    bst = xgb.train(params, dtrain, num_boost_round=100)

    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "xgb_model.json")

    bst.save_model(model_path)
    print(f"Model saved to {model_path}")
