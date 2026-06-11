"""
End-to-end sample:
 - Feature engineering + labeling
 - PCA + preprocessing
 - Train LogisticRegression & XGBoost
 - Persist artifacts
 - Generate FastAPI scoring app
 - SAVE DATA & MODEL INTO ./data và ./model
 - Evaluate model + confusion matrix
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 0) CHECK & CREATE FOLDERS
# =========================================================
os.makedirs("./data", exist_ok=True)
os.makedirs("./model", exist_ok=True)
os.makedirs("./docker", exist_ok=True)
os.makedirs("./report", exist_ok=True)


# =========================================================
# 1) Dummy transaction generator
# =========================================================
def create_dummy_transactions(n_customers=1000, n_tx=8000, seed=42):
    np.random.seed(seed)
    cust_ids = np.arange(1, n_customers + 1)
    start = datetime(2024, 1, 1)
    txs = []
    for i in range(n_tx):
        cid = np.random.choice(cust_ids)
        days = np.random.randint(0, 365)
        dt = start + timedelta(days=int(days), hours=np.random.randint(0, 24))
        qty = np.random.choice([1, 1, 1, 2, 3], p=[0.6, 0.1, 0.1, 0.1, 0.1])
        price = float(np.round(np.random.choice([10, 20, 30, 50, 100]) * (1 + np.random.rand() * 0.3), 2))
        product_id = np.random.randint(1, 200)
        channel = np.random.choice(['web', 'app', 'store'])
        promo = np.random.choice([0, 1], p=[0.85, 0.15])
        txs.append([i + 1, cid, dt, product_id, qty, price, channel, promo])

    tx_df = pd.DataFrame(
        txs,
        columns=[
            'transaction_id', 'customer_id', 'purchase_datetime', 'product_id',
            'quantity', 'price', 'channel', 'promo_flag'
        ]
    )
    tx_df['amount'] = tx_df['quantity'] * tx_df['price']
    return tx_df


tx_df = create_dummy_transactions()
tx_df.to_csv("./data/dummy_transactions.csv", index=False)
print("Saved ./data/dummy_transactions.csv")


# =========================================================
# 2) Label generation
# =========================================================
last_date = tx_df['purchase_datetime'].max()
cust_agg = tx_df.groupby('customer_id').agg(
    last_purchase=('purchase_datetime', 'max'),
    total_amount=('amount', 'sum'),
    tx_count=('transaction_id', 'count')
).reset_index()

cust_agg['days_since_last'] = (last_date - cust_agg['last_purchase']).dt.days

median_tx = cust_agg['tx_count'].median()
cust_agg['label'] = ((cust_agg['days_since_last'] < 60) &
                     (cust_agg['tx_count'] > median_tx)).astype(int)

cust_agg.to_csv("./data/dummy_transactions_label.csv", index=False)
print("Saved ./data/dummy_transactions_label.csv")


# =========================================================
# 3) Feature engineering
# =========================================================
def build_customer_features(tx_df, reference_date=None):
    if reference_date is None:
        reference_date = tx_df['purchase_datetime'].max() + timedelta(days=1)

    agg = tx_df.groupby('customer_id').agg(
        recency_days=('purchase_datetime', lambda x: (reference_date - x.max()).days),
        frequency=('transaction_id', 'count'),
        monetary=('amount', 'sum'),
        avg_amount=('amount', 'mean'),
        qty_mean=('quantity', 'mean'),
        channels=('channel', lambda x: x.nunique())
    ).reset_index()

    first = tx_df.groupby('customer_id')['purchase_datetime'].min().reset_index()
    first = first.rename(columns={'purchase_datetime': 'first_purchase'})
    agg = agg.merge(first, on='customer_id', how='left')
    agg['tenure_days'] = (reference_date - agg['first_purchase']).dt.days
    return agg


cust_features = build_customer_features(tx_df)

df = cust_features.merge(
    cust_agg[['customer_id', 'label']],
    on='customer_id',
    how='left'
).fillna({'label': 0})

df.to_csv("./data/features.csv", index=False)
print("Saved ./data/features.csv")


# =========================================================
# 4) Numeric + categorical features
# =========================================================
numeric_features = [
    'recency_days', 'frequency', 'monetary',
    'avg_amount', 'qty_mean', 'channels', 'tenure_days'
]
categorical_features = []


# =========================================================
# 5) Correlation filter
# =========================================================
corr = df[numeric_features].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
numeric_features = [f for f in numeric_features if f not in to_drop]


# =========================================================
# 6) Information Value
# =========================================================
def calc_iv(df, feature, target, bins=10):
    ser = df[[feature, target]].copy()
    if ser[feature].dtype == 'O':
        grouped = ser.groupby(feature)[target].agg(['count', 'sum'])
    else:
        ser['bucket'] = pd.qcut(ser[feature].rank(method='first'), q=bins, duplicates='drop')
        grouped = ser.groupby('bucket')[target].agg(['count', 'sum'])

    grouped = grouped.rename(columns={'sum': 'event'})
    grouped['non_event'] = grouped['count'] - grouped['event']

    grouped['event_rate'] = (grouped['event'] + 0.5) / (grouped['event'].sum() + 0.5 * len(grouped))
    grouped['non_event_rate'] = (grouped['non_event'] + 0.5) / (grouped['non_event'].sum() + 0.5 * len(grouped))

    grouped['woe'] = np.log(grouped['event_rate'] / grouped['non_event_rate'])
    grouped['iv'] = (grouped['event_rate'] - grouped['non_event_rate']) * grouped['woe']
    return grouped['iv'].sum()


iv_values = {f: calc_iv(df, f, 'label', bins=6) for f in numeric_features}
numeric_features = [f for f, v in iv_values.items() if v > 0.01]


# =========================================================
# 7) Preprocessing pipeline
# =========================================================
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

transformers = [('num', num_pipe, numeric_features)]
preprocessor = ColumnTransformer(transformers)


# =========================================================
# 8) Train-test split
# =========================================================
X = df[numeric_features]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

# Save training dataset
train_json = pd.DataFrame(X_train)
train_json['label'] = y_train.values
train_json.to_json("./data/train_dataset.json", orient="records", indent=2)
print("Saved ./data/train_dataset.json")


# Fit preprocess
preprocessor.fit(X_train)
X_train_t = preprocessor.transform(X_train)
X_test_t = preprocessor.transform(X_test)

# PCA
n_num = len(numeric_features)
max_components = 3
pca_temp = PCA(n_components=min(max_components, n_num))
pca_temp.fit(X_train_t[:, :n_num])
cumvar = np.cumsum(pca_temp.explained_variance_ratio_)
n_comp = int(np.searchsorted(cumvar, 0.95) + 1)
n_comp = max(1, min(n_comp, n_num))

pca = PCA(n_components=n_comp)
X_train_final = pca.fit_transform(X_train_t[:, :n_num])
X_test_final = pca.transform(X_test_t[:, :n_num])


# =========================================================
# 9) Train models
# =========================================================
log_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
log_clf.fit(X_train_final, y_train)

xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9
)
xgb.fit(X_train_final, y_train)

print("Training finished.")


# =========================================================
# 10) Evaluate model + Confusion Matrix
# =========================================================
def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Logistic
log_pred = log_clf.predict(X_test_final)
log_auc = roc_auc_score(y_test, log_clf.predict_proba(X_test_final)[:, 1])
print("\n=== Logistic Regression ===")
print(classification_report(y_test, log_pred))
print("ROC AUC:", log_auc)

save_confusion_matrix(
    y_test, log_pred,
    "Confusion Matrix - Logistic Regression",
    "./report/confusion_matrix_logistic.png"
)
print("Saved ./report/confusion_matrix_logistic.png")


# XGBoost
xgb_pred = xgb.predict(X_test_final)
xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test_final)[:, 1])
print("\n=== XGBoost ===")
print(classification_report(y_test, xgb_pred))
print("ROC AUC:", xgb_auc)

save_confusion_matrix(
    y_test, xgb_pred,
    "Confusion Matrix - XGBoost",
    "./report/confusion_matrix_xgb.png"
)
print("Saved ./report/confusion_matrix_xgb.png")


# =========================================================
# 11) Save model artifacts
# =========================================================
artifacts = {
    'preprocessor': preprocessor,
    'pca': pca,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'logistic': log_clf,
    'xgb': xgb
}

joblib.dump(artifacts, "./model/scoring_artifacts.pkl")
print("Saved model → ./model/scoring_artifacts.pkl")


# =========================================================
# 12) Generate FastAPI scoring app
# =========================================================
fastapi_app_code = '''
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
'''

with open("fastapi_scoring_app.py", "w", encoding="utf-8") as f:
    f.write(fastapi_app_code)

print("Wrote fastapi_scoring_app.py")
print("Run API with: uvicorn fastapi_scoring_app:app --reload")
