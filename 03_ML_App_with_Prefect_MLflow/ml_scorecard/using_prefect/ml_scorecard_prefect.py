#####################################################################
# ml_scorecard_prefect.py
#
# Command:
# prefect deploy:
#    prefect deployment build ml_scorecard_prefect.py:train_flow -n train_daily
#    prefect deployment apply train_daily.yaml

from prefect import flow, task
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib
import category_encoders as ce
import json
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1) Task Tạo/Load dữ liệu
# =========================================================
@task(name="Create Dummy Transactions")
def create_dummy_transactions(n_customers=1000, n_tx=8000, seed=42):
    """Tạo dữ liệu giao dịch giả lập."""
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
    print(f"Created {len(tx_df)} dummy transactions.")
    tx_df.to_csv("./data/dummy_transactions.csv", index=False)
    return tx_df

# =========================================================
# 2) Task Tạo Label
# =========================================================
@task(name="Generate Customer Labels")
def generate_customer_labels(tx_df):
    """Tạo label giả lập (mua / không mua) cho từng khách hàng."""
    last_date = tx_df['purchase_datetime'].max()
    cust_agg = tx_df.groupby('customer_id').agg(
        last_purchase=('purchase_datetime', 'max'),
        total_amount=('amount', 'sum'),
        tx_count=('transaction_id', 'count')
    ).reset_index()

    cust_agg['days_since_last'] = (last_date - cust_agg['last_purchase']).dt.days

    # Giả lập label: recency < 60 và tx_count > median
    median_tx = cust_agg['tx_count'].median()
    cust_agg['label'] = ((cust_agg['days_since_last'] < 60) &
                         (cust_agg['tx_count'] > median_tx)).astype(int)
    
    print(f"Generated labels for {len(cust_agg)} customers. Label 1 count: {cust_agg['label'].sum()}")
    cust_agg.to_csv("./data/dummy_transactions_label.csv", index=False)

    return cust_agg[['customer_id', 'label']]

# =========================================================
# 3) Task Feature Engineering
# =========================================================
@task(name="Feature Engineering (RFM + Rolling)")
def build_customer_features(tx_df, cust_labels):
    """Tạo các feature từ transaction (RFM, tenure) và join với label."""
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
    
    # Join label
    df = agg.merge(
        cust_labels,
        on='customer_id',
        how='left'
    ).fillna({'label': 0})
    df.to_csv("./data/features.csv", index=False)
    
    print(f"Shape full dataset with features: {df.shape}")
    return df

# =========================================================
# 4) Task Feature Selection (Correlation + IV)
# =========================================================
def calc_iv(df, feature, target, bins=6):
    """Tính Information Value cho 1 feature."""
    # ... (giữ nguyên hàm calc_iv từ code gốc) ...
    ser = df[[feature, target]].copy()
    if ser[feature].dtype == 'O' or str(ser[feature].dtype).startswith('category'):
        grouped = ser.groupby(feature)[target].agg(['count', 'sum'])
    else:
        # Numeric -> qcut
        # Thêm kiểm tra số lượng giá trị duy nhất
        if ser[feature].nunique() < bins:
            q = ser[feature].nunique()
        else:
            q = bins

        if q <= 1: # Không đủ giá trị để bin
            return 0.0

        ser['bucket'] = pd.qcut(
            ser[feature].rank(method='first'),
            q=q,
            duplicates='drop'
        )
        grouped = ser.groupby('bucket')[target].agg(['count', 'sum'])

    grouped = grouped.rename(columns={'sum': 'event'})
    grouped['non_event'] = grouped['count'] - grouped['event']

    # Thêm constant để tránh chia 0
    grouped['event_rate'] = (grouped['event'] + 0.5) / (grouped['event'].sum() + 0.5 * len(grouped))
    grouped['non_event_rate'] = (grouped['non_event'] + 0.5) / (grouped['non_event'].sum() + 0.5 * len(grouped))

    # Xử lý log(0) bằng cách thêm 1e-6 vào mẫu nếu rate = 0 (dù đã có +0.5)
    grouped['woe'] = np.log(grouped['event_rate'] / grouped['non_event_rate'].replace(0, 1e-6))
    grouped['iv'] = (grouped['event_rate'] - grouped['non_event_rate']) * grouped['woe']

    return grouped['iv'].sum()

@task(name="Feature Selection (Corr + IV)")
def feature_selection(df, numeric_features, categorical_features, iv_threshold=0.01):
    """Lọc các feature theo Correlation và Information Value."""
    
    # 5) Correlation filter
    corr = df[numeric_features].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
    
    numeric_features = [f for f in numeric_features if f not in to_drop]
    print(f"Highly correlated features dropped: {to_drop}")
    
    # 6) IV filter
    iv_values = {}
    for f in numeric_features + categorical_features:
        iv_values[f] = calc_iv(df, f, 'label', bins=6)
    
    selected_by_iv = [f for f, v in iv_values.items() if v > iv_threshold]
    
    final_numeric = [f for f in numeric_features if f in selected_by_iv]
    final_categorical = [f for f in categorical_features if f in selected_by_iv]
    
    print(f"Features dropped by low IV (< {iv_threshold}): {[f for f,v in iv_values.items() if v <= iv_threshold]}")
    print(f"Final numeric features: {final_numeric}")
    print(f"Final categorical features: {final_categorical}")
    
    return final_numeric, final_categorical

# =========================================================
# 5) Task Preprocessing & Dimensionality Reduction
# =========================================================
@task(name="Preprocess, PCA, Split Data")
def preprocess_and_split(df, numeric_features, categorical_features, max_components=3):
    """Xây dựng preprocessor, PCA, và chia dữ liệu."""
    X = df[numeric_features + categorical_features]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.25,
        random_state=42
    )
    
    # ---- 7) Pipeline tiền xử lý ----
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = None
    if len(categorical_features) > 0:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('target_enc', ce.TargetEncoder(cols=categorical_features, smoothing=0.3))
        ])

    transformers = [('num', num_pipe, numeric_features)]
    if cat_pipe is not None:
        transformers.append(('cat', cat_pipe, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    # Fit preprocessor
    preprocessor.fit(X_train, y_train)
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    n_num = len(numeric_features)
    X_train_num = X_train_trans[:, :n_num]
    X_test_num = X_test_trans[:, :n_num]
    
    # ---- PCA cho numeric ----
    pca = None
    if n_num > 0:
        pca_temp = PCA(n_components=min(max_components, n_num))
        pca_temp.fit(X_train_num)

        cumvar = np.cumsum(pca_temp.explained_variance_ratio_)
        # chọn số component đủ cover 95% variance hoặc <= max_components
        if cumvar[-1] >= 0.95:
            n_comp = int(np.searchsorted(cumvar, 0.95) + 1)
        else:
            n_comp = min(max_components, n_num)

        n_comp = max(1, min(n_comp, n_num))
        
        # Fit lại PCA với số component cuối cùng
        pca = PCA(n_components=n_comp)
        X_train_num_pca = pca.fit_transform(X_train_num)
        X_test_num_pca = pca.transform(X_test_num)
        print(f"PCA reduced numeric features from {n_num} to {n_comp} components.")
    else:
        X_train_num_pca = np.empty((X_train.shape[0], 0))
        X_test_num_pca = np.empty((X_test.shape[0], 0))

    # ---- Ghép final features ----
    if len(categorical_features) > 0:
        X_train_cat = X_train_trans[:, n_num:]
        X_test_cat = X_test_trans[:, n_num:]
        X_train_final = np.hstack([X_train_num_pca, X_train_cat])
        X_test_final = np.hstack([X_test_num_pca, X_test_cat])
    else:
        X_train_final = X_train_num_pca
        X_test_final = X_test_num_pca
        
    print(f"Final feature shapes: Train {X_train_final.shape}, Test {X_test_final.shape}")

    # Trả về tất cả các artifacts cần thiết cho bước huấn luyện tiếp theo
    return {
        'X_train': X_train_final, 'X_test': X_test_final, 
        'y_train': y_train, 'y_test': y_test,
        'preprocessor': preprocessor, 'pca': pca
    }

# =========================================================
# 6) Task Huấn luyện & Đánh giá mô hình
# =========================================================
@task(name="Train and Evaluate Models")
def train_and_evaluate(processed_data, numeric_features, categorical_features):
    """Huấn luyện Logistic Regression và XGBoost, đánh giá."""
    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    
    models = {}
    metrics = {}
    
    # Logistic Regression
    log_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    log_clf.fit(X_train, y_train)
    y_pred_proba_log = log_clf.predict_proba(X_test)[:, 1]
    auc_log = roc_auc_score(y_test, y_pred_proba_log)
    print(f"Logistic AUC: {auc_log:.4f}")
    models['logistic'] = log_clf
    metrics['logistic_auc'] = auc_log
    
    # XGBoost
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    y_pred_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
    print(f"XGBoost AUC: {auc_xgb:.4f}")
    models['xgb'] = xgb
    metrics['xgb_auc'] = auc_xgb
    
    # Log classification report cho mô hình tốt hơn (ví dụ XGB)
    y_pred_xgb = (y_pred_proba_xgb >= 0.5).astype(int)
    print("\nXGB classification report:")
    print(classification_report(y_test, y_pred_xgb))

    # Lựa chọn mô hình có AUC tốt hơn
    best_model_name = 'xgb' if auc_xgb > auc_log else 'logistic'
    print(f"\nBest model selected: {best_model_name} with AUC {metrics[f'{best_model_name}_auc']:.4f}")
    print(f" numeric_features: {numeric_features}\n categorical_features: {categorical_features}")
    
    return {
        'models': models, 
        'metrics': metrics, 
        'best_model': best_model_name,
        'features': {'numeric': numeric_features, 'categorical': categorical_features}
    }

# =========================================================
# 7) Task Lưu Artifacts
# =========================================================
@task(name="Save Scoring Artifacts")
def save_artifacts(processed_data, model_results, filename="./model/scoring_artifacts.pkl"):
    """Lưu preprocessor, PCA, models và metadata vào một file."""
    artifacts = {
        'preprocessor': processed_data['preprocessor'],
        'pca': processed_data['pca'],
        'numeric_features': model_results['features']['numeric'],
        'categorical_features': model_results['features']['categorical'],
        'logistic': model_results['models']['logistic'],
        'xgb': model_results['models']['xgb'],
        'best_model': model_results['best_model']
    }

    joblib.dump(artifacts, filename)
    print(f"Saved artifacts to {filename}")
    return filename

# =========================================================
# 8) Flow chính
# =========================================================
@flow(name="Customer Demand Forecast Model Training")
def customer_ml_flow(n_customers=1000, n_tx=8000, iv_threshold=0.01):
    
    # Khai báo feature ban đầu (thêm các feature categorical vào đây nếu có)
    initial_numeric_features = [
        'recency_days', 'frequency', 'monetary',
        'avg_amount', 'qty_mean', 'channels', 'tenure_days'
    ]
    initial_categorical_features = [] # Ví dụ: ['region', 'gender']

    # 1-2. Load dữ liệu & Tạo label
    tx_df = create_dummy_transactions(n_customers=n_customers, n_tx=n_tx)
    cust_labels = generate_customer_labels(tx_df)

    # 3. Feature Engineering
    df_full = build_customer_features(tx_df, cust_labels)

    # 4. Feature Selection
    numeric_feats, categorical_feats = feature_selection(
        df_full, 
        initial_numeric_features, 
        initial_categorical_features, 
        iv_threshold=iv_threshold
    )    
    print(f"numeric_feats: {numeric_feats}\ncategorical_feats: {categorical_feats}")
    
    # 5. Preprocessing & PCA
    processed_data = preprocess_and_split(
        df_full, 
        numeric_feats, 
        categorical_feats
    )

    # 6. Train và Evaluate
    model_results = train_and_evaluate(
        processed_data, 
        numeric_feats, 
        categorical_feats
    )

    # 7. Lưu Artifacts
    artifact_path = save_artifacts(processed_data, model_results)
    
    print(f"\n--- Flow finished successfully ---")
    print(f"Artifacts saved at: {artifact_path}")
    print(f"Best Model (for API): {model_results['best_model']}")


if __name__ == "__main__":
    # Chạy flow
    customer_ml_flow(n_customers=1000, n_tx=8000, iv_threshold=0.01)
    
#########################################################
#
# Guide: python ml_scorecard_prefect.py
#