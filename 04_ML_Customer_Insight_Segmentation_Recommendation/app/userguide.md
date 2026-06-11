# User Guide - Customer Transaction App

## 0) Command Run (Quick Start)

Tu thu muc `05_customer_transaction`:

### 0.1 Cai dependencies

```powershell
pip install -r requirements.txt
```

### 0.2 Chay pipeline offline (build insights + artifacts + quality report)

```powershell
python -m app.jobs.refresh_insights
```

### 0.3 Chay API read-only

```powershell
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

### 0.4 Test API nhanh

```powershell
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/customers/17850
curl -X POST http://127.0.0.1:8000/v1/customers/insights `
  -H "Content-Type: application/json" `
  -d "{\"customer_ids\":[17850,13047,999999]}"
```

### 0.5 Chay Streamlit dashboard

```powershell
streamlit run app/streamlit/dashboard.py
```

---

## 1) Muc dich cua app

App nay cung cap quy trinh phan tich transaction theo kieu offline-first:

- Chay pipeline offline de lam sach data, tao feature, phan khuc khach hang, tinh insight, tao recommendation, va luu artifact.
- Chay FastAPI read-only de tra cuu insight theo `CustomerID` voi do tre thap.
- Viec cap nhat insight duoc thuc hien boi job dinh ky, khong nam trong API.
- Co he thong quality monitoring (test/measurement) sau moi lan refresh de theo doi drift va chat luong.

---

## 2) Cau truc thu muc `app/`

```text
app/
├─ __init__.py
├─ __main__.py
├─ config.py
├─ userguide.md
├─ api/
│  ├─ __init__.py
│  ├─ main.py
│  ├─ schemas.py
│  └─ serialize.py
├─ data/
│  ├─ __init__.py
│  └─ loader.py
├─ preprocessing/
│  ├─ __init__.py
│  ├─ profile.py
│  └─ cleaning.py
├─ features/
│  ├─ __init__.py
│  └─ customer_features.py
├─ modeling/
│  ├─ __init__.py
│  ├─ cluster_model.py
│  └─ recommender.py
├─ insights/
│  ├─ __init__.py
│  └─ extended_metrics.py
├─ storage/
│  ├─ __init__.py
│  ├─ artifact_store.py
│  └─ insight_store.py
├─ evaluation/
│  ├─ __init__.py
│  ├─ thresholds.py
│  ├─ baselines.py
│  ├─ checks.py
│  ├─ evaluator.py
│  └─ reporter.py
├─ pipeline/
│  ├─ __init__.py
│  └─ customer_insights_pipeline.py
├─ jobs/
│  ├─ __init__.py
│  └─ refresh_insights.py
├─ streamlit/
│  ├─ README.md
│  └─ dashboard.py
└─ artifacts/
   ├─ customer_insights.parquet
   ├─ clustering_bundle.joblib
   ├─ pipeline_meta.json
   └─ evaluation/
      ├─ evaluation_report.json
      └─ history/run_<UTC>.json
```

Luu y:
- `artifacts/` la output sau moi lan chay pipeline offline.
- API chi doc `customer_insights.parquet` de tra cuu nhanh.
- `artifacts/evaluation/` la output test/measurement sau moi lan refresh.

---

## 3) Danh sach file, class va chuc nang

### 3.1 File goc

- `app/__init__.py`
  - Khai bao version package.
- `app/__main__.py`
  - Huong dan nhanh cac entrypoint.
- `app/config.py`
  - **Class `Settings`**: quan ly tham so he thong bang `pydantic-settings`.
  - **Ham `get_settings()`**: tao doi tuong config.
  - Ho tro override qua env prefix `RETAIL_`.

### 3.2 API layer (`app/api/`)

- `app/api/main.py`
  - Tao FastAPI app voi lifespan load `InsightStore` vao RAM.
  - Endpoint:
    - `GET /health`
    - `GET /v1/customers/{customer_id}`
    - `POST /v1/customers/insights`
  - Khong train model trong API.
- `app/api/schemas.py`
  - Schema request/response cho health, single customer, batch customer.
- `app/api/serialize.py`
  - Convert pandas/numpy/datetime sang JSON-safe.

### 3.3 Data ingestion (`app/data/`)

- `app/data/loader.py`
  - Ham `load_transactions(settings)`:
    - Doc CSV transaction.
    - Kiem tra schema bat buoc.
    - Parse `InvoiceDate`.

### 3.4 Preprocessing (`app/preprocessing/`)

- `app/preprocessing/profile.py`
  - **Class `DatasetProfile`**:
    - Thong ke null ratio, duplicate ratio, cancellation ratio...
    - Tao profile de dieu chinh cleaning theo data thuc te.
  - Ham `column_skew()`: tinh do lech phan phoi.
- `app/preprocessing/cleaning.py`
  - **Class `CleaningReport`**: thong tin before/after moi buoc cleaning.
  - **Class `TransactionCleaner`**:
    - Loai dong khong co `CustomerID`, `InvoiceDate`, `Description`.
    - Loai `UnitPrice <= 0`.
    - Loai stock code bat thuong.
    - Danh dau cancellation (`_is_cancellation`).
    - Trim quantity cuc doan.

### 3.5 Feature engineering (`app/features/`)

- `app/features/customer_features.py`
  - **Class `FeatureBuildReport`**: mo ta output feature.
  - **Class `CustomerFeatureBuilder`**:
    - Tao RFM va bien hanh vi:
      - `RecencyDays`, `Frequency`, `Monetary`
      - `AvgOrderValue`, `OrdersPerMonth`, `UniqueProducts`
      - `CancelRate`, `IsUK`, `TenureDays`, `SpendTrendScore`

### 3.6 Modeling (`app/modeling/`)

- `app/modeling/cluster_model.py`
  - **Class `ClusteringResult`**: ket qua train (labels, model, scaler, pca, metrics...).
  - **Class `ClusteringTrainer`**:
    - Adaptive transform (`log1p`, tuy chon Yeo-Johnson).
    - StandardScaler.
    - PCA khi du dieu kien.
    - Chon `k` bang silhouette (neu du mau).
    - Train `KMeans` hoac `GaussianMixture` theo config.
- `app/modeling/recommender.py`
  - **Class `SegmentRecommender`**:
    - Tao top product theo segment.
    - Sinh danh sach SKU goi y cho tung khach hang.

### 3.7 Insight scoring (`app/insights/`)

- `app/insights/extended_metrics.py`
  - **Class `ExtendedInsightCalculator`**:
    - Them score marketing:
      - `churn_risk_score`
      - `clv_estimate`
      - `transactional_promoter_score` (proxy tu giao dich, khong phai survey NPS)
      - `R_score`, `F_score`, `M_score`, `rfm_composite_score`
      - `campaign_hints`

### 3.8 Storage (`app/storage/`)

- `app/storage/artifact_store.py`
  - **Class `ArtifactStore`**:
    - Luu/nap model bundle (`clustering_bundle.joblib`).
    - Luu metadata run (`pipeline_meta.json`).
- `app/storage/insight_store.py`
  - **Class `InsightStore`**:
    - Load parquet vao memory dictionary de truy van nhanh.
    - Cung cap `get()`, `get_many()`.

### 3.9 Quality monitoring / test layer (`app/evaluation/`)

- `app/evaluation/thresholds.py`
  - **Class `EvalThresholds`**: bundle nguong tu `Settings` (override qua `RETAIL_EVAL_*`).
- `app/evaluation/baselines.py`
  - **Class `Baseline`** + ham `load_previous_baseline(settings)`:
    - Doc baseline run truoc (`pipeline_meta.json` + `customer_insights.parquet`) truoc khi ghi de.
    - Run dau tien tra ve baseline rong.
- `app/evaluation/checks.py`
  - **Class `CheckResult`**: `{name, category, status, value, threshold, message}`.
  - Gom 5 nhom check:
    - Data quality
    - Feature quality
    - Model quality
    - Business metrics
    - Regression vs previous run
- `app/evaluation/evaluator.py`
  - **Class `EvaluationOrchestrator`**: chay toan bo check va tong hop status.
  - **Class `EvaluationReport`**: payload JSON-safe cho report.
- `app/evaluation/reporter.py`
  - **Class `EvaluationReporter`**:
    - Ghi report latest.
    - Luu history moi run.
    - Co helper tom tat log.

### 3.10 Pipeline orchestration (`app/pipeline/`)

- `app/pipeline/customer_insights_pipeline.py`
  - **Class `PipelineRunResult`**: ket qua run.
  - **Class `CustomerInsightsPipeline`**:
    1. Load data
    2. Profile
    3. Clean
    4. Build feature
    5. Cluster
    6. Recommendation
    7. Extended insights
    8. Save artifacts
    9. Chay quality checks (neu enable)

### 3.11 Jobs (`app/jobs/`)

- `app/jobs/refresh_insights.py`
  - CLI job de build lai toan bo insight/artifact.
  - Dung cho scheduler (Task Scheduler/cron/Airflow).
  - Stdout JSON co them:
    - `evaluation_status`
    - `evaluation_warnings`
    - `evaluation_failed`
    - `evaluation_counts`

### 3.12 Streamlit (`app/streamlit/`)

- `app/streamlit/dashboard.py`
  - Tab `Data Analysis`
  - Tab `Customer Personalization`
  - Tab `Quality Monitoring` (report + trend)

---

## 4) Artifact output sau pipeline

Sau khi chay thanh cong, thu muc `app/artifacts/` co:

- `customer_insights.parquet`
  - Bang insight customer-level cho API query.
- `clustering_bundle.joblib`
  - Model clustering + scaler + pca + thong tin transform.
- `pipeline_meta.json`
  - Metadata run: profile data, cleaning report, feature report, metric clustering, timestamp.
  - Khi `RETAIL_EVAL_ENABLED=true` co them `evaluation_summary` + `evaluation_run_id`.
- `evaluation/evaluation_report.json`
  - Bao cao quality/test moi nhat.
- `evaluation/history/run_<UTC>.json`
  - Snapshot lich su de ve trend.

---

## 5) Huong dan chay chi tiet

### 5.1 Yeu cau

- Dang dung trong thu muc: `05_customer_transaction`
- Co file: `dataset/transaction_data.csv`
- Da cai package: `pip install -r requirements.txt`

### 5.2 Chay pipeline phan tich + build model + quality report (offline)

```powershell
python -m app.jobs.refresh_insights
```

Ket qua mong doi:
- In ra JSON `ok: true`, `insights_path`, `n_customers`
- Neu quality monitoring bat: co them `evaluation_status`, `evaluation_counts`
- File trong `app/artifacts/` duoc tao/cap nhat

Neu muon chay dinh ky:
- Tao lich voi Task Scheduler (Windows), cron, hoac Airflow
- Muc tieu la chay lai job moi khi transaction data cap nhat

### 5.3 Chay API (read-only)

```powershell
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

Kiem tra:

```powershell
curl http://127.0.0.1:8000/health
```

Lay insight 1 customer:

```powershell
curl http://127.0.0.1:8000/v1/customers/17850
```

Lay insight theo danh sach:

```powershell
curl -X POST http://127.0.0.1:8000/v1/customers/insights `
  -H "Content-Type: application/json" `
  -d "{\"customer_ids\":[17850,13047,999999]}"
```

### 5.4 Chay Streamlit Dashboard

```powershell
streamlit run app/streamlit/dashboard.py
```

Neu muon panel API inference hoat dong day du, chay API song song.

### 5.5 Quy trinh van hanh de latency thap

1. Data transaction duoc cap nhat (hang gio/hang ngay)
2. Scheduler chay `python -m app.jobs.refresh_insights`
3. API process doc parquet moi (restart service hoac reload)
4. API tiep tuc chi tra cuu read-only tu RAM

Ly do:
- Tinh toan nang (feature engineering, clustering, scoring) tach khoi API
- API chi phuc vu query insight nhanh

---

## 6) Cau hinh quan trong (env `RETAIL_`)

Vi du:

```powershell
$env:RETAIL_CLUSTER_ALGORITHM="gmm"
$env:RETAIL_K_MIN="2"
$env:RETAIL_K_MAX="12"
$env:RETAIL_TOP_PRODUCTS_PER_SEGMENT="40"
$env:RETAIL_RECOMMENDATIONS_PER_CUSTOMER="10"
python -m app.jobs.refresh_insights
```

Bien cho pipeline/model:
- `RETAIL_TRANSACTION_CSV`
- `RETAIL_ARTIFACTS_DIR`
- `RETAIL_CLUSTER_ALGORITHM` (`kmeans` | `gmm`)
- `RETAIL_K_MIN`, `RETAIL_K_MAX`
- `RETAIL_PCA_VARIANCE_RATIO`
- `RETAIL_TOP_PRODUCTS_PER_SEGMENT`
- `RETAIL_RECOMMENDATIONS_PER_CUSTOMER`

Bien cho quality monitoring (`RETAIL_EVAL_*`):
- `RETAIL_EVAL_ENABLED` (mac dinh `true`)
- `RETAIL_EVAL_MIN_SILHOUETTE` (mac dinh `0.20`)
- `RETAIL_EVAL_MAX_DAVIES_BOULDIN` (mac dinh `2.0`)
- `RETAIL_EVAL_MIN_CLUSTER_FRACTION` (mac dinh `0.03`)
- `RETAIL_EVAL_ROW_COUNT_WARN_PCT` (mac dinh `25.0`)
- `RETAIL_EVAL_N_CUSTOMERS_WARN_PCT` (mac dinh `25.0`)
- `RETAIL_EVAL_PSI_WARN` (mac dinh `0.20`)
- `RETAIL_EVAL_SEGMENT_DISTRIBUTION_WARN_PCT` (mac dinh `15.0`)
- `RETAIL_EVAL_RECOMMENDATION_COVERAGE_WARN` (mac dinh `0.95`)
- `RETAIL_EVAL_SCORE_DRIFT_WARN_PCT` (mac dinh `20.0`)
- `RETAIL_EVAL_SEGMENT_CHANGE_RATE_WARN` (mac dinh `0.30`)
- `RETAIL_EVAL_HISTORY_KEEP_RUNS` (mac dinh `60`)

---

## 7) Test & quality monitoring sau moi lan refresh

Moi lan chay `python -m app.jobs.refresh_insights`, neu `RETAIL_EVAL_ENABLED=true`:

1. Pipeline load baseline run truoc (`pipeline_meta.json` + `customer_insights.parquet`) truoc khi ghi de
2. Chay 5 nhom checks:
   - Data quality: schema, row count drift, null/duplicate/cancel/price/date ratios
   - Feature quality: customer count drift, null/inf, sanity bounds, PSI
   - Model quality: silhouette, davies_bouldin, calinski_harabasz, cluster balance, K stability
   - Business metrics: recommendation coverage, campaign hints coverage, churn/CLV/promoter drift, segment distribution drift
   - Regression: segment change rate + score stability tren tap CustomerID giao nhau
3. Ghi report:
   - `app/artifacts/evaluation/evaluation_report.json` (latest)
   - `app/artifacts/evaluation/history/run_<UTC>.json` (history)
4. Cap nhat metadata:
   - `pipeline_meta.json` co `evaluation_summary` + `evaluation_run_id`
5. Dashboard tab `Quality Monitoring` hien:
   - Overall status badge (`ok` / `warning` / `failed`)
   - Bang chi tiet theo tung nhom checks
   - Trend qua nhieu runs

Luu y:
- Soft-report mode: pipeline khong fail du check warning/failed
- Muc tieu la giam sat va do luong chat luong, khong chan luong du lieu vao pipeline

---

## 8) Troubleshooting nhanh

- Loi `ModuleNotFoundError`
  - Chay lai: `pip install -r requirements.txt`
- API 404 theo `CustomerID`
  - Customer khong co trong parquet hien tai hoac chua chay refresh moi
- Data moi nhung API chua thay doi
  - Chay lai `python -m app.jobs.refresh_insights` va reload/restart API service
- Chua co quality report
  - Kiem tra `RETAIL_EVAL_ENABLED`
  - Chay lai refresh job de tao `app/artifacts/evaluation/evaluation_report.json`

---

## 9) Ghi chu ve insight marketing

- `transactional_promoter_score` la proxy tu hanh vi giao dich
- Khong thay the NPS survey thuc te
- Nen dung ket hop voi campaign tracking va survey neu can danh gia advocacy chuan

