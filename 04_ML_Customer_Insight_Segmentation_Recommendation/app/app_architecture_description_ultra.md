# Ultra-Short Prompt for Flux Max (Technical Block Diagram)

Draw a single clean **technical block diagram** (16:9 landscape) titled:
**“05_customer_transaction - Offline Customer Insights Architecture”**.

Show 6 left-to-right zones:
1) **Data Source**: `dataset/transaction_data.csv` (transaction fields).
2) **Offline Pipeline** (`app/pipeline/customer_insights_pipeline.py`): load -> profile -> clean -> feature engineering (RFM + behavior) -> clustering (`kmeans/gmm`, scaler, optional PCA, silhouette K) -> extended scoring (churn risk, CLV, transactional promoter proxy, campaign hints) -> segment recommender.
3) **Artifact Storage** (`app/artifacts`): `customer_insights.parquet`, `clustering_bundle.joblib`, `pipeline_meta.json`.
4) **FastAPI Read-Only Serving** (`app/api/main.py`): startup loads parquet into `InsightStore` in-memory index; endpoints `GET /health`, `GET /v1/customers/{id}`, `POST /v1/customers/insights`; serialization to JSON-safe payloads.
5) **Consumers**: Streamlit dashboard (`app/streamlit/dashboard.py`) + external API clients. Dashboard reads parquet for analytics and calls API for per-customer lookup.
6) **Operations/Config**: scheduler (Task Scheduler/cron/Airflow) triggers `python -m app.jobs.refresh_insights`; `app/config.py` with `RETAIL_` env vars configures pipeline and API.

Flow rules:
- Offline pipeline **writes** artifacts.
- API and dashboard **read** artifacts (parquet).
- API is **read-only** and does **no training/heavy compute** at request time.
- Add numbered workflow 1-9 from CSV ingest to client/API response.

Style:
- Color by zone, rounded rectangles, clear arrows, minimal crossing lines.
- Solid arrows = data flow, dashed arrows = control/config flow.
- Include legend.
