# Prompt for Flux Max: Technical Block Diagram of `05_customer_transaction` App

Create a **professional technical block diagram** (single image, landscape orientation, 16:9) that shows the **full architecture and workflow** of the application `05_customer_transaction`.

Use a clean engineering style: white or light background, thin connector lines, rounded rectangles for components, color-coded layers, clear directional arrows, and concise labels.  
Target audience: data engineers, ML engineers, backend engineers, and marketing analytics stakeholders.

---

## 1) Diagram Goal

Visualize the complete end-to-end system with:
- Offline data processing and model/insight generation pipeline
- Artifact persistence and versioned outputs
- Read-only low-latency API serving layer
- Streamlit analytics + personalization dashboard layer
- External actors and runtime operations (scheduler, API clients, marketing users)

Show not only components, but also **workflow sequence**, **data contracts**, and **read/write boundaries**.

---

## 2) High-Level Layout (left to right)

Organize into 6 vertical zones from left to right:

1. **Data Source Zone**
2. **Offline Pipeline Zone (Batch Compute)**
3. **Artifact Storage Zone**
4. **Online Serving Zone (FastAPI)**
5. **Consumer Applications Zone (Dashboard / Clients)**
6. **Operations & Configuration Zone**

Add a title at top:
**“05_customer_transaction - Customer Insights Platform Architecture (Offline-First + Read-Only API)”**

---

## 3) Detailed Components by Zone

### Zone 1 - Data Source

Block:
- **Transaction Dataset (CSV)**
  - Path example: `dataset/transaction_data.csv`
  - Key columns: `CustomerID`, `InvoiceDate`, `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `UnitPrice`, `Country`

Arrow to Offline Pipeline with label:
- “Batch read (CSV ingestion)”

---

### Zone 2 - Offline Pipeline (Batch Compute in `app/pipeline/customer_insights_pipeline.py`)

Represent as a large container named:
- **CustomerInsightsPipeline (Orchestrator)**

Inside, place sequential sub-blocks connected by arrows:

1. **Data Loader (`app/data/loader.py`)**
   - Reads CSV
   - Validates required schema
   - Parses `InvoiceDate`

2. **Dataset Profiling (`app/preprocessing/profile.py`)**
   - Computes null ratios, duplicates, cancellation ratio, skewness
   - Produces dataset profile for adaptive cleaning/model decisions

3. **Transaction Cleaning (`app/preprocessing/cleaning.py`)**
   - Drops invalid rows (missing IDs/dates/descriptions)
   - Filters non-positive prices
   - Flags cancellations
   - Trims extreme quantity outliers
   - Produces cleaning report

4. **Feature Engineering (`app/features/customer_features.py`)**
   - Builds customer-level features
   - RFM + behavioral features:
     - `RecencyDays`, `Frequency`, `Monetary`
     - `AvgOrderValue`, `OrdersPerMonth`, `UniqueProducts`
     - `CancelRate`, `IsUK`, `TenureDays`, `SpendTrendScore`

5. **Clustering Training (`app/modeling/cluster_model.py`)**
   - Adaptive transforms (`log1p`, optional power transform)
   - StandardScaler
   - Optional PCA
   - Selects K via silhouette (within config bounds)
   - Algorithm configurable: `kmeans` or `gmm`
   - Output: segment label per customer + model bundle

6. **Extended Insight Scoring (`app/insights/extended_metrics.py`)**
   - Computes marketing scores:
     - `churn_risk_score`
     - `clv_estimate`
     - `transactional_promoter_score` (transaction-based proxy, not survey NPS)
     - `R_score`, `F_score`, `M_score`, `rfm_composite_score`
     - `campaign_hints`

7. **Segment-Based Recommender (`app/modeling/recommender.py`)**
   - Learns top products by segment
   - Produces `recommended_stock_codes` per customer

8. **Final Insight Assembly**
   - Merges customer features + segment + scores + recommendations
   - Output table at customer granularity

Add side annotation on this container:
- “Heavy compute occurs offline only (not in API requests)”

---

### Zone 3 - Artifact Storage (`app/storage`)

Use a storage container with 3 artifact blocks:

1. **`customer_insights.parquet`**
   - Main customer-level insight store used by API and dashboard

2. **`clustering_bundle.joblib`**
   - Serialized model package:
     - model, scaler, PCA, transformers, selected features, algorithm, K

3. **`pipeline_meta.json`**
   - Run metadata:
     - generation timestamp, dataset profile, cleaning report, feature report, clustering metrics, source path

Show arrows:
- Offline Pipeline **writes** all 3 artifacts
- API InsightStore **reads** `customer_insights.parquet`
- Dashboard **reads** `customer_insights.parquet` directly for local analytics

Use write/read labels explicitly:
- “write artifacts”
- “read-only load”

---

### Zone 4 - Online Serving (FastAPI in `app/api/main.py`)

Container name:
- **FastAPI Read-Only Service**

Internal blocks:

1. **App Lifespan Startup**
   - Instantiates `InsightStore`
   - Loads `customer_insights.parquet` into in-memory dictionary index (`CustomerID -> record`)
   - Purpose: low-latency retrieval

2. **InsightStore (`app/storage/insight_store.py`)**
   - `load()`, `get(customer_id)`, `get_many(customer_ids)`
   - Thread-safe in-memory access

3. **Serialization Layer (`app/api/serialize.py`)**
   - Converts pandas/numpy/datetime values to JSON-safe response payloads

4. **REST Endpoints**
   - `GET /health`
   - `GET /v1/customers/{customer_id}`
   - `POST /v1/customers/insights` (batch IDs)

Add a bold note near API container:
- **“No model training or heavy feature computation during request handling.”**

Show incoming arrows from:
- API clients
- Streamlit personalization panel (for live lookup)

---

### Zone 5 - Consumer Applications

Add two main consumer blocks:

1. **Streamlit Dashboard (`app/streamlit/dashboard.py`)**
   - Part 1: Data Analysis
     - KPIs, revenue trends, countries/products/hour charts
     - segment distribution and profile
     - churn and monetary distributions
   - Part 2: Customer Personalization
     - user inputs `Customer ID`
     - displays customer transaction history
     - displays local customer features from parquet
     - calls FastAPI for live inference display (JSON)

2. **External API Consumers**
   - Internal tools, scripts, or downstream applications querying customer insights

Arrows:
- Dashboard -> API (HTTP request for selected customer)
- API -> Dashboard/Clients (JSON response)
- Dashboard <- Artifacts (`customer_insights.parquet`) for local visual analytics

---

### Zone 6 - Operations & Configuration

Add operations blocks:

1. **Refresh Job (`python -m app.jobs.refresh_insights`)**
   - Triggers full offline pipeline rebuild
   - Intended for scheduled execution

2. **Scheduler**
   - Windows Task Scheduler / cron / Airflow
   - Periodically runs refresh job

3. **Config Management (`app/config.py`)**
   - `Settings` via environment variables prefix `RETAIL_`
   - Controls paths, clustering algorithm, K range, PCA ratio, recommendation limits, API host/port

4. **Service Runtime Command**
   - `uvicorn app.api.main:app --host 0.0.0.0 --port 8000`

Show control arrows:
- Scheduler -> Refresh Job -> Offline Pipeline
- Config -> Pipeline, Job, API (dashed “configuration flow” lines)

---

## 4) Workflow Overlay (numbered callouts)

Add numbered markers (1..9) on arrows to show workflow:

1. Read transaction CSV  
2. Profile + clean + transform data  
3. Build customer features  
4. Train clustering + assign segments  
5. Compute extended marketing insights + recommendations  
6. Persist artifacts (`parquet`, `joblib`, `meta.json`)  
7. API startup loads parquet into in-memory index  
8. Clients request single/batch customer insights via REST  
9. Dashboard combines local analytics + API personalization lookup

---

## 5) Architecture Principles (small side panel)

Add a compact “Design Principles” panel listing:
- **Offline-first processing**
- **Online read-only serving**
- **Low-latency in-memory lookup**
- **Decoupled refresh from API runtime**
- **Configurable via environment variables**
- **Artifact-driven reproducibility**

---

## 6) Visual Styling Requirements

- Use distinct colors per zone:
  - Data Source: gray/blue
  - Offline Pipeline: teal
  - Storage: amber
  - API Serving: purple
  - Consumer Apps: green
  - Operations/Config: orange
- Solid arrows = data flow
- Dashed arrows = control/configuration flow
- Add legend for arrow styles and color zones
- Keep labels readable and technical (no marketing slogans)
- Ensure no crossing lines where avoidable; route connectors cleanly

---

## 7) Output Expectations

The final image should look like an engineering architecture board used for design review:
- detailed but clean
- modular blocks with clear boundaries
- explicit workflow and dependency relationships
- immediately understandable by both backend and data teams

---

# Short Prompt Version (for tighter token/character limits)

Create a single professional **technical block diagram** (16:9 landscape) for the app **`05_customer_transaction`**, titled:
**“05_customer_transaction - Customer Insights Platform Architecture (Offline-First + Read-Only API)”**.

Use 6 left-to-right zones:
1) Data Source, 2) Offline Pipeline, 3) Artifact Storage, 4) FastAPI Serving, 5) Consumers, 6) Operations/Config.

Show these blocks and flows:

- **Data Source**
  - `dataset/transaction_data.csv` (CustomerID, InvoiceDate, InvoiceNo, StockCode, Description, Quantity, UnitPrice, Country)
  - Arrow: batch read -> Offline Pipeline

- **Offline Pipeline (`app/pipeline/customer_insights_pipeline.py`)**
  - Data Loader (`app/data/loader.py`)
  - Dataset Profile (`app/preprocessing/profile.py`)
  - Transaction Cleaner (`app/preprocessing/cleaning.py`)
  - Feature Builder (`app/features/customer_features.py`) with RFM + behavioral features
  - Clustering Trainer (`app/modeling/cluster_model.py`): scaler, optional PCA, silhouette-based K, `kmeans`/`gmm`
  - Extended Insights (`app/insights/extended_metrics.py`): churn risk, CLV, transactional promoter proxy, RFM composite, campaign hints
  - Segment Recommender (`app/modeling/recommender.py`): recommended stock codes
  - Output: customer-level insights table
  - Note: heavy compute is offline only

- **Artifact Storage (`app/storage`)**
  - `customer_insights.parquet` (main serving store)
  - `clustering_bundle.joblib` (model/scaler/PCA/transform metadata)
  - `pipeline_meta.json` (profile, cleaning report, feature report, clustering metrics, timestamp)
  - Pipeline writes all artifacts

- **FastAPI Read-Only Serving (`app/api/main.py`)**
  - Startup lifespan loads parquet via `InsightStore` (`app/storage/insight_store.py`) into memory index (`CustomerID -> record`)
  - Serialization layer (`app/api/serialize.py`) for JSON-safe output
  - Endpoints: `GET /health`, `GET /v1/customers/{customer_id}`, `POST /v1/customers/insights`
  - Bold note: no training/heavy computation during requests

- **Consumers**
  - Streamlit dashboard (`app/streamlit/dashboard.py`):
    - Part 1: transaction analytics and segment visuals
    - Part 2: customer personalization (input customer ID, history, features, API JSON lookup)
  - External API clients/tools
  - Dashboard reads parquet for analytics and calls API for live customer lookup

- **Operations/Config**
  - Refresh job: `python -m app.jobs.refresh_insights`
  - Scheduler: Task Scheduler / cron / Airflow
  - Config: `app/config.py` (`RETAIL_` env vars control paths, clustering params, recommender limits, API host/port)
  - API runtime command: `uvicorn app.api.main:app --host 0.0.0.0 --port 8000`
  - Dashed control flow: Scheduler -> Refresh Job -> Pipeline; Config -> Pipeline/API/Job

Add numbered workflow callouts (1-9):
1) read CSV, 2) profile+clean, 3) build features, 4) cluster+segment, 5) score+recommend, 6) write artifacts, 7) API loads parquet to memory, 8) clients query REST, 9) dashboard combines local analytics + API lookup.

Style requirements:
- Color by zone (source gray/blue, pipeline teal, storage amber, API purple, consumers green, ops orange)
- Solid arrows for data flow, dashed arrows for control/config
- Include legend
- Clean routing, minimal line crossing, concise technical labels.

