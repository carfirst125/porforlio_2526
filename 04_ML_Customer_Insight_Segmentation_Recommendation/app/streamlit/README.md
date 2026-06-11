# Streamlit Dashboard

## Run

From folder `05_customer_transaction`:

```powershell
streamlit run app/streamlit/dashboard.py
```

## Dashboard Sections

- **Part 1: Data Analysis - Customer Understanding through Transaction Data**
  - KPI cards (transactions, customers, net revenue, cancellation rate)
  - Monthly revenue trend
  - Top countries by revenue
  - Top products by revenue
  - Revenue by hour
  - Segment distribution and segment profile (if `customer_insights.parquet` exists)
  - Monetary and churn score distributions

- **Part 2: Customer Personalization**
  - Input `Customer ID`
  - Chart: transaction value over time for selected customer
  - Feature table from local insights parquet
  - API inference result from `GET /v1/customers/{customer_id}`

## Notes

- This dashboard reads:
  - `dataset/transaction_data.csv`
  - `app/artifacts/customer_insights.parquet` (optional but recommended)
- Make sure FastAPI is running if you want API inference panel to work.
