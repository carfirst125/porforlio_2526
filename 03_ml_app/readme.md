
This is ML application project creating scorecard in forecasting customer probability in purchasing conversion currently:

The project performs following operations:

1. Data Preparation and Training Model
```
  - Data Preparation:
     ├── EDA
     ├── Missing Value Imputation
     ├── Outlier Treatment
     ├── Data Transformation
     ├── Feature Engineering
     ├── Feature Selection
     ├── Data Normalization
     ├── Imbalance processing
  - Model Selection
  - Model Training
  - Model Evaluation
```

2. FastAPI Inference
Expose API for inference model
Client requests API url to get inference response


The app is implemented in 03 versions:

1. Normal Python: Suitable for local running, not support orchestrator, model version management or parallel running.
2. Using Orchestrator: Decompose a linear/standalone script into discrete tasks and define their execution flow using an orchestrator (e.g., Prefect or Airflow).
3. Using Orchestrator and Model version Management: using Prefect as orchestrator and MLflow to manage model version.
