# Project 03 - Implementing ML project (training, exposing forecasting REST API) in three versions 

## Project Overview

This project is an end-to-end machine learning application that builds a scorecard model to forecast the probability of customer purchase conversion.

This project is designed to illustrate best practices in transitioning from experimental machine learning code to a scalable and maintainable production-ready workflow.

### Tech Stack
- Python
- Prefect
- MLflow
- FastAPI
- Scikit-learn / XGBoost (if applicable)

### Application Implementations

There are 3 versions are performed:

**Normal Python**

A standalone Python implementation suitable for local execution.
This version does not support orchestration, model versioning, or parallel execution.

**Using an Orchestrator**

The monolithic script is decomposed into discrete tasks, with execution order and dependencies managed by an orchestrator (e.g., Prefect or Airflow). In this project, Prefect is selected.

**Using an Orchestrator with Model Version Management**

An enhanced workflow that uses Prefect for orchestration and MLflow for experiment tracking and model version management.

### Covered Functionalities in each Version

**1. Data Preparation and Model Training**

```
  - Data Preparation
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

**2. Model Inference via REST API**

The trained model is exposed through a RESTful inference API using FastAPI.

Clients can send prediction requests to the API endpoint and receive inference results in real time.

---

## Implementation

```
  - normal
     ├── ml_scorecard.py          # Data preparation and model training
     ├        > python ml_scorecard.py
     ├── fastapi_scoring_app.py   # fastapi inference 
     ├        > uvicorn fastapi_scoring_app:app --reload --port 8000
     ├── client_call_api.py       # client request api url
     ├        > python client_call_api.py

  - using_prefect
     ├── ml_scorecard_prefect.py  # Data preparation and model training with task and flow
     ├── fastapi_prefect.py       # fastapi inference 
     ├── client_call_api.py       # client request api url

  - using_prefect_mlflow
     ├── ml_scorecard_prefect_mlflow.py  # Data preparation, model training with tasks/flow and model managed by mlflow
     ├── fastapi_mlflow_inference.py     # fastapi inference 
     ├── client_call_api.py              # client request api url

```

## Working Images

### Prefect 

**Orchestrator - Flow of Tasks**

The following image illustrates the task execution flow managed by Prefect, showing how individual tasks are orchestrated and executed as part of the ML pipeline.

![Prefect UI - Flow of Tasks run](https://github.com/carfirst125/porforlio_2526/blob/main/03_ml_app/images/prefect_flow_run.png)

### MLflow 

**Model version storage** 

This image shows how trained models are stored and versioned in MLflow, enabling reproducibility and traceability across experiments.

![Prefect UI - Flow of Tasks run](https://github.com/carfirst125/porforlio_2526/blob/main/03_ml_app/images/mlflow_model_version_storage.png)

**Best artifact registration**

The following image demonstrates the registration of the best-performing model artifact into the MLflow Model Registry, allowing consistent model promotion and deployment.

![Prefect UI - Flow of Tasks run](https://github.com/carfirst125/porforlio_2526/blob/main/03_ml_app/images/mlflow_model_register.png)


