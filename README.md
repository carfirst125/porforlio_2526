# porforlio_2526
This is my portfolio repository performing my competences at period of 2025-2026. 


## Portfolio Summary

This portfolio showcases three end-to-end AI & Machine Learning demo projects that demonstrate my ability to design, build, and deploy production-oriented data and AI systems, covering the full lifecycle from experimentation to cloud deployment.

### Covered Domains & Skills

#### LLM & Agentic AI Systems

- Agentic chatbot architecture (LangChain)
- Retrieval-Augmented Generation (RAG)
- Vector search (FAISS)
- LLM integration and tool-based reasoning

#### Machine Learning & Modeling

- End-to-end ML workflows: data preparation, feature engineering, model training, and evaluation
- Classical ML models and scorecard-style prediction systems
- Model inference via REST APIs

#### MLOps & Workflow Orchestration

- ML pipeline orchestration (Amazon SageMaker Pipelines, Prefect)
- Experiment tracking and model versioning (MLflow)
- Reproducible and modular ML project structure

#### Cloud, Infrastructure & Deployment

- Containerized ML/AI applications using Docker
- Infrastructure as Code (Terraform)
- CI/CD pipelines for automated build and deployment
- Deployment on cloud platforms (Azure, AWS)



## Project Highlights

### Project 01 – Agentic Chatbot (LLM + Cloud Deployment)

- Designed and implemented an agentic LLM chatbot using LangChain, supporting Retrieval-Augmented Generation (RAG) with FAISS vector store and embedding-based similarity search.

- Built a production-ready FastAPI inference service, containerized the application using Docker, and managed container images via Azure Container Registry (ACR).

- Provisioned and deployed the entire cloud infrastructure using Terraform (Infrastructure as Code), and implemented CI/CD pipelines with Azure DevOps Pipelines to automate container build, registry push, and deployment to Azure Container Apps (ACA).

### Project 02 – Training ML Pipeline with Amazon SageMaker

- Implemented an end-to-end machine learning pipeline using Amazon SageMaker Pipelines, modeling the workflow as a DAG composed of multiple ProcessingSteps.

- Utilized ScriptProcessor for data ingestion, preprocessing, and feature engineering, with all intermediate outputs and model artifacts persisted to Amazon S3.

- Demonstrated best practices for modular ML pipeline design, reproducible execution, and separation of data processing, training logic, and artifact management in a managed cloud ML platform.

### Project 03 – ML Application with Progressive MLOps Architecture

- Developed an end-to-end machine learning application that evolves from a standalone Python training script to an orchestrated ML workflow using Prefect for task and flow management.

- Integrated MLflow for experiment tracking, model versioning, and artifact management, enabling reproducible training and controlled model promotion.

- Exposed trained models through a RESTful inference API using FastAPI, supporting real-time prediction requests and demonstrating the transition from experimental ML code to a production-oriented MLOps workflow.

# END