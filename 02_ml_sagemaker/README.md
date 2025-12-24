# Project 02 - Using sagemaker pipeline to training ML model and saving artifact

This is demo project to express:
 - how using Sagemaker pipeline to implement full flow of ML application in getting data, processing/preparing data, feature engineering and training model.
 - This includes not only folder structure for source code storage, processing data log, but also code flow indicates how ProcessingStep and Pipeline are defined.
 

## Bucket orgainization

All code and log could be stored manually in S3 bucket
s3://bucket/
 - app: for storing code 
 - other: for storing log, processing step output, model artifact (depend on you)

Besides, sagemaker also log running flow in CloudWatch, you could monitor here.

### Sagemaker pipeline source code
```text
app
├── src
├    ├──read_data.py
├	 ├──clean_data.py
├	 ├──prepare_data.py
├	 ├──normalize_data.py
├	 ├──train.py
└── sagemaker_pipeline.py
```

where:
- sagemaker_pipeline.py: this is primary code file which define ProcessingStep, Pipeline flow and indicate relevant source code for particular Processing Step.
- ./src: store source code of each Processing Step.

## Sagemaker pipeline Execution Flow

![Sagemaker Pipeline Execution Flow](https://github.com/carfirst125/porforlio_2526/blob/main/02_ml_sagemaker/images/sagemaker-pipeline-training.png)

Sagemaker first defines ProcessingSteps with working processor. In each step, you need defined the input data source, output destination, and the processing script for the processing step. 
ProcessingSteps in DAG forms a Pipeline.

Sagemaker express the pipeline DAG first, then calling .start() for running.
Note: When pipeline runs, each processing step is run in its separate container. 
SageMaker Pipelines does not introspect or modify the processing script; it only orchestrates container execution and data movement.


