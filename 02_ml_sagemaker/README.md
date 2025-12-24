# Project 02 - Using sagemaker pipeline to train ML model and saving artifact

## Overview

This is demo project to express:
 - how using Sagemaker pipeline to implement full flow of ML application in getting data, processing/preparing data, feature engineering and training model.
 - This includes not only folder structure for source code storage, processing data log, but also code flow indicates how ProcessingStep and Pipeline are defined.
 

## Bucket organization

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

__where__:
- sagemaker_pipeline.py: this is primary code file which define ProcessingStep, Pipeline flow and indicate relevant source code for particular Processing Step.
- ./src: store source code of each Processing Step.

## Sagemaker pipeline Execution Flow

![Sagemaker Pipeline Execution Flow](https://github.com/carfirst125/porforlio_2526/blob/main/02_ml_sagemaker/images/sagemaker-pipeline-training.png)

Sagemaker first defines ProcessingSteps with working processor. In each step, you need defined the input data source, output destination, and the processing script for the processing step. 
ProcessingSteps in DAG forms a Pipeline.

Sagemaker express the pipeline DAG first, then calling .start() for running.

**Note**: 
- When pipeline runs, each processing step is run in its separate container. 
- SageMaker Pipelines does not introspect or modify the processing script; it only orchestrates container execution and data movement.

After Pipeline completed, training model is stored in your indicated path.

For debug, openning Cloud Watch and check the log for error information.


## Additions

From Local: For creating folder structure and clone/push new update code to s3 bucket, you could refer 2 below script:

```
1 - Creating any folder structure as you wish, I refer one as in my code, just run:
python create_s3_folder_structure_ml_project_sagemaker.py

2 - Clone data in your s3 bucket and push/update your change in local back to s3 bucket like new file, update file, new folder
python clone_s3_bucket_update_push.py --mode="s3_pull"    # download s3 bucket on cloud to local
python clone_s3_bucket_update_push.py --mode="s3_push"    # push/update s3 bucket on cloud with your local change.
```

**Note**: for access AWS cloud from local, you need to provide credentials in **.env**


