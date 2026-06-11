# sagemaker_pipeline.py
#
# Action Steps:
# Step 1: You need to python source code in your s3 bucket (s3://<bucket>/app/src) to notebook working folder first.
#   !aws s3 cp s3://<YOUR-BUCKER>/app/src ./src --recursive
#   !aws s3 cp s3://sagemaker-house-price-prediction-bucket/app/src ./src --recursive (mine)
# Step 2: Run sagemaker_pipeline.py (this file)
#   Note: processingstep run by sagemaker in its assigned container, let check s3 for your output result or run Monitor code at last of file.
# 
# SUCCEEDED



import os
import json
import time
import boto3
import sagemaker

from sagemaker.session import Session
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker import image_uris
from sagemaker import get_execution_role

# =====================================================================
# NOTEBOOK MODE
# =====================================================================
region = boto3.session.Session().region_name
print(f"sagemaker region: {region}")
role_arn = get_execution_role()
boto_sess = boto3.Session(region_name=region)
sagemaker_session = Session()

bucket = "sagemaker-house-price-prediction-bucket"
SAGEMAKER_PROCESSING_RS = "ml.t3.medium" 
SAGEMAKER_TRAIN_RS = "ml.m5.large" 
#SAGEMAKER_COMPUTE_RS = "ml.m5.large" #"ml.t3.medium" #"local" 

print("Region:", region)
print("Role:", role_arn)

# =====================================================================
# PARAMETERS
# =====================================================================
raw_data_s3_param = ParameterString(
    name="RawDataS3",
    default_value=f"s3://{bucket}/data/raw"
)

processing_data_s3 = f"s3://{bucket}/data/interim"
model_output_s3     = f"s3://{bucket}/models"
code_src_s3         = f"s3://{bucket}/app"

# =====================================================================
# PROCESSING STEPS
# =====================================================================
sklearn_processor = SKLearnProcessor(
    framework_version="1.4-2",          # giữ nguyên version
    role=role_arn,
    instance_type=SAGEMAKER_PROCESSING_RS,
    instance_count=1,
    sagemaker_session=sagemaker_session
)

# STEP 1: READ DATA
read_data_step = ProcessingStep(
    name="ReadData",
    processor=sklearn_processor,
    inputs=[ ProcessingInput(source=raw_data_s3_param, destination="/opt/ml/processing/input") ],
    outputs=[
        ProcessingOutput(
            output_name="read_output",
            source="/opt/ml/processing/output",
            destination=f"{processing_data_s3}/input"
        )
    ],
    code="src/read_data.py"
)

# STEP 2: CLEAN
clean_data_step = ProcessingStep(
    name="CleanData",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(
            source=read_data_step.properties.ProcessingOutputConfig.Outputs["read_output"].S3Output.S3Uri,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="clean_output",
            source="/opt/ml/processing/output",
            destination=f"{processing_data_s3}/clean"
        )
    ],
    code="src/clean_data.py"
)

# STEP 3: PREPARE
prepare_data_step = ProcessingStep(
    name="PrepareData",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(
            source=clean_data_step.properties.ProcessingOutputConfig.Outputs["clean_output"].S3Output.S3Uri,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="prepared_output",
            source="/opt/ml/processing/output",
            destination=f"{processing_data_s3}/prepared"
        )
    ],
    code="src/prepare_data.py"
)

# STEP 4: NORMALIZE
normalize_data_step = ProcessingStep(
    name="NormalizeData",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(
            source=prepare_data_step.properties.ProcessingOutputConfig.Outputs["prepared_output"].S3Output.S3Uri,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="normalized_output",
            source="/opt/ml/processing/output",
            destination=f"{processing_data_s3}/normalized"
        )
    ],
    code="src/normalize_data.py"
)

# =====================================================================
# TRAINING STEP — SCRIPT MODE (XGBOOST)
# =====================================================================
xgb_image = image_uris.retrieve(
    framework="xgboost",
    region=region,
    version="1.7-1"
)

xgb_estimator = Estimator(
    image_uri=xgb_image,
    role=role_arn,
    instance_count=1,
    instance_type=SAGEMAKER_TRAIN_RS,
    output_path=model_output_s3,
    entry_point="train.py",             # SCRIPT MODE
    source_dir="src",                   # thư mục local sẽ upload lên S3 tự động
    sagemaker_session=sagemaker_session
)

train_input = TrainingInput(
    s3_data=normalize_data_step.properties.ProcessingOutputConfig.Outputs["normalized_output"].S3Output.S3Uri,
    content_type="text/csv"
)

train_step = TrainingStep(
    name="TrainModel",
    estimator=xgb_estimator,
    inputs={"train": train_input}
)

# =====================================================================
# PIPELINE
# =====================================================================
pipeline = Pipeline(
    name="HousePriceForecastPipeline",
    steps=[
        read_data_step,
        clean_data_step,
        prepare_data_step,
        normalize_data_step,
        train_step
    ],
    parameters=[raw_data_s3_param],
    sagemaker_session=sagemaker_session
)

print("Pipeline created.")

pipeline.upsert(role_arn=role_arn)
execution = pipeline.start()
print("Pipeline execution started:", execution.arn)


###########################################################
# Log: check status of sagemaker pipelines through execution.arn

# Monitoring code
# This code for you if would like to following how sagemaker ProcessingStep going on.
'''
import boto3

# Lấy pipeline execution ARN từ output trước đó
execution_arn = execution.arn #execution.arn #"arn:aws:sagemaker:ap-southeast-1:062370857011:pipeline/HousePriceForecastPipeline/execution/oub5p1pkvo4x"

# Tạo SageMaker client
sagemaker_client = boto3.client('sagemaker', region_name='ap-southeast-1')

# Kiểm tra trạng thái execution
response = sagemaker_client.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
print(f"Status: {response['PipelineExecutionStatus']}")
print(f"Start Time: {response.get('CreationTime')}")
if 'LastModifiedTime' in response:
    print(f"Last Modified: {response['LastModifiedTime']}")


#--------------------------------------------------------------
# Lấy danh sách các steps trong execution
steps_response = sagemaker_client.list_pipeline_execution_steps(PipelineExecutionArn=execution_arn)

for step in steps_response['PipelineExecutionSteps']:
    print(f"Step: {step['StepName']}")
    print(f"Status: {step['StepStatus']}")
    if 'FailureReason' in step:
        print(f"Error: {step['FailureReason']}")
    print("---")


#--------------------------------------------------------------
import time

def monitor_pipeline_execution(execution_arn):
    while True:
        response = sagemaker_client.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
        status = response['PipelineExecutionStatus']
        print(f"Current status: {status}")
        
        if status in ['Succeeded', 'Failed', 'Stopped']:
            break
            
        time.sleep(30)  # Kiểm tra mỗi 30 giây
    
    return status

final_status = monitor_pipeline_execution(execution_arn)
print(f"Pipeline finished with status: {final_status}")


#--------------------------------------------------------------
# Nếu có step bị fail, xem CloudWatch logs
steps_response = sagemaker_client.list_pipeline_execution_steps(PipelineExecutionArn=execution_arn)

for step in steps_response['PipelineExecutionSteps']:
    if step['StepStatus'] == 'Failed':
        print(f"Failed step: {step['StepName']}")
        if 'Metadata' in step and 'ProcessingJob' in step['Metadata']:
            job_name = step['Metadata']['ProcessingJob']['Arn'].split('/')[-1]
            print(f"Job name: {job_name}")
            print("Check CloudWatch logs for detailed error messages")

'''