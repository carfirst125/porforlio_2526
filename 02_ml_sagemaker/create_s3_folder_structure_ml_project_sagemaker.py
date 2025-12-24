############################################################
# Create S3 Folder Structure using for ML project sagemaker
# Date: 25/11/2025
#
'''
s3://your-bucket/
│
├── data/
│   ├── raw/                         # Raw data từ sources (CSV, JSON, Parquet)
│   │   ├── dataset1.csv
│   │   ├── dataset2.parquet
│   │   └── 2024-11-01-dump/
│   │
│   ├── interim/                     # Step processing tạo ra (processing job output)
│   │   ├── cleaned/
│   │   ├── features/
│   │   └── splits/                  # train/test/val split
│   │
│   └── processed/                   # Data final (ready for training)
│       ├── train/
│       ├── val/
│       └── test/
│
├── pipelines/                       # Pipeline definition + metadata
│   ├── pipeline-name/
│   │   ├── executions/
│   │   ├── logs/
│   │   └── pipeline-definition.json
│
├── models/
│   ├── model-name/
│   │   ├── artifacts/               # model.tar.gz từ training job
│   │   │   ├── version-01/
│   │   │   ├── version-02/
│   │   │   └── latest/
│   │   ├── metadata/                # metrics, hyperparameters
│   │   ├── evaluation/              # evaluation report (usually processing job)
│   │   └── monitoring/              # model monitor result
│
├── inference/
│   ├── batch-transform/
│   ├── endpoints/
│   │   └── endpoint-name/
│   └── logs/
│
└── temp/
    ├── processing-jobs/             # Temp output từ processing container
    ├── training-jobs/               # Temp input & logs
    └── cache/

'''

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
load_dotenv()
import os

# ====== AWS Credentials ======
aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY") 
aws_region            = os.getenv("AWS_REGION", "ap-southeast-1")

# ====== Bucket Name ======
bucket_name = os.getenv("PROJECT_BUCKET_NAME")

# ====== Khởi tạo session ======
session = boto3.Session(
    aws_access_key_id     = aws_access_key_id,
    aws_secret_access_key = aws_secret_access_key,
    region_name           = aws_region
)

s3 = session.client("s3")
s3_resource = session.resource("s3")


# ==============================
#   CHECK BUCKET EXISTENCE
# ==============================
def bucket_exists(bucket_name):
    try:
        s3.head_bucket(Bucket=bucket_name)
        return True
    except ClientError:
        return False


# ==============================
#   CREATE BUCKET
# ==============================
def create_bucket():   

    try:
        if aws_region == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': aws_region}
            )
        print(f"Bucket is created: {bucket_name}")
    except ClientError as e:
        print("Error when creating bucket:", e)




# ==============================
#   CHECK FOLDER EXISTENCE
# ==============================
def folder_exists(bucket_name, prefix):
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=1)
    return "Contents" in response


# ====== Folder structure cần tạo ======
folders = [
    "data/raw/",
    "data/raw/2024-11-01-dump/",
    "data/interim/",
    "data/interim/cleaned/",
    "data/interim/features/",
    "data/interim/splits/",
    "data/processed/train/",
    "data/processed/val/",
    "data/processed/test/",
    
    "pipelines/pipeline-name/executions/",
    "pipelines/pipeline-name/logs/",
    "pipelines/pipeline-name/",
    
    "models/model-name/artifacts/version-01/",
    "models/model-name/artifacts/version-02/",
    "models/model-name/artifacts/latest/",
    "models/model-name/metadata/",
    "models/model-name/evaluation/",
    "models/model-name/monitoring/",
    
    "inference/batch-transform/",
    "inference/endpoints/endpoint-name/",
    "inference/logs/",
    
    "temp/processing-jobs/",
    "temp/training-jobs/",
    "temp/cache/"
]


# ==============================
#   CREATE FOLDERS
# ==============================
def create_folders():
    for folder in folders:
        if folder_exists(bucket_name, folder):
            print(f"Folder exists: s3://{bucket_name}/{folder}")
        else:
            key = folder + ".keep"
            s3.put_object(Bucket=bucket_name, Key=key, Body=b"")
            print(f"Created folder: s3://{bucket_name}/{folder}")

def create_folder_structure():
    
    if bucket_exists(bucket_name):
        print(f"Bucket đã tồn tại: {bucket_name}")
    else:
        print(f"Create bucket {bucket_name}")
        create_bucket()        
     
    create_folders()

#########################################

if __name__== "__main__":
    
    create_folder_structure()

## END
