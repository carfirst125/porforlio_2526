import boto3
from dotenv import load_dotenv
load_dotenv()
import os

AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION                = os.getenv("AWS_REGION", "ap-southeast-1")
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=REGION
)

sts = session.client("sts")
print(sts.get_caller_identity())
