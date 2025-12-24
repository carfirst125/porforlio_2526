import boto3
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ==== CẤU HÌNH ====
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")  # mặc định nếu không có
BUCKET_NAME = os.getenv("PROJECT_BUCKET_NAME")     # Tên bucket S3
LOCAL_DIR = './s3_bucket'                         # Thư mục local để lưu file
DELETE_EXTRA_S3 = True                             # Nếu True, xóa file trên S3 không tồn tại local

# ==== KHỞI TẠO S3 RESOURCE ====
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)
s3 = session.resource('s3')
bucket = s3.Bucket(BUCKET_NAME)

# ==== HÀM PULL TỪ S3 VỀ LOCAL ====
def s3_pull():
    print(f"Tải tất cả file từ bucket {BUCKET_NAME} về {LOCAL_DIR}...")
    for obj in bucket.objects.all():
        local_path = os.path.join(LOCAL_DIR, obj.key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        bucket.download_file(obj.key, local_path)
    print("Hoàn tất tải file.")

# ==== HÀM PUSH TỪ LOCAL LÊN S3 ====
def s3_push():
    print("Đang upload file lên S3...")
    local_files = set()
    for root, dirs, files in os.walk(LOCAL_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            s3_key = os.path.relpath(local_path, LOCAL_DIR).replace("\\", "/")
            bucket.upload_file(local_path, s3_key)
            local_files.add(s3_key)
    print("Hoàn tất upload.")

    if DELETE_EXTRA_S3:
        print("Kiểm tra và xóa file trên S3 không tồn tại local...")
        for obj in bucket.objects.all():
            if obj.key not in local_files:
                print(f"Xóa {obj.key} trên S3")
                obj.delete()
        print("Hoàn tất xóa file thừa trên S3.")

# ==== MAIN ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync S3 bucket với local folder")
    parser.add_argument(
        "--mode",
        choices=["s3_pull", "s3_push"],
        required=True,
        help="Chọn mode: s3_pull (tải từ S3 về local) hoặc s3_push (upload từ local lên S3)"
    )
    args = parser.parse_args()

    if args.mode == "s3_pull":
        s3_pull()
    elif args.mode == "s3_push":
        s3_push()
