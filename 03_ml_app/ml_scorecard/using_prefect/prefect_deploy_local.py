##############################################
# prefect_deploy_local.py
# 
'''
from prefect import flow
from prefect.filesystems import LocalFileSystem
from ml_scorecard_prefect import customer_ml_flow


@flow
def train_flow():
    print("Training...")
    customer_ml_flow(n_customers=1000, n_tx=8000, iv_threshold=0.01)

if __name__ == "__main__":
    storage = LocalFileSystem(basepath=".")
    train_flow.deploy(
        name="train_daily",
        work_pool_name="default-agent-pool",
        storage=storage,
        entrypoint="prefect_deploy_local.py:train_flow",
    )

'''
'''
from prefect import flow
from prefect.deployments import Deployment
from prefect.filesystems import LocalFileSystem
from prefect.client.schemas.schedules import CronSchedule
from ml_scorecard_prefect import customer_ml_flow

@flow
def train_flow():
    print("Training...")
    customer_ml_flow(n_customers=1000, n_tx=8000, iv_threshold=0.01)

if __name__ == "__main__":
    # Khai báo storage local
    storage = LocalFileSystem(basepath=".")

    # Tạo deployment
    deployment = Deployment.build_from_flow(
        flow=train_flow,
        name="train_daily",
        work_pool_name="default-agent-pool",
        entrypoint="prefect_deploy_local.py:train_flow",
        storage=storage,
        schedule=CronSchedule(cron="0 1 * * *"),  # mỗi ngày 1AM
    )

    # Apply deployment vào Prefect server
    deployment.apply()

    print("✅ Deployment created successfully!")
'''

'''
from prefect import flow
from prefect.client.schemas.schedules import CronSchedule

@flow
def train_flow():
    print("Training...")

if __name__ == "__main__":
    train_flow.deploy(
        name="train_daily",
        work_pool_name="default-agent-pool",
        entrypoint="ml_scorecard_prefect.py:customer_ml_flow",
        schedule=CronSchedule(cron="0 1 * * *"),  # chạy lúc 1h sáng mỗi ngày
    )

'''

from prefect import flow
from prefect.client.schemas.schedules import CronSchedule
from ml_scorecard_prefect import customer_ml_flow


@flow
def train_flow():
    print("Training...")
    customer_ml_flow(n_customers=1000, n_tx=8000, iv_threshold=0.01)

if __name__ == "__main__":
    train_flow.deploy(
        name="train_daily",
        work_pool_name="default-agent-pool",
        schedule=CronSchedule(cron="0 1 * * *"),
    )
