FROM public.ecr.aws/sagemaker/sagemaker-inference:latest

COPY inference.py /opt/program/inference.py

RUN pip install pandas xgboost

ENV SAGEMAKER_PROGRAM inference.py
