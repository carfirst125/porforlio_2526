FROM python:3.11-slim   
#debian full version instead of slim/bullseye

WORKDIR /

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libffi-dev \
       libssl-dev \
       libxml2-dev \
       libxslt-dev \
       git \
    || apt-get install -y --fix-missing \
    && rm -rf /var/lib/apt/lists/*


COPY app/requirements.txt requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt --no-deps

COPY app/ ./app/

ENV PYTHONPATH=/ 

EXPOSE 8000

CMD ["uvicorn", "app.chatbot:app", "--host", "0.0.0.0", "--port", "8000"]
