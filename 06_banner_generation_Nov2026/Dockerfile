# --- Stage 1: build dependencies ---
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Stage 2: runtime ---
FROM python:3.11-slim
WORKDIR /app

# Copy dependencies already installed by the builder stage.
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# App code + the domain data it reads at runtime (guidelines, design rules,
# bucket asset library, model_usage.json). `input/` and `output/` are NOT
# baked in -- mount them as volumes so real request files / generated images
# persist outside the container (see docker-compose.yml).
COPY app/ ./app/
COPY lib/ ./lib/
COPY config/ ./config/
COPY prompts/ ./prompts/

RUN mkdir -p /app/input /app/output && useradd -m appuser && chown -R appuser /app
USER appuser

# Settings are read from environment variables passed in at `docker run` /
# by the orchestrator (see .env.example) -- .env itself is NEVER baked into
# the image (see .dockerignore).
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
