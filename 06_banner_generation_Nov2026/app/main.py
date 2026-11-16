"""FastAPI entrypoint (REQUEST.md [6])."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.logging_conf import get_logger
from app.settings import get_settings

logger = get_logger("main")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("=== %s starting (environment=%s) ===", settings.app_name, settings.environment)
    logger.info("input_dir=%s", settings.input_dir)
    logger.info("output_dir=%s", settings.output_dir)
    logger.info("asset_library_dir=%s", settings.asset_library_dir)
    logger.info("guidelines_file=%s (exists=%s)", settings.guidelines_file, settings.guidelines_file.exists())
    logger.info("model_usage_file=%s (exists=%s)", settings.model_usage_file, settings.model_usage_file.exists())
    for name, path in [
        ("input_dir", settings.input_dir),
        ("output_dir", settings.output_dir),
    ]:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.exception("Could not create %s (%s)", name, path)
    yield
    logger.info("=== %s shutting down ===", settings.app_name)


app = FastAPI(title=get_settings().app_name, version="1.0.0", lifespan=_lifespan)
app.include_router(router)


# ---------------------------------------------------------------------------
# Chạy độc lập file này:
#   cd version_1 && uvicorn app.main:app --reload --reload-dir app --host 0.0.0.0 --port 8000
#
# Yêu cầu trước khi chạy:
#   - `.env` đã có (xem app/settings.py + .env.example)
#   - `pip install -r requirements.txt`
#   - Chạy đúng từ thư mục `version_1/` để import `app.*` resolve đúng
#
# QUAN TRỌNG: luôn thêm `--reload-dir app` -- nếu không, uvicorn theo dõi
# TOÀN BỘ `version_1/` (kể cả input/, output/, lib/) để tự restart, trong
# khi chính app này liên tục ghi vào những thư mục đó lúc đang chạy (prompt
# log, ảnh cache, và cả file log app.log ở output/_logs/). Thiếu
# `--reload-dir app` gây ra vòng lặp tự-restart, biểu hiện ra ngoài là
# frontend nhận ConnectionResetError/"connection aborted" giữa chừng khi
# gọi API -- xem 2-project-technote.md, mục "QUAN TRỌNG -- --reload-dir app".
# ---------------------------------------------------------------------------
