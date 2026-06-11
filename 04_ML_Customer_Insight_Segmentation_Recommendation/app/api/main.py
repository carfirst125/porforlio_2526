from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from app.api.schemas import BatchInsightRequest, BatchInsightResponse, HealthResponse
from app.api.serialize import record_for_api
from app.config import get_settings
from app.storage.insight_store import InsightStore

_store: InsightStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store
    s = get_settings()
    _store = InsightStore(s.insights_path())
    _store.load()
    yield
    _store = None


app = FastAPI(
    title="Retail customer insights",
    version="0.1.0",
    lifespan=lifespan,
    description="Read-only API over precomputed segmentation and recommendations. "
    "Run `python -m app.jobs.refresh_insights` to rebuild the parquet store.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    assert _store is not None
    s = get_settings()
    return HealthResponse(
        status="ok",
        indexed_customers=_store.size,
        insights_path=str(s.insights_path()),
    )


@app.get("/v1/customers/{customer_id}")
def get_customer(customer_id: int) -> dict[str, Any]:
    assert _store is not None
    row = _store.get(customer_id)
    if row is None:
        raise HTTPException(status_code=404, detail="CustomerID not found in insights store")
    return record_for_api(row)


@app.post("/v1/customers/insights", response_model=BatchInsightResponse)
def batch_insights(body: BatchInsightRequest) -> BatchInsightResponse:
    assert _store is not None
    ids = list(dict.fromkeys(body.customer_ids))
    found_raw = _store.get_many(ids)
    found_ids = {int(r["CustomerID"]) for r in found_raw}
    missing = [i for i in ids if i not in found_ids]
    found = [record_for_api(r) for r in found_raw]
    return BatchInsightResponse(found=found, missing_ids=missing)


def create_app() -> FastAPI:
    return app
