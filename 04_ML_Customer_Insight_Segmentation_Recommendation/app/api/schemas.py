from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CustomerInsightResponse(BaseModel):
    """Documented subset of the insight record (full records are returned as dict from endpoints)."""

    model_config = ConfigDict(extra="allow")

    CustomerID: int
    segment_id: int = Field(..., description="Cluster assignment from latest offline training run.")
    recommended_stock_codes: list[str] = Field(default_factory=list)
    churn_risk_score: float | None = None
    clv_estimate: float | None = None
    transactional_promoter_score: float | None = Field(
        None,
        description="0-100 behavioral proxy; not survey NPS.",
    )
    campaign_hints: str | None = None
    RecencyDays: float | None = None
    Frequency: float | None = None
    Monetary: float | None = None


class BatchInsightRequest(BaseModel):
    customer_ids: list[int] = Field(..., min_length=1, max_length=500)


class BatchInsightResponse(BaseModel):
    """Each item in ``found`` is a full insight record (all parquet columns, JSON-serializable)."""

    found: list[dict[str, Any]]
    missing_ids: list[int]


class HealthResponse(BaseModel):
    status: str
    indexed_customers: int
    insights_path: str
