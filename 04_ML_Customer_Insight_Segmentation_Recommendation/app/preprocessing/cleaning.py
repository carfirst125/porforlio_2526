from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.config import Settings
from app.preprocessing.profile import DatasetProfile


@dataclass
class CleaningReport:
    rows_in: int
    rows_out: int
    steps: list[str]

    def to_dict(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "steps": self.steps,
        }


class TransactionCleaner:
    """Apply rules guided by dataset profile rather than a single fixed recipe."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def _stockcode_digit_count(self, codes: pd.Series) -> pd.Series:
        return codes.astype(str).str.findall(r"\d").str.len()

    def clean(self, df: pd.DataFrame, profile: DatasetProfile) -> tuple[pd.DataFrame, CleaningReport]:
        steps: list[str] = []
        rows_in = len(df)
        out = df.copy()

        # Always require identifiable customers for downstream customer-level features
        before = len(out)
        out = out.dropna(subset=["CustomerID"])
        out["CustomerID"] = out["CustomerID"].astype(int)
        steps.append(f"drop_missing_customer_id: {before} -> {len(out)}")

        before = len(out)
        out = out.dropna(subset=["InvoiceDate"])
        steps.append(f"drop_bad_invoice_date: {before} -> {len(out)}")

        if self.settings.duplicate_row_action == "drop" and profile.duplicate_ratio > 0:
            before = len(out)
            out = out.drop_duplicates()
            steps.append(f"drop_duplicate_rows: {before} -> {len(out)}")

        before = len(out)
        out = out.dropna(subset=["Description"])
        steps.append(f"drop_missing_description: {before} -> {len(out)}")

        # Invoice lines: positive unit price; quantity may be negative for returns — keep for net revenue
        before = len(out)
        out = out[out["UnitPrice"] > 0]
        steps.append(f"drop_non_positive_unit_price: {before} -> {len(out)}")

        # Adaptive: remove stock codes with too few digits (postage, bank charges, etc.)
        digit_counts = self._stockcode_digit_count(out["StockCode"])
        before = len(out)
        mask_valid = digit_counts >= self.settings.anomaly_stock_min_digits
        out = out.loc[mask_valid]
        steps.append(
            f"drop_anomaly_stockcode(<{self.settings.anomaly_stock_min_digits} digits): "
            f"{before} -> {len(out)}"
        )

        # Mark cancellations; optionally exclude from revenue-positive clustering space
        inv = out["InvoiceNo"].astype(str).str.strip().str.upper()
        is_cancel = inv.str.startswith("C")
        out = out.assign(_is_cancellation=is_cancel)

        # If cancellations dominate, still keep flag; exclude cancelled lines from spend features in engineering
        before = len(out)
        # Remove rows with absurd quantities that are likely data errors (optional adaptive cap)
        q = out["Quantity"].abs()
        hi = q.quantile(0.9995)
        if np.isfinite(hi) and hi > 1:
            cap = float(max(hi, 10000))
            out = out[q <= cap]
            steps.append(f"trim_extreme_quantity_99.95pct_cap_{cap:.0f}: {before} -> {len(out)}")

        report = CleaningReport(rows_in=rows_in, rows_out=len(out), steps=steps)
        return out, report
