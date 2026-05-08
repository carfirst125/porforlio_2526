from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DatasetProfile:
    """Lightweight statistics used to choose cleaning and transforms."""

    n_rows: int = 0
    n_cols: int = 0
    customer_id_null_ratio: float = 0.0
    duplicate_ratio: float = 0.0
    cancellation_ratio: float = 0.0
    zero_or_negative_price_ratio: float = 0.0
    invoice_date_parse_fail_ratio: float = 0.0
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> DatasetProfile:
        n = len(df)
        notes: list[str] = []
        if n == 0:
            return cls(n_rows=0, n_cols=len(df.columns), notes=["empty dataframe"])

        cust_null = float(df["CustomerID"].isna().mean()) if "CustomerID" in df else 1.0
        dup_ratio = float(df.duplicated().sum()) / n if n else 0.0

        inv = df["InvoiceNo"].astype(str) if "InvoiceNo" in df else pd.Series([""] * n)
        cancel_ratio = float(inv.str.strip().str.upper().str.startswith("C").mean())

        bad_price = 0.0
        if "UnitPrice" in df:
            bad_price = float((df["UnitPrice"] <= 0).mean())

        bad_dates = 0.0
        if "InvoiceDate" in df:
            bad_dates = float(df["InvoiceDate"].isna().mean())

        if cust_null > 0.3:
            notes.append("high_missing_customer_id")
        if dup_ratio > 0.01:
            notes.append("meaningful_duplicate_rows")
        if cancel_ratio > 0.05:
            notes.append("substantial_cancellations")

        return cls(
            n_rows=n,
            n_cols=len(df.columns),
            customer_id_null_ratio=cust_null,
            duplicate_ratio=dup_ratio,
            cancellation_ratio=cancel_ratio,
            zero_or_negative_price_ratio=bad_price,
            invoice_date_parse_fail_ratio=bad_dates,
            notes=notes,
        )

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "customer_id_null_ratio": round(self.customer_id_null_ratio, 6),
            "duplicate_ratio": round(self.duplicate_ratio, 6),
            "cancellation_ratio": round(self.cancellation_ratio, 6),
            "zero_or_negative_price_ratio": round(self.zero_or_negative_price_ratio, 6),
            "invoice_date_parse_fail_ratio": round(self.invoice_date_parse_fail_ratio, 6),
            "notes": list(self.notes),
        }


def column_skew(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 3:
        return 0.0
    return float(s.skew())
