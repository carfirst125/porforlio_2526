from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.config import Settings


@dataclass
class FeatureBuildReport:
    n_customers: int
    reference_date: pd.Timestamp
    feature_columns: list[str]

    def to_dict(self) -> dict:
        return {
            "n_customers": self.n_customers,
            "reference_date": str(self.reference_date.date()),
            "feature_columns": self.feature_columns,
        }


class CustomerFeatureBuilder:
    """Customer-level aggregates from cleaned line items (incl. cancellation flag)."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def build(self, df: pd.DataFrame) -> tuple[pd.DataFrame, FeatureBuildReport]:
        if df.empty:
            raise ValueError("No rows left after cleaning; cannot build customer features.")

        tx = df.copy()
        tx["LineTotal"] = tx["Quantity"].astype(float) * tx["UnitPrice"].astype(float)
        is_cancel = tx["_is_cancellation"].astype(bool)
        positive = ~is_cancel

        ref = tx.loc[positive, "InvoiceDate"].max()
        if pd.isna(ref):
            ref = tx["InvoiceDate"].max()
        if self.settings.reference_date:
            ref = pd.Timestamp(self.settings.reference_date)

        # --- Core RFM-style on valid purchases ---
        pos = tx.loc[positive].copy()

        def recency_days(g: pd.DataFrame) -> float:
            last = g["InvoiceDate"].max()
            return float((ref - last).days)

        agg = (
            pos.groupby("CustomerID")
            .agg(
                Monetary=("LineTotal", "sum"),
                Frequency=("InvoiceNo", "nunique"),
                LastPurchaseDate=("InvoiceDate", "max"),
                FirstPurchaseDate=("InvoiceDate", "min"),
                AvgLineValue=("LineTotal", "mean"),
                TotalQuantity=("Quantity", "sum"),
                UniqueProducts=("StockCode", "nunique"),
                UniqueInvoices=("InvoiceNo", "nunique"),
            )
            .reset_index()
        )

        agg["RecencyDays"] = (ref - agg["LastPurchaseDate"]).dt.days.astype(float)
        agg["TenureDays"] = (agg["LastPurchaseDate"] - agg["FirstPurchaseDate"]).dt.days.clip(lower=1).astype(float)

        # Cancellations / returns (all lines)
        cstat = tx.assign(_c=is_cancel.astype(int)).groupby("CustomerID").agg(
            NLines=("InvoiceNo", "count"),
            CancelLines=("_c", "sum"),
        )
        cstat["CancelRate"] = (cstat["CancelLines"] / cstat["NLines"].replace(0, np.nan)).fillna(0.0)

        # Country mode (from positive spend)
        country_mode = (
            pos.groupby("CustomerID")["Country"]
            .agg(lambda s: s.value_counts().index[0] if len(s) else "")
            .rename("PrimaryCountry")
        )

        feat = agg.merge(cstat[["CancelRate"]], on="CustomerID", how="left")
        feat = feat.merge(country_mode, on="CustomerID", how="left")
        feat["PrimaryCountry"] = feat["PrimaryCountry"].fillna("")
        feat["IsUK"] = (feat["PrimaryCountry"].str.upper() == "UNITED KINGDOM").astype(int)

        # Behavioral rates
        feat["OrdersPerMonth"] = feat["Frequency"] / (feat["TenureDays"] / 30.0).replace(0, np.nan)
        feat["OrdersPerMonth"] = feat["OrdersPerMonth"].replace([np.inf, -np.inf], np.nan).fillna(feat["Frequency"])
        feat["AvgOrderValue"] = feat["Monetary"] / feat["Frequency"].replace(0, np.nan)
        feat["AvgOrderValue"] = feat["AvgOrderValue"].fillna(0.0)

        # Purchase velocity trend: compare first half vs second half spend (adaptive simple split)
        trend = self._spend_trend_slope(tx)
        feat = feat.merge(trend, on="CustomerID", how="left")
        feat["SpendTrendScore"] = feat["SpendTrendScore"].fillna(0.0)

        feature_cols = [
            "RecencyDays",
            "Frequency",
            "Monetary",
            "AvgOrderValue",
            "OrdersPerMonth",
            "UniqueProducts",
            "CancelRate",
            "IsUK",
            "TenureDays",
            "SpendTrendScore",
        ]
        for c in feature_cols:
            if c not in feat.columns:
                feat[c] = 0.0

        report = FeatureBuildReport(
            n_customers=len(feat),
            reference_date=ref,
            feature_columns=feature_cols,
        )
        return feat, report

    def _spend_trend_slope(self, tx: pd.DataFrame) -> pd.DataFrame:
        """Per customer, rough growth: (recent 50% days spend - older 50%) / total."""
        pos = tx.loc[~tx["_is_cancellation"]].copy()
        if pos.empty:
            return pd.DataFrame({"CustomerID": pd.Series(dtype=int), "SpendTrendScore": pd.Series(dtype=float)})

        def customer_trend(g: pd.DataFrame) -> float:
            g = g.sort_values("InvoiceDate")
            mid = g["InvoiceDate"].median()
            early = g.loc[g["InvoiceDate"] < mid, "LineTotal"].sum()
            late = g.loc[g["InvoiceDate"] >= mid, "LineTotal"].sum()
            total = g["LineTotal"].sum()
            if total <= 0:
                return 0.0
            return float((late - early) / (total + 1e-6))

        rows = [
            {"CustomerID": int(cid), "SpendTrendScore": customer_trend(g)}
            for cid, g in pos.groupby("CustomerID")
        ]
        return pd.DataFrame(rows)
