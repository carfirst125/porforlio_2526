from __future__ import annotations

import numpy as np
import pandas as pd


class ExtendedInsightCalculator:
    """
    Derive marketing-oriented scores from transactional features.

    Note: True NPS requires survey responses. Here we expose ``transactional_promoter_score``,
    a 0-100 proxy combining repeat behavior, spend trend, and low cancellation friction.
    """

    def __init__(
        self,
        churn_recency_weight: float = 0.55,
        churn_frequency_weight: float = 0.35,
        churn_cancel_weight: float = 0.10,
        clv_horizon_months: float = 12.0,
    ):
        self.churn_recency_weight = churn_recency_weight
        self.churn_frequency_weight = churn_frequency_weight
        self.churn_cancel_weight = churn_cancel_weight
        self.clv_horizon_months = clv_horizon_months

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = self._add_rfm_scores(out)
        out = self._add_churn_score(out)
        out = self._add_clv_estimate(out)
        out = self._add_transactional_promoter_score(out)
        out = self._add_engagement_and_campaign_hints(out)
        return out

    def _add_rfm_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Quintile-style 1-5 scores for R/F/M (higher is better for F and M, lower recency days is better)."""
        r = df["RecencyDays"]
        f = df["Frequency"]
        m = df["Monetary"]

        def qcut_safe(s: pd.Series, *, ascending_good: bool) -> pd.Series:
            if s.nunique() < 2:
                return pd.Series(3, index=s.index, dtype=float)
            try:
                ranks = pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
            except ValueError:
                ranks = pd.cut(s, bins=5, labels=[1, 2, 3, 4, 5])
            ranks = pd.to_numeric(ranks, errors="coerce").fillna(3)
            if not ascending_good:
                ranks = 6 - ranks  # invert so low recency -> high score
            return ranks

        out = df.copy()
        out["R_score"] = qcut_safe(r, ascending_good=False)
        out["F_score"] = qcut_safe(f, ascending_good=True)
        out["M_score"] = qcut_safe(m, ascending_good=True)
        out["rfm_composite_score"] = (
            (out["R_score"] + out["F_score"] + out["M_score"]) / 15.0 * 100.0
        ).round(2)
        return out

    def _add_churn_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """0-100 probability-style risk: higher = more likely churned / disengaged."""
        r = df["RecencyDays"].astype(float)
        f = df["Frequency"].astype(float)
        c = df["CancelRate"].astype(float)

        r_norm = self._minmax(r)
        f_norm = 1.0 - self._minmax(f)
        c_norm = self._minmax(c)

        raw = (
            self.churn_recency_weight * r_norm
            + self.churn_frequency_weight * f_norm
            + self.churn_cancel_weight * c_norm
        )
        out = df.copy()
        out["churn_risk_score"] = (raw * 100).clip(0, 100).round(2)
        return out

    def _add_clv_estimate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simple historic + forward heuristic CLV (currency units of Monetary):
        avg_order_value * orders_per_month * horizon * engagement_factor.
        """
        aov = df["AvgOrderValue"].astype(float).clip(lower=0)
        opm = df["OrdersPerMonth"].astype(float).clip(lower=0)
        engage = (df["F_score"] + df["M_score"]) / 10.0  # ~0.2 .. 1.0
        base = aov * opm * self.clv_horizon_months * engage.replace(0, 0.5)
        out = df.copy()
        out["clv_estimate"] = base.round(2)
        return out

    def _add_transactional_promoter_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """0-100 proxy for advocacy / referral likelihood from behavior (not survey NPS)."""
        repeat = self._minmax(df["Frequency"].astype(float))
        trend = self._minmax(df["SpendTrendScore"].astype(float) + 0.5)
        low_cancel = 1.0 - self._minmax(df["CancelRate"].astype(float))
        monetary = self._minmax(np.log1p(df["Monetary"].astype(float)))

        raw = 0.35 * repeat + 0.25 * trend + 0.20 * low_cancel + 0.20 * monetary
        out = df.copy()
        out["transactional_promoter_score"] = (raw * 100).clip(0, 100).round(2)
        return out

    def _add_engagement_and_campaign_hints(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        def tag_row(row: pd.Series) -> str:
            tags: list[str] = []
            m_med = float(row["_m_median"])
            m_q75 = float(row["_m_q75"])
            p_q75 = float(row["_p_q75"])
            aov_q25 = float(row["_aov_q25"])
            if row["churn_risk_score"] >= 70:
                tags.append("win_back")
            elif row["RecencyDays"] > 90 and row["Monetary"] > m_med:
                tags.append("reengage_high_value")
            if row["transactional_promoter_score"] >= 70 and row["Monetary"] >= m_q75:
                tags.append("vip_advocate_candidate")
            if row["UniqueProducts"] >= p_q75 and row["Frequency"] >= 3:
                tags.append("cross_sell_ready")
            if row["AvgOrderValue"] < aov_q25 and row["Frequency"] >= 4:
                tags.append("basket_build_up")
            if row["CancelRate"] >= 0.15:
                tags.append("returns_watchlist")
            if not tags:
                tags.append("core_grow")
            return "|".join(sorted(set(tags)))

        med_m = out["Monetary"].median()
        q75_m = out["Monetary"].quantile(0.75)
        q75_p = out["UniqueProducts"].quantile(0.75)
        q25_aov = out["AvgOrderValue"].quantile(0.25)
        out["_m_median"] = med_m
        out["_m_q75"] = q75_m
        out["_p_q75"] = q75_p
        out["_aov_q25"] = q25_aov
        out["campaign_hints"] = out.apply(tag_row, axis=1)
        out.drop(columns=["_m_median", "_m_q75", "_p_q75", "_aov_q25"], inplace=True)
        return out

    @staticmethod
    def _minmax(s: pd.Series | np.ndarray) -> np.ndarray:
        x = np.asarray(s, dtype=float)
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            return np.zeros_like(x, dtype=float)
        return (x - lo) / (hi - lo)
