from __future__ import annotations

import pandas as pd

from app.config import Settings


class SegmentRecommender:
    """
    For each segment, rank top products by revenue on non-cancelled lines;
    recommend items each customer has not purchased yet.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def build_recommendations(
        self,
        transactions: pd.DataFrame,
        customer_segments: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        transactions : cleaned dataframe with _is_cancellation, LineTotal can be missing — recompute
        customer_segments : columns CustomerID, segment_id
        """
        tx = transactions.loc[~transactions["_is_cancellation"]].copy()
        tx["LineTotal"] = tx["Quantity"].astype(float) * tx["UnitPrice"].astype(float)

        seg_map = customer_segments.set_index("CustomerID")["segment_id"]
        tx = tx.merge(seg_map.rename("segment_id"), left_on="CustomerID", right_index=True, how="inner")

        top_n = self.settings.top_products_per_segment
        segment_top: dict[int, list[str]] = {}
        for sid, g in tx.groupby("segment_id"):
            prod_rev = g.groupby("StockCode")["LineTotal"].sum().sort_values(ascending=False)
            segment_top[int(sid)] = prod_rev.head(top_n).index.astype(str).tolist()

        cust_products = (
            tx.groupby("CustomerID")["StockCode"].apply(lambda s: set(s.astype(str))).to_dict()
        )

        rec_per = self.settings.recommendations_per_customer
        rows: list[dict] = []
        for cid, seg in seg_map.items():
            tops = segment_top.get(int(seg), [])
            owned = cust_products.get(cid, set())
            picks = [p for p in tops if p not in owned][:rec_per]
            rows.append({"CustomerID": int(cid), "recommended_stock_codes": picks})

        return pd.DataFrame(rows)
