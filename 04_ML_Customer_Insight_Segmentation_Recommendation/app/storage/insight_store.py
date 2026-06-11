from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

import pandas as pd


class InsightStore:
    """
    In-memory index for low-latency API reads. Load once at process start;
    optional reload when batch job finishes (restart process or call reload).
    """

    def __init__(self, parquet_path: Path):
        self.parquet_path = Path(parquet_path)
        self._lock = RLock()
        self._by_id: dict[int, dict[str, Any]] = {}

    def load(self) -> None:
        if not self.parquet_path.is_file():
            self._by_id = {}
            return
        df = pd.read_parquet(self.parquet_path)
        if "CustomerID" not in df.columns:
            raise ValueError("insights parquet must contain CustomerID")
        df["CustomerID"] = df["CustomerID"].astype(int)
        with self._lock:
            self._by_id = {
                int(row["CustomerID"]): row
                for row in df.to_dict(orient="records")
            }

    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        d = df.copy()
        d["CustomerID"] = d["CustomerID"].astype(int)
        with self._lock:
            self._by_id = {}
            records = d.to_dict(orient="records")
            for row in records:
                cid = int(row["CustomerID"])
                self._by_id[cid] = row

    def reload(self) -> None:
        self.load()

    def get(self, customer_id: int) -> dict[str, Any] | None:
        with self._lock:
            return self._by_id.get(int(customer_id))

    def get_many(self, customer_ids: list[int]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        with self._lock:
            for cid in customer_ids:
                row = self._by_id.get(int(cid))
                if row is not None:
                    out.append(row)
        return out

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._by_id)
