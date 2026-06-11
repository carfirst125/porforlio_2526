from __future__ import annotations

import pandas as pd

from app.config import Settings


def load_transactions(settings: Settings) -> pd.DataFrame:
    path = settings.resolved_transaction_csv()
    if not path.is_file():
        raise FileNotFoundError(f"Transaction CSV not found: {path}")
    df = pd.read_csv(path, encoding=settings.encoding, low_memory=False)
    expected = {
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df
