"""One-off/inspection script: dump every table in `usage/usage_log.db`
(SQLite) out to CSV files so the usage data can be opened in Excel/Sheets
without needing a SQLite client.

By default it reads `usage/usage_log.db` (two tables as of 2026-09-07:
`runs` and `run_versions`) and writes one CSV per table into the same
`usage/` folder (e.g. `usage/runs.csv`, `usage/run_versions.csv`).

This is a preparation/inspection script (`script/`), not runtime app logic.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "usage" / "usage_log.db"
DEFAULT_OUT_DIR = PROJECT_ROOT / "usage"


def _list_tables(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    return [row[0] for row in cur.fetchall()]


def _export_table(conn: sqlite3.Connection, table: str, out_path: Path) -> int:
    cur = conn.execute(f"SELECT * FROM {table}")
    columns = [d[0] for d in cur.description]
    rows = cur.fetchall()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to usage_log.db (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory to write CSV files into (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--table",
        action="append",
        dest="tables",
        default=None,
        help="Table name to export (repeatable). Default: export all tables found.",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"DB not found: {args.db}")
        return 1

    conn = sqlite3.connect(str(args.db))
    try:
        tables = args.tables or _list_tables(conn)
        if not tables:
            print(f"No tables found in {args.db}")
            return 1

        for table in tables:
            out_path = args.out_dir / f"{table}.csv"
            n_rows = _export_table(conn, table, out_path)
            print(f"{table:<20} {n_rows:>6} row(s) -> {out_path}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------------
# Chạy độc lập file này:
#   cd version_2h && python script/read_usage_log_db.py
#
# Mặc định đọc usage/usage_log.db và xuất mỗi table ra 1 file CSV riêng
# trong usage/ (runs.csv, run_versions.csv, ...).
#
# Tuỳ chọn:
#   --db <path>        đường dẫn tới file .db khác
#   --out-dir <path>   thư mục xuất CSV khác
#   --table <name>     chỉ xuất 1 (hoặc nhiều, lặp lại flag) table cụ thể
# ---------------------------------------------------------------------------
