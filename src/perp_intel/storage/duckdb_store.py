from __future__ import annotations

import os
import duckdb
from typing import Any, Dict, Iterable
from .schema import SCHEMA_SQL

class DuckDBStore:
    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        self.con = duckdb.connect(path)
        self.con.execute(SCHEMA_SQL)

    def insert_mark_price(self, row: Dict[str, Any]) -> None:
        self.con.execute(
            "INSERT INTO mark_price VALUES (?, ?, ?, ?, ?, ?)",
            [row["ts_ms"], row["symbol"], row["mark_price"], row["index_price"],
             row["funding_rate"], row["next_funding_ts_ms"]],
        )

    def insert_force_order(self, row: Dict[str, Any]) -> None:
        self.con.execute(
            "INSERT INTO force_orders VALUES (?, ?, ?, ?, ?)",
            [row["ts_ms"], row["symbol"], row["side"], row["price"], row["qty"]],
        )

    def insert_many_funding(self, rows: Iterable[Dict[str, Any]]) -> None:
        data = [[r["funding_ts_ms"], r["symbol"], r["funding_rate"], r["mark_price"]] for r in rows]
        if data:
            self.con.executemany("INSERT INTO funding_history VALUES (?, ?, ?, ?)", data)

    def insert_many_oi(self, rows: Iterable[Dict[str, Any]]) -> None:
        data = [[r["ts_ms"], r["symbol"], r["sum_open_interest"], r["sum_open_interest_value"]] for r in rows]
        if data:
            self.con.executemany("INSERT INTO open_interest_hist VALUES (?, ?, ?, ?)", data)

    def insert_many_basis(self, rows: Iterable[Dict[str, Any]]) -> None:
        data = [[r["ts_ms"], r["pair"], r["futures_price"], r["index_price"], r["basis"], r["basis_rate"]] for r in rows]
        if data:
            self.con.executemany("INSERT INTO basis_hist VALUES (?, ?, ?, ?, ?, ?)", data)
