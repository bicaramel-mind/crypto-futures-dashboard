from __future__ import annotations

import asyncio
from typing import List
from ..binance.ws import combined_stream_url, stream_messages
from ..storage.duckdb_store import DuckDBStore
from ..log import get_logger

log = get_logger("realtime")

def _parse_mark_price(data: dict) -> dict:
    return {
        "ts_ms": int(data["E"]),
        "symbol": data["s"],
        "mark_price": float(data["p"]),
        "index_price": float(data["i"]),
        "funding_rate": float(data["r"]),
        "next_funding_ts_ms": int(data["T"]),
    }

def _parse_force_order(data: dict) -> dict:
    o = data["o"]
    return {
        "ts_ms": int(data["E"]),
        "symbol": o["s"],
        "side": o["S"],
        "price": float(o["p"]),
        "qty": float(o["q"]),
    }

async def run_realtime(ws_base: str, store: DuckDBStore, symbols: List[str]) -> None:
    # Keep it simple: markPrice + forceOrder for each symbol
    streams = []
    for s in symbols:
        streams.append(f"{s.lower()}@markPrice")
        streams.append(f"{s.lower()}@forceOrder")

    url = combined_stream_url(ws_base, streams)
    log.info("WS connected: %s (streams=%d)", ws_base, len(streams))

    async for msg in stream_messages(url):
        data = msg.get("data", {})
        et = data.get("e")
        if et == "markPriceUpdate":
            store.insert_mark_price(_parse_mark_price(data))
        elif et == "forceOrder":
            store.insert_force_order(_parse_force_order(data))
