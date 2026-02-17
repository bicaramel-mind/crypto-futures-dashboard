from __future__ import annotations

import asyncio
from typing import List
from ..binance.rest import BinanceFapiRest
from ..storage.duckdb_store import DuckDBStore
from ..binance.symbols import get_usdt_perp_symbols
from ..log import get_logger

log = get_logger("backfill")

async def backfill_all(rest: BinanceFapiRest, store: DuckDBStore, symbols: List[str]) -> None:
    # funding history (recent 200 by default per endpoint behavior)
    for sym in symbols:
        fh = await rest.funding_rate_history(symbol=sym, startTime=1770771600000, limit=1000)
        store.insert_many_funding([
            {
                "funding_ts_ms": int(x["fundingTime"]),
                "symbol": x["symbol"],
                "funding_rate": float(x["fundingRate"]),
                "mark_price": float(x.get("markPrice", "nan")),
            } for x in fh
        ])

        oih = await rest.open_interest_hist(symbol=sym, period="1h", limit=1000)
        store.insert_many_oi([
            {
                "ts_ms": int(x["timestamp"]),
                "symbol": x["symbol"],
                "sum_open_interest": float(x["sumOpenInterest"]),
                "sum_open_interest_value": float(x["sumOpenInterestValue"]),
            } for x in oih
        ])

        bh = await rest.basis_hist(pair=sym, contractType="PERPETUAL", period="1h", limit=1000)
        store.insert_many_basis([
            {
                "ts_ms": int(x["timestamp"]),
                "pair": x["pair"],
                "futures_price": float(x["futuresPrice"]),
                "index_price": float(x["indexPrice"]),
                "basis": float(x["basis"]),
                "basis_rate": float(x["basisRate"]),
            } for x in bh
        ])

        log.info("Backfilled %s", sym)

async def run_backfill(base_url: str, duckdb_path: str, quote_asset: str, max_symbols: int) -> None:
    rest = BinanceFapiRest(base_url)
    store = DuckDBStore(duckdb_path)

    symbols = await get_usdt_perp_symbols(rest, quote_asset=quote_asset, max_symbols=max_symbols)
    log.info("Universe size=%d", len(symbols))

    await backfill_all(rest, store, symbols)
    await rest.close()
