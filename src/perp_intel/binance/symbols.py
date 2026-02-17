from __future__ import annotations

from typing import List
from .rest import BinanceFapiRest

async def get_usdt_perp_symbols(rest: BinanceFapiRest, quote_asset: str, max_symbols: int) -> List[str]:
    info = await rest.exchange_info()
    syms = []
    for s in info.get("symbols", []):
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("status") != "TRADING":
            continue
        if s.get("quoteAsset") != quote_asset:
            continue
        syms.append(s["symbol"])
    # simple heuristic: keep first N (you can later sort by volume via /ticker/24hr)
    return syms[:max_symbols]
