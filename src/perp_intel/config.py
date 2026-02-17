from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    fapi_rest: str = os.getenv("BINANCE_FAPI_REST", "https://fapi.binance.com")
    fstream_ws: str = os.getenv("BINANCE_FSTREAM_WS", "wss://fstream.binance.com")

    quote_asset: str = os.getenv("QUOTE_ASSET", "USDT")
    max_symbols: int = int(os.getenv("MAX_SYMBOLS", "30"))

    duckdb_path: str = os.getenv("DUCKDB_PATH", "data/perp_intel.duckdb")
    rest_poll_seconds: int = int(os.getenv("REST_POLL_SECONDS", "30"))

settings = Settings()
