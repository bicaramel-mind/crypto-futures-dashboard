import asyncio
from perp_intel.config import settings
from perp_intel.binance.rest import BinanceFapiRest
from perp_intel.binance.symbols import get_usdt_perp_symbols
from perp_intel.storage.duckdb_store import DuckDBStore
from perp_intel.pipelines.realtime import run_realtime

async def main():
    rest = BinanceFapiRest(settings.fapi_rest)
    symbols = await get_usdt_perp_symbols(rest, settings.quote_asset, settings.max_symbols)
    await rest.close()

    store = DuckDBStore(settings.duckdb_path)
    await run_realtime(settings.fstream_ws, store, symbols)

if __name__ == "__main__":
    asyncio.run(main())
