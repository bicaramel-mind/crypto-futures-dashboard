import asyncio
from perp_intel.config import settings
from perp_intel.pipelines.backfill import run_backfill

if __name__ == "__main__":
    asyncio.run(
        run_backfill(
            base_url=settings.fapi_rest,
            duckdb_path=settings.duckdb_path,
            quote_asset=settings.quote_asset,
            max_symbols=settings.max_symbols,
        )
    )
