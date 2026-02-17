from __future__ import annotations

import httpx
from typing import Any, Dict, Optional

class BinanceFapiRest:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=20.0)

    async def close(self) -> None:
        await self._client.aclose()

    async def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        r = await self._client.get(path, params=params)
        r.raise_for_status()
        return r.json()

    # ---- Market data endpoints (USD-M) ----
    async def exchange_info(self) -> Any:
        return await self.get("/fapi/v1/exchangeInfo")

    async def premium_index(self, symbol: Optional[str] = None) -> Any:
        params = {"symbol": symbol} if symbol else None
        return await self.get("/fapi/v1/premiumIndex", params=params)

    async def funding_rate_history(self, symbol: Optional[str] = None, startTime: Optional[int] = None,
                                   endTime: Optional[int] = None, limit: int = 200) -> Any:
        params: Dict[str, Any] = {"limit": limit}
        if symbol: params["symbol"] = symbol
        if startTime: params["startTime"] = startTime
        if endTime: params["endTime"] = endTime
        return await self.get("/fapi/v1/fundingRate", params=params)

    async def funding_info(self) -> Any:
        return await self.get("/fapi/v1/fundingInfo")

    async def open_interest(self, symbol: str) -> Any:
        return await self.get("/fapi/v1/openInterest", params={"symbol": symbol})

    async def open_interest_hist(self, symbol: str, period: str = "1h", limit: int = 200) -> Any:
        return await self.get("/futures/data/openInterestHist", params={
            "symbol": symbol,
            "period": period,
            "limit": limit,
        })

    async def basis_hist(self, pair: str, contractType: str = "PERPETUAL", period: str = "1h", limit: int = 200) -> Any:
        return await self.get("/futures/data/basis", params={
            "pair": pair,
            "contractType": contractType,
            "period": period,
            "limit": limit,
        })

    async def depth(self, symbol: str, limit: int = 100) -> Any:
        return await self.get("/fapi/v1/depth", params={"symbol": symbol, "limit": limit})
