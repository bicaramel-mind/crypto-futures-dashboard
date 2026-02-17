from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Dict, List, Any
import websockets

def combined_stream_url(base_ws: str, streams: List[str]) -> str:
    base_ws = base_ws.rstrip("/")
    return f"{base_ws}/stream?streams=" + "/".join(streams)

async def stream_messages(url: str) -> AsyncIterator[Dict[str, Any]]:
    async with websockets.connect(url, ping_interval=60, ping_timeout=20) as ws:
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            # combined stream wraps as {"stream": "...", "data": {...}}
            yield msg

async def run_forever(coro, restart_delay: float = 2.0) -> None:
    while True:
        try:
            await coro()
        except Exception:
            await asyncio.sleep(restart_delay)
