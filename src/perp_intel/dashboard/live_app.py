"""
live_app.py — Streamlit realtime perp dashboard (NO DB)
UPDATED (Improved strategy):
- Signal timeframe remains 15m (as requested).
- Funding treated as a SLOW regime: funding_z computed on 8H grid and forward-filled to 15m.
- OI used as SOFT confirmation:
    * oi_notional = oi_last * mark_close
    * oi_notional_z used to boost confidence (never gates direction)
- Held position with hysteresis:
    * flat -> enter when |score| >= entry_score
    * long flips only if score <= -flip_score
    * short flips only if score >= +flip_score
    * otherwise holds (no neutral churn)
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Deque, Optional, Tuple

import altair as alt
import httpx
import numpy as np
import pandas as pd
import streamlit as st
import websockets

alt.data_transformers.disable_max_rows()

# -----------------------------
# Config
# -----------------------------
WS_BASE = "wss://fstream.binance.com"
REST_BASE = "https://fapi.binance.com"

DEFAULT_SYMBOL = "BTCUSDT"

MAX_LIVE_TICKS = 400_000
MAX_OI_SNAPS = 80_000

# Signal timeframe fixed to 15 minutes
SIGNAL_TF = "15min"        # pandas offset alias
SIGNAL_TF_BINANCE = "15m"  # Binance interval/period

# Funding regime grid
FUND_TF = "8H"             # pandas offset alias for funding regime z-score

DEFAULT_LIVE_WINDOW_MIN = 60

# Improved strategy defaults
LOOKBACK_BASIS_15M_DEFAULT = 192   # 2 days of 15m bars
LOOKBACK_OI_15M_DEFAULT = 192      # 2 days (effective only when OI exists)
LOOKBACK_FUND_8H_DEFAULT = 21      # ~7 days on 8h grid

ENTRY_SCORE_DEFAULT = 1.4
FLIP_SCORE_DEFAULT = 1.6

# Feature weights (OI is soft confirmation)
W_FUND = 0.9
W_BASIS = 0.7
W_OI = 0.4
W_MOM = 0.5


# -----------------------------
# Data models
# -----------------------------
@dataclass(frozen=True)
class MarkTick:
    ts_ms: int
    symbol: str
    mark_price: float
    index_price: float
    funding_rate: float
    next_funding_ts_ms: int


@dataclass(frozen=True)
class OISnap:
    ts_ms: int
    symbol: str
    open_interest: float


# -----------------------------
# Time helpers
# -----------------------------
def now_ms() -> int:
    return int(time.time() * 1000)


def prune_by_minutes(buf: Deque, minutes: int, ts_attr: str = "ts_ms") -> None:
    cutoff = now_ms() - int(minutes * 60 * 1000)
    while buf and getattr(buf[0], ts_attr) < cutoff:
        buf.popleft()


def _ms_range_last_hours(hours: int) -> Tuple[int, int]:
    end = now_ms()
    start = end - hours * 60 * 60 * 1000
    return start, end


# -----------------------------
# Rolling stats + helpers
# -----------------------------
def rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    if x.empty:
        return x
    minp = max(10, window // 3)
    m = x.rolling(window, min_periods=minp).mean()
    s = x.rolling(window, min_periods=minp).std(ddof=0).replace(0, np.nan)
    return (x - m) / s


def crowding_score_from_funding_z(fz: pd.Series) -> pd.Series:
    zabs = fz.abs().clip(0, 4)
    return (zabs / 4.0) * 100.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_funding_z_8h_to_15m(df_bars: pd.DataFrame, lookback_8h: int) -> pd.Series:
    """
    funding_last: piecewise constant at 15m (we keep updating with WS r)
    funding_z: computed on 8H grid, then ffilled to 15m timestamps
    """
    if df_bars.empty or "funding_last" not in df_bars.columns:
        return pd.Series(index=df_bars.index, data=np.nan)

    tmp = df_bars[["ts", "funding_last"]].dropna().copy()
    if tmp.empty:
        return pd.Series(index=df_bars.index, data=np.nan)

    s8 = tmp.set_index("ts")["funding_last"].sort_index().resample(FUND_TF).last().ffill()
    if len(s8) < 5:
        return pd.Series(index=df_bars.index, data=np.nan)

    w = int(min(lookback_8h, max(10, len(s8))))
    z8 = rolling_zscore(s8, w)

    z15 = z8.resample(SIGNAL_TF).ffill()
    return df_bars["ts"].map(z15).astype(float)


# -----------------------------
# Binance: WebSocket markPrice
# -----------------------------
def ws_url(symbol: str) -> str:
    return f"{WS_BASE}/ws/{symbol.lower()}@markPrice"


def parse_mark_price_update(msg: dict) -> Optional[MarkTick]:
    if msg.get("e") != "markPriceUpdate":
        return None
    try:
        return MarkTick(
            ts_ms=int(msg["E"]),
            symbol=str(msg["s"]),
            mark_price=float(msg["p"]),
            index_price=float(msg["i"]),
            funding_rate=float(msg["r"]),
            next_funding_ts_ms=int(msg["T"]),
        )
    except Exception:
        return None


async def ws_consumer(symbol: str, out_q: Queue, stop_flag: threading.Event) -> None:
    url = ws_url(symbol)
    backoff = 1.0
    while not stop_flag.is_set():
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                backoff = 1.0
                while not stop_flag.is_set():
                    raw = await ws.recv()
                    tick = parse_mark_price_update(json.loads(raw))
                    if tick:
                        out_q.put(tick)
        except Exception:
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 10.0)


# -----------------------------
# Binance: REST polling OI (instant snapshot)
# -----------------------------
def fetch_open_interest(symbol: str, client: httpx.Client) -> Optional[float]:
    try:
        r = client.get(f"{REST_BASE}/fapi/v1/openInterest", params={"symbol": symbol}, timeout=10.0)
        r.raise_for_status()
        return float(r.json()["openInterest"])
    except Exception:
        return None


def oi_poller(symbol: str, out_q: Queue, stop_flag: threading.Event, poll_seconds: float) -> None:
    with httpx.Client() as client:
        while not stop_flag.is_set():
            oi = fetch_open_interest(symbol, client)
            if oi is not None:
                out_q.put(OISnap(ts_ms=now_ms(), symbol=symbol, open_interest=oi))
            end = time.time() + poll_seconds
            while time.time() < end and not stop_flag.is_set():
                time.sleep(0.1)


# -----------------------------
# Seeding (REST) for 15m feature bars
# -----------------------------
def fetch_mark_price_klines(symbol: str, hours: int, interval: str) -> pd.DataFrame:
    start, end = _ms_range_last_hours(hours)
    params = {"symbol": symbol, "interval": interval, "startTime": start, "endTime": end, "limit": 1500}
    r = httpx.get(f"{REST_BASE}/fapi/v1/markPriceKlines", params=params, timeout=15.0)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(
        data,
        columns=["openTime", "open", "high", "low", "close", "volume", "closeTime", "qv", "n", "tbbv", "tbqv", "ignore"],
    )
    df["ts"] = pd.to_datetime(df["openTime"], unit="ms", utc=True)
    df["mark_close"] = df["close"].astype(float)
    return df[["ts", "mark_close"]].sort_values("ts")


def fetch_index_price_klines(symbol: str, hours: int, interval: str) -> pd.DataFrame:
    # IMPORTANT: this endpoint uses pair= (NOT symbol=)
    start, end = _ms_range_last_hours(hours)
    params = {"pair": symbol, "interval": interval, "startTime": start, "endTime": end, "limit": 1500}
    r = httpx.get(f"{REST_BASE}/fapi/v1/indexPriceKlines", params=params, timeout=15.0)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(
        data,
        columns=["openTime", "open", "high", "low", "close", "volume", "closeTime", "qv", "n", "tbbv", "tbqv", "ignore"],
    )
    df["ts"] = pd.to_datetime(df["openTime"], unit="ms", utc=True)
    df["index_close"] = df["close"].astype(float)
    return df[["ts", "index_close"]].sort_values("ts")


def _tf_minutes(period: str) -> int:
    if period.endswith("m"):
        return int(period[:-1])
    if period.endswith("h"):
        return int(period[:-1]) * 60
    if period.endswith("d"):
        return int(period[:-1]) * 24 * 60
    raise ValueError(f"Unsupported period: {period}")


def fetch_open_interest_hist(symbol: str, hours: int, period: str) -> pd.DataFrame:
    start, end = _ms_range_last_hours(hours)
    minutes = _tf_minutes(period)
    expected = int(np.ceil((hours * 60) / minutes)) + 5
    limit = max(30, min(500, expected))  # Binance max limit is 500

    params = {"symbol": symbol, "period": period, "startTime": start, "endTime": end, "limit": limit}
    r = httpx.get(f"{REST_BASE}/futures/data/openInterestHist", params=params, timeout=15.0)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["oi_last"] = df["sumOpenInterest"].astype(float)
    return df[["ts", "oi_last"]].sort_values("ts")


def fetch_funding_rate_history_resampled(symbol: str, hours: int, tf: str) -> pd.DataFrame:
    """
    fundingRate returns events (typically every 8h). We resample to tf and forward-fill.
    """
    start, end = _ms_range_last_hours(hours)
    params = {"symbol": symbol, "startTime": start, "endTime": end, "limit": 1000}
    r = httpx.get(f"{REST_BASE}/fapi/v1/fundingRate", params=params, timeout=15.0)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_event"] = df["fundingRate"].astype(float)

    s = df.set_index("ts")["funding_event"].sort_index()
    s = s.resample(tf).last().ffill()
    return s.reset_index().rename(columns={"funding_event": "funding_last"})


def seed_features(symbol: str, seed_hours: int) -> pd.DataFrame:
    mk = fetch_mark_price_klines(symbol, seed_hours, SIGNAL_TF_BINANCE)
    ix = fetch_index_price_klines(symbol, seed_hours, SIGNAL_TF_BINANCE)
    oi = fetch_open_interest_hist(symbol, seed_hours, SIGNAL_TF_BINANCE)
    fr = fetch_funding_rate_history_resampled(symbol, seed_hours, SIGNAL_TF)

    df = mk.merge(ix, on="ts", how="inner").merge(oi, on="ts", how="left")
    if not fr.empty:
        df = df.merge(fr, on="ts", how="left")
    else:
        df["funding_last"] = np.nan

    df = df.sort_values("ts").reset_index(drop=True)

    # Derived features
    df["funding_last"] = df["funding_last"].ffill()
    df["basis_close"] = (df["mark_close"] - df["index_close"]) / df["index_close"].replace(0, np.nan)

    df["ret_1"] = df["mark_close"].pct_change()
    df["ret_4"] = df["mark_close"].pct_change(4)  # 1h drift

    df["oi_notional"] = df["oi_last"] * df["mark_close"]
    df["oi_notional_chg"] = df["oi_notional"].diff()

    return df


# -----------------------------
# Feature maintenance + indicators + signal (improved)
# -----------------------------
def upsert_current_bar_15m(
    df: pd.DataFrame,
    ts: pd.Timestamp,
    mark: float,
    index: float,
    funding: float,
    oi_snapshot: Optional[float],
) -> pd.DataFrame:
    bar = ts.floor(SIGNAL_TF)

    if df.empty:
        df = pd.DataFrame(
            [
                {
                    "ts": bar,
                    "mark_close": mark,
                    "index_close": index,
                    "funding_last": funding,
                    "oi_last": oi_snapshot,
                }
            ]
        )
    else:
        last_bar = pd.to_datetime(df["ts"].iloc[-1], utc=True)
        if last_bar < bar:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            {
                                "ts": bar,
                                "mark_close": mark,
                                "index_close": index,
                                "funding_last": funding,
                                "oi_last": oi_snapshot,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        else:
            df.loc[df.index[-1], "mark_close"] = mark
            df.loc[df.index[-1], "index_close"] = index
            df.loc[df.index[-1], "funding_last"] = funding
            if oi_snapshot is not None:
                df.loc[df.index[-1], "oi_last"] = oi_snapshot

    # Recompute derived columns (cheap enough for typical bar counts)
    df = df.sort_values("ts").reset_index(drop=True)
    df["funding_last"] = df["funding_last"].ffill()

    df["basis_close"] = (df["mark_close"] - df["index_close"]) / df["index_close"].replace(0, np.nan)
    df["ret_1"] = df["mark_close"].pct_change()
    df["ret_4"] = df["mark_close"].pct_change(4)

    df["oi_notional"] = df["oi_last"] * df["mark_close"]
    df["oi_notional_chg"] = df["oi_notional"].diff()

    return df


def compute_indicators(
    df: pd.DataFrame,
    lookback_basis_15m: int,
    lookback_oi_15m: int,
    lookback_fund_8h: int,
) -> pd.DataFrame:
    d = df.copy().sort_values("ts").reset_index(drop=True)

    # basis z on 15m
    w_b = int(min(lookback_basis_15m, max(20, len(d))))
    d["basis_z"] = rolling_zscore(d["basis_close"], w_b)

    # oi z on 15m (NaN if OI missing)
    w_o = int(min(lookback_oi_15m, max(20, len(d))))
    d["oi_notional_z"] = rolling_zscore(d["oi_notional_chg"], w_o)

    # funding z computed on 8h grid, ffilled to 15m
    d["funding_z"] = compute_funding_z_8h_to_15m(d, lookback_8h=int(lookback_fund_8h))

    # momentum
    d["mom"] = d["ret_4"].fillna(0.0)

    # crowding
    d["crowding_score"] = crowding_score_from_funding_z(d["funding_z"])

    return d


def score_row(r: pd.Series) -> float:
    """
    Positive => LONG, Negative => SHORT
    - funding_z and basis_z are contrarian (crowding): score uses -z
    - oi_notional_z boosts magnitude only (soft confirmation)
    - momentum adds a small trend prior
    """
    fz = float(r.get("funding_z", np.nan))
    bz = float(r.get("basis_z", np.nan))
    oz = float(r.get("oi_notional_z", np.nan))
    mom = float(r.get("mom", 0.0))

    fund_term = 0.0 if not np.isfinite(fz) else (-fz)
    basis_term = 0.0 if not np.isfinite(bz) else (-bz)

    mom_term = clamp(mom * 10.0, -2.0, 2.0)

    base = W_FUND * fund_term + W_BASIS * basis_term + W_MOM * mom_term

    oi_term = 0.0
    if np.isfinite(oz) and abs(base) > 0.3:
        oi_term = abs(clamp(oz, -2.0, 2.0))

    score = base + W_OI * oi_term
    return float(clamp(score, -6.0, 6.0))


def update_held_position(current_pos: int, score: float, entry_score: float, flip_score: float) -> int:
    """
    Hysteresis position state:
      - flat -> enter when score crosses +/- entry_score
      - long flips only if score <= -flip_score
      - short flips only if score >= +flip_score
      - otherwise holds
    """
    if current_pos == 0:
        if score >= entry_score:
            return 1
        if score <= -entry_score:
            return -1
        return 0

    if current_pos == 1:
        if score <= -flip_score:
            return -1
        return 1

    if current_pos == -1:
        if score >= flip_score:
            return 1
        return -1

    return 0


# -----------------------------
# Live buffer → DataFrames
# -----------------------------
def live_to_df(buf: Deque[MarkTick]) -> pd.DataFrame:
    if not buf:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "ts": [pd.to_datetime(t.ts_ms, unit="ms", utc=True) for t in buf],
            "ts_ms": [t.ts_ms for t in buf],
            "mark_price": [t.mark_price for t in buf],
            "index_price": [t.index_price for t in buf],
            "funding_rate": [t.funding_rate for t in buf],
            "next_funding_ts_ms": [t.next_funding_ts_ms for t in buf],
        }
    ).sort_values("ts")
    df["basis"] = (df["mark_price"] - df["index_price"]) / df["index_price"].replace(0, np.nan)
    return df


def oi_to_df(buf: Deque[OISnap]) -> pd.DataFrame:
    if not buf:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "ts": [pd.to_datetime(t.ts_ms, unit="ms", utc=True) for t in buf],
            "ts_ms": [t.ts_ms for t in buf],
            "open_interest": [t.open_interest for t in buf],
        }
    ).sort_values("ts")


# -----------------------------
# Threads + session state
# -----------------------------
def ensure_state() -> None:
    if "sym" not in st.session_state:
        st.session_state["sym"] = None

    if "ws_q" not in st.session_state:
        st.session_state["ws_q"] = Queue()
    if "oi_q" not in st.session_state:
        st.session_state["oi_q"] = Queue()

    if "live_buf" not in st.session_state:
        st.session_state["live_buf"] = deque(maxlen=MAX_LIVE_TICKS)
    if "oi_buf" not in st.session_state:
        st.session_state["oi_buf"] = deque(maxlen=MAX_OI_SNAPS)

    if "ws_thread" not in st.session_state:
        st.session_state["ws_thread"] = None
    if "oi_thread" not in st.session_state:
        st.session_state["oi_thread"] = None

    if "ws_stop" not in st.session_state:
        st.session_state["ws_stop"] = threading.Event()
    if "oi_stop" not in st.session_state:
        st.session_state["oi_stop"] = threading.Event()

    if "feat_df" not in st.session_state:
        st.session_state["feat_df"] = pd.DataFrame()
    if "seed_key" not in st.session_state:
        st.session_state["seed_key"] = None

    # Strategy state
    if "held_pos" not in st.session_state:
        st.session_state["held_pos"] = 0  # -1 short, 0 flat, +1 long


def stop_threads() -> None:
    st.session_state["ws_stop"].set()
    st.session_state["oi_stop"].set()
    st.session_state["ws_thread"] = None
    st.session_state["oi_thread"] = None


def drain_queue(q: Queue, buf: Deque, max_per_run: int = 200_000) -> int:
    n = 0
    while n < max_per_run:
        try:
            item = q.get_nowait()
        except Empty:
            break
        buf.append(item)
        n += 1
    return n


def start_for_symbol(symbol: str, seed_hours: int, oi_poll_seconds: float) -> None:
    ensure_state()
    symbol = symbol.upper().strip()

    # On symbol change, reset everything
    if st.session_state["sym"] and st.session_state["sym"] != symbol:
        stop_threads()
        st.session_state["ws_stop"] = threading.Event()
        st.session_state["oi_stop"] = threading.Event()
        st.session_state["ws_q"] = Queue()
        st.session_state["oi_q"] = Queue()
        st.session_state["live_buf"].clear()
        st.session_state["oi_buf"].clear()
        st.session_state["feat_df"] = pd.DataFrame()
        st.session_state["seed_key"] = None
        st.session_state["held_pos"] = 0

    st.session_state["sym"] = symbol

    # Seed if needed
    seed_key = (symbol, int(seed_hours))
    if st.session_state["seed_key"] != seed_key:
        with st.spinner(f"Seeding {seed_hours}h of {SIGNAL_TF_BINANCE} feature bars…"):
            st.session_state["feat_df"] = seed_features(symbol, int(seed_hours))
        st.session_state["seed_key"] = seed_key
        st.session_state["held_pos"] = 0

    # Start WS thread
    if st.session_state["ws_thread"] is None:
        ws_q: Queue = st.session_state["ws_q"]
        ws_stop: threading.Event = st.session_state["ws_stop"]

        def ws_runner():
            asyncio.run(ws_consumer(symbol, ws_q, ws_stop))

        t = threading.Thread(target=ws_runner, daemon=True)
        t.start()
        st.session_state["ws_thread"] = t

    # Start OI poller
    if st.session_state["oi_thread"] is None:
        oi_q: Queue = st.session_state["oi_q"]
        oi_stop: threading.Event = st.session_state["oi_stop"]
        t2 = threading.Thread(target=oi_poller, args=(symbol, oi_q, oi_stop, oi_poll_seconds), daemon=True)
        t2.start()
        st.session_state["oi_thread"] = t2


# -----------------------------
# Altair charts
# -----------------------------
def line_chart(df: pd.DataFrame, x: str, y: str, title: str, height: int = 300) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(f"{x}:T", title="Time", scale=alt.Scale(zero=False)),
            y=alt.Y(f"{y}:Q", title=title, scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip(f"{x}:T"), alt.Tooltip(f"{y}:Q")],
        )
        .properties(height=height)
        .interactive()
    )


def two_series(df: pd.DataFrame, x: str, y1: str, y2: str, title: str, height: int = 320) -> alt.Chart:
    tmp = df[[x, y1, y2]].copy().melt(id_vars=[x], value_vars=[y1, y2], var_name="series", value_name="value")
    return (
        alt.Chart(tmp)
        .mark_line()
        .encode(
            x=alt.X(f"{x}:T", title="Time", scale=alt.Scale(zero=False)),
            y=alt.Y("value:Q", title=title, scale=alt.Scale(zero=False)),
            color="series:N",
            tooltip=[alt.Tooltip(f"{x}:T"), "series:N", alt.Tooltip("value:Q")],
        )
        .properties(height=height)
        .interactive()
    )


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Perp Intel Live", layout="wide")
st.title("Perp Intel — Live (WS + REST) + 15m Signal (improved strategy)")

ensure_state()

with st.sidebar:
    st.header("Controls")

    symbol = st.text_input("Symbol (USD-M)", DEFAULT_SYMBOL).upper().strip()
    seed_hours = st.selectbox("Seed history (hours)", [24, 48, 72, 168], index=0)

    st.subheader("Signal parameters (15m)")
    lookback_basis_15m = st.selectbox("Basis z lookback (15m bars)", [96, 192, 288, 672], index=1)
    lookback_oi_15m = st.selectbox("OI notional z lookback (15m bars)", [96, 192, 288, 672], index=1)
    lookback_fund_8h = st.selectbox("Funding z lookback (8h bars)", [15, 21, 30, 45], index=1)

    entry_score = st.slider("Entry score", 0.8, 3.0, float(ENTRY_SCORE_DEFAULT), 0.1)
    flip_score = st.slider("Flip score", 1.0, 4.0, float(FLIP_SCORE_DEFAULT), 0.1)

    st.subheader("Display / refresh")
    live_window_min = st.selectbox("Live chart window (minutes)", [15, 30, 60, 120], index=2)
    ui_refresh_ms = st.slider("UI refresh (ms)", 300, 2500, 1000, step=100)
    oi_poll_seconds = st.slider("OI poll (sec)", 5, 60, 15, step=5)
    auto_refresh = st.toggle("Auto-refresh UI", value=True)

    st.caption("Funding z computed on 8H grid; OI is soft confirmation; signal holds with hysteresis.")

if not symbol:
    st.error("Enter a symbol like BTCUSDT")
    st.stop()

start_for_symbol(symbol, int(seed_hours), float(oi_poll_seconds))

# Drain queues into buffers
ws_drained = drain_queue(st.session_state["ws_q"], st.session_state["live_buf"])
oi_drained = drain_queue(st.session_state["oi_q"], st.session_state["oi_buf"])

# Prune live buffers to desired window
prune_by_minutes(st.session_state["live_buf"], int(live_window_min))
prune_by_minutes(st.session_state["oi_buf"], int(live_window_min))

df_live = live_to_df(st.session_state["live_buf"])
df_oi_live = oi_to_df(st.session_state["oi_buf"])

# Latest live values
latest_ts = None
latest_mark = latest_index = latest_funding = None
next_funding_ts = None

if not df_live.empty:
    last = df_live.iloc[-1]
    latest_ts = pd.to_datetime(int(last["ts_ms"]), unit="ms", utc=True)
    latest_mark = float(last["mark_price"])
    latest_index = float(last["index_price"])
    latest_funding = float(last["funding_rate"])
    next_funding_ts = pd.to_datetime(int(last["next_funding_ts_ms"]), unit="ms", utc=True)

latest_oi = float(df_oi_live.iloc[-1]["open_interest"]) if not df_oi_live.empty else None

# Update 15m feature bars with live values
feat = st.session_state["feat_df"].copy()
if latest_ts is not None and latest_mark is not None and latest_index is not None and latest_funding is not None:
    feat = upsert_current_bar_15m(feat, latest_ts, latest_mark, latest_index, latest_funding, latest_oi)

feat_ind = compute_indicators(
    feat,
    lookback_basis_15m=int(lookback_basis_15m),
    lookback_oi_15m=int(lookback_oi_15m),
    lookback_fund_8h=int(lookback_fund_8h),
)
st.session_state["feat_df"] = feat  # persist raw bars

# -----------------------------
# KPIs
# -----------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Symbol", symbol)
k2.metric("Mark", "…" if latest_mark is None else f"{latest_mark:.4f}")
k3.metric("Funding (live)", "…" if latest_funding is None else f"{latest_funding:.6f}")
if latest_mark is not None and latest_index is not None and latest_index != 0:
    k4.metric("Basis (live proxy)", f"{(latest_mark - latest_index)/latest_index:.6f}")
else:
    k4.metric("Basis (live proxy)", "…")
k5.metric("Next funding (UTC)", "…" if next_funding_ts is None else f"{next_funding_ts:%Y-%m-%d %H:%M}")

st.caption(
    f"Drained this run: WS={ws_drained}, OI={oi_drained} | "
    f"Live window={live_window_min}m: ticks={len(df_live)}, OI_snaps={len(df_oi_live)} | "
    f"Feature bars (15m): {len(feat_ind)}"
)

# -----------------------------
# Signal panel
# -----------------------------
st.subheader("15m Indicator (soft OI, 8h funding regime, hysteresis hold)")
if len(feat_ind) >= 40:
    latest_bar = feat_ind.iloc[-1]
    score = score_row(latest_bar)

    st.session_state["held_pos"] = update_held_position(
        st.session_state["held_pos"],
        float(score),
        entry_score=float(entry_score),
        flip_score=float(flip_score),
    )

    pos = int(st.session_state["held_pos"])
    signal = "LONG" if pos == 1 else "SHORT" if pos == -1 else "FLAT"

    s1, s2, s3, s4, s5 = st.columns([1.4, 0.8, 0.9, 0.9, 0.9])
    s1.markdown(f"**Signal:** {signal}")
    s2.metric("Score", f"{score:.2f}")

    fz = float(latest_bar.get("funding_z", np.nan))
    bz = float(latest_bar.get("basis_z", np.nan))
    oz = float(latest_bar.get("oi_notional_z", np.nan))

    s3.metric("Funding z (8h→15m)", "…" if not np.isfinite(fz) else f"{fz:.2f}")
    s4.metric("Basis z (15m)", "…" if not np.isfinite(bz) else f"{bz:.2f}")
    s5.metric("OI notional z (15m)", "…" if not np.isfinite(oz) else f"{oz:.2f}")

    st.caption(
        f"bar_ts={pd.to_datetime(latest_bar['ts'], utc=True)} | "
        f"ret_1(15m)={latest_bar.get('ret_1', np.nan):.3%} | ret_4(1h)={latest_bar.get('ret_4', np.nan):.3%} | "
        f"crowding={latest_bar.get('crowding_score', np.nan):.0f}/100 | held_pos={pos}"
    )

    with st.expander("Debug: last 24 feature bars"):
        cols = [
            "ts",
            "mark_close",
            "index_close",
            "basis_close",
            "basis_z",
            "funding_last",
            "funding_z",
            "oi_last",
            "oi_notional",
            "oi_notional_z",
            "ret_1",
            "ret_4",
            "mom",
            "crowding_score",
        ]
        show = feat_ind[cols].tail(24)
        show = show.assign(score=feat_ind.tail(24).apply(score_row, axis=1).values)
        st.dataframe(show, use_container_width=True)
else:
    st.info("Waiting for enough feature bars… (increase seed hours if needed).")

st.divider()

# -----------------------------
# Charts
# -----------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader(f"Live ({live_window_min}m): Mark vs Index")
    if not df_live.empty:
        st.altair_chart(two_series(df_live, "ts", "mark_price", "index_price", "Price"), use_container_width=True)
    else:
        st.info("Waiting for WS ticks…")

with c2:
    st.subheader(f"Live ({live_window_min}m): Funding (instantaneous)")
    if not df_live.empty:
        st.altair_chart(line_chart(df_live, "ts", "funding_rate", "Funding"), use_container_width=True)
    else:
        st.info("Waiting for WS ticks…")

c3, c4 = st.columns(2)
with c3:
    st.subheader("15m Feature Bars: Funding z (8h→15m) + Crowding")
    if not feat_ind.empty:
        tmp = feat_ind[["ts", "funding_z", "crowding_score"]].dropna()
        if not tmp.empty:
            st.altair_chart(two_series(tmp, "ts", "funding_z", "crowding_score", "Funding z & Crowding"), use_container_width=True)
        else:
            st.info("Funding z not ready (need more funding history).")
    else:
        st.info("No feature bars yet.")

with c4:
    st.subheader("15m Feature Bars: Basis z + OI notional z")
    if not feat_ind.empty:
        tmp = feat_ind[["ts", "basis_z", "oi_notional_z"]]
        st.altair_chart(two_series(tmp, "ts", "basis_z", "oi_notional_z", "Basis z & OI notional z"), use_container_width=True)
    else:
        st.info("No feature bars yet.")

st.subheader("15m Feature Bars: Mark Close + OI Notional")
c5, c6 = st.columns(2)
with c5:
    if not feat_ind.empty:
        st.altair_chart(line_chart(feat_ind, "ts", "mark_close", "15m Mark Close"), use_container_width=True)
with c6:
    if "oi_notional" in feat_ind.columns and feat_ind["oi_notional"].notna().any():
        st.altair_chart(line_chart(feat_ind.dropna(subset=["oi_notional"]), "ts", "oi_notional", "15m OI Notional"), use_container_width=True)
    else:
        st.info("OI notional missing (OI hist empty for symbol / seed window too long).")

# Manual refresh
st.button("Refresh now")

# Auto refresh
if auto_refresh:
    time.sleep(ui_refresh_ms / 1000.0)
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()
