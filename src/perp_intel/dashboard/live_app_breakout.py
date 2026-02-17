from __future__ import annotations

import time
from typing import Dict, Tuple

import altair as alt
import httpx
import numpy as np
import pandas as pd
import streamlit as st

alt.data_transformers.disable_max_rows()

REST = "https://fapi.binance.com"
SYMBOL = "ETHUSDT"

# ---------- Strategy params ----------
# Sweep definition
SWEEP_LOOKBACK = 72          # 6h on 5m
RECLAIM_WINDOW = 6           # must reclaim within 30 minutes
WICK_MIN = 0.45              # wick fraction threshold (0..1)
VOL_SPIKE_Q = 0.80           # volume spike quantile

# Liquidation proxy (OI)
OI_PERIOD_PRIMARY = "5m"     # will fallback to 15m if rejected
OI_Z_WIN = 96                # 8h
OI_DROP_Z = -1.25            # liquidation-ish OI drop

# Perps regime filters
FUNDING_BAD_LONG = -0.0008   # skip catching knife if funding too negative
FUNDING_BAD_SHORT = 0.0008   # skip if funding too positive

# Trend-day filter (skip mean reversion when trend is strong)
EMA_FAST = 48
EMA_SLOW = 240
TREND_STRENGTH_MAX = 0.004   # |ema_fast-ema_slow|/price must be <= this
MOMENTUM_1H_MAX = 0.008      # |1h return| must be <= this

# Suggested stop/TP for display
ATR_N = 14
STOP_ATR_MULT = 0.6
TP_TO_VWAP = True

# ---------- Dashboard params ----------
DEFAULT_REFRESH_SEC = 10
KLINES_LIMIT = 1500


def now_ms() -> int:
    return int(time.time() * 1000)


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(5, n // 2)).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (tp * df["volume"]).cumsum()
    vv = df["volume"].cumsum().replace(0, np.nan)
    return pv / vv


def zscore(s: pd.Series, w: int) -> pd.Series:
    m = s.rolling(w, min_periods=max(10, w // 3)).mean()
    sd = s.rolling(w, min_periods=max(10, w // 3)).std(ddof=0).replace(0, np.nan)
    return (s - m) / sd


def fetch_klines(symbol: str, interval: str = "5m", limit: int = 1500) -> pd.DataFrame:
    with httpx.Client(timeout=20) as c:
        r = c.get(f"{REST}/fapi/v1/klines", params={"symbol": symbol, "interval": interval, "limit": limit})
        r.raise_for_status()
        data = r.json()

    df = pd.DataFrame(
        data,
        columns=["openTime","open","high","low","close","volume","closeTime","qv","n","tbbv","tbqv","ignore"]
    )
    df["ts"] = pd.to_datetime(df["openTime"].astype(int), unit="ms", utc=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df[["ts","open","high","low","close","volume"]].sort_values("ts").reset_index(drop=True)


def fetch_premium_index(symbol: str) -> Dict[str, float]:
    with httpx.Client(timeout=10) as c:
        r = c.get(f"{REST}/fapi/v1/premiumIndex", params={"symbol": symbol})
        r.raise_for_status()
        j = r.json()
    mark = float(j["markPrice"])
    index = float(j["indexPrice"])
    funding = float(j["lastFundingRate"])
    basis = (mark - index) / index if index else np.nan
    return {"mark": mark, "index": index, "funding": funding, "basis": basis}


def fetch_open_interest_hist(symbol: str, period: str, start_ms: int, end_ms: int, limit: int = 500) -> pd.DataFrame:
    with httpx.Client(timeout=20) as c:
        r = c.get(
            f"{REST}/futures/data/openInterestHist",
            params={"symbol": symbol, "period": period, "startTime": start_ms, "endTime": end_ms, "limit": limit},
        )
        r.raise_for_status()
        data = r.json()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df["oi"] = df["sumOpenInterest"].astype(float)
    return df[["ts","oi"]].sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


def fetch_oi_series(symbol: str) -> Tuple[pd.DataFrame, str]:
    end = now_ms()
    start = end - 2 * 24 * 60 * 60 * 1000  # 2 days

    # try 5m
    try:
        oi5 = fetch_open_interest_hist(symbol, OI_PERIOD_PRIMARY, start, end, limit=500)
        if not oi5.empty:
            oi5["ts"] = oi5["ts"].dt.floor("5min")
            return oi5, OI_PERIOD_PRIMARY
    except Exception:
        pass

    # fallback 15m -> resample to 5m
    try:
        oi15 = fetch_open_interest_hist(symbol, "15m", start, end, limit=500)
        if oi15.empty:
            return pd.DataFrame(columns=["ts","oi"]), "none"
        oi15["ts"] = oi15["ts"].dt.floor("15min")
        s = oi15.set_index("ts")["oi"].resample("5min").ffill()
        out = pd.DataFrame({"ts": s.index, "oi": s.values})
        return out.reset_index(drop=True), "15m→5m"
    except Exception:
        return pd.DataFrame(columns=["ts","oi"]), "none"


def wick_fraction(row: pd.Series) -> float:
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    rng = max(h - l, 1e-12)
    if c >= o:
        # green: lower wick = o-l
        return float((o - l) / rng)
    else:
        # red: lower wick = c-l
        return float((c - l) / rng)


def compute_liq_sweep_signal(df: pd.DataFrame, prem: Dict[str, float]) -> Dict[str, str]:
    d = df.copy()

    # indicators
    d["ema_fast"] = ema(d["close"], EMA_FAST)
    d["ema_slow"] = ema(d["close"], EMA_SLOW)
    d["trend_strength"] = (d["ema_fast"] - d["ema_slow"]).abs() / d["close"].replace(0, np.nan)
    d["ret_1h"] = d["close"].pct_change(12).abs()  # 12 * 5m = 1h

    d["vwap"] = vwap(d)
    d["atr"] = atr(d, ATR_N)
    d["wick_frac"] = d.apply(wick_fraction, axis=1)

    # sweep levels from PRIOR bars only (no lookahead)
    d["prior_low"] = d["low"].shift(1).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK//2).min()
    d["prior_high"] = d["high"].shift(1).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK//2).max()

    # sweep event: breach prior low, then reclaim above prior low within N bars
    d["swept_low"] = d["low"] < d["prior_low"]
    d["reclaim_low"] = d["close"] > d["prior_low"]

    # bars since last sweep
    d["last_sweep_ts"] = d["ts"].where(d["swept_low"]).ffill()
    d["bars_since_sweep"] = ((d["ts"] - d["last_sweep_ts"]).dt.total_seconds() / 300.0).round().astype("Int64")

    # reclaim within window and on current bar
    d["reclaim_ok"] = d["reclaim_low"] & d["bars_since_sweep"].between(0, RECLAIM_WINDOW)

    # volume spike
    d["vol_q"] = d["volume"].rolling(288, min_periods=60).quantile(VOL_SPIKE_Q)  # 24h quantile-ish
    d["vol_spike"] = d["volume"] >= d["vol_q"]

    # OI merge + z-score
    oi_df, oi_src = fetch_oi_series(SYMBOL)
    if not oi_df.empty:
        d = d.merge(oi_df, on="ts", how="left")
        d["oi"] = d["oi"].ffill()
        d["oi_chg"] = d["oi"].diff()
        d["oi_z"] = zscore(d["oi_chg"], OI_Z_WIN)
    else:
        d["oi_z"] = np.nan
        oi_src = "none"

    last = d.iloc[-1]
    funding = float(prem.get("funding", np.nan))

    # Trend-day filter (mean-reversion hates strong trend)
    trend_ok = (float(last["trend_strength"]) <= TREND_STRENGTH_MAX) and (float(last["ret_1h"]) <= MOMENTUM_1H_MAX)

    # Long liquidation sweep reversal conditions
    liq_ok = (np.isfinite(last["oi_z"]) and float(last["oi_z"]) <= OI_DROP_Z) or (not np.isfinite(last["oi_z"]))
    wick_ok = float(last["wick_frac"]) >= WICK_MIN
    vol_ok = bool(last["vol_spike"])

    long_ok = bool(last["reclaim_ok"]) and trend_ok and liq_ok and wick_ok and vol_ok and (funding >= FUNDING_BAD_LONG)

    # Suggested levels
    if np.isfinite(last["atr"]) and last["atr"] > 0:
        stop = float(last["low"] - STOP_ATR_MULT * last["atr"])
    else:
        stop = float(last["low"] * 0.997)

    tp = float(last["vwap"]) if TP_TO_VWAP and np.isfinite(last["vwap"]) else float(last["close"] * 1.01)

    if long_ok:
        return {
            "side": "LONG",
            "reason": f"liq-sweep: breach {SWEEP_LOOKBACK*5}m low + OI drop + reclaim + wick+vol (OI={oi_src})",
            "stop": f"{stop:.2f}",
            "tp": f"{tp:.2f}",
            "oi_src": oi_src,
        }

    # Short version (optional): sweep high + reclaim down + OI drop too (short liquidations)
    # For now keep it long-only (ETH has strong squeeze/reclaim dynamics).
    return {"side": "FLAT", "reason": f"no setup (OI={oi_src})", "stop": "—", "tp": "—", "oi_src": oi_src}


def line(df: pd.DataFrame, y: str, title: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("ts:T", title="Time", scale=alt.Scale(zero=False)),
            y=alt.Y(f"{y}:Q", title=title, scale=alt.Scale(zero=False)),
            tooltip=["ts:T", alt.Tooltip(f"{y}:Q")],
        )
        .properties(height=280)
        .interactive()
    )


def multi(df: pd.DataFrame, cols: list[str], title: str) -> alt.Chart:
    t = df[["ts"] + cols].melt("ts", var_name="series", value_name="value")
    return (
        alt.Chart(t)
        .mark_line()
        .encode(
            x=alt.X("ts:T", title="Time", scale=alt.Scale(zero=False)),
            y=alt.Y("value:Q", title=title, scale=alt.Scale(zero=False)),
            color="series:N",
            tooltip=["ts:T", "series:N", "value:Q"],
        )
        .properties(height=320)
        .interactive()
    )


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="ETH 5m Liquidation Sweep Reversal", layout="wide")
st.title("ETHUSDT — 5m Liquidation Sweep Reversal (OI + reclaim + wick + volume)")

with st.sidebar:
    refresh_sec = st.slider("Refresh (seconds)", 3, 60, 10)
    show_hours = st.selectbox("Chart window (hours)", [6, 12, 24, 48], index=2)
    auto_refresh = st.toggle("Auto-refresh", value=True)

# Fetch
df = fetch_klines(SYMBOL, "5m", KLINES_LIMIT)
prem = fetch_premium_index(SYMBOL)

sig = compute_liq_sweep_signal(df, prem)

# indicators for chart window
cut = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=int(show_hours))
dw = df[df["ts"] >= cut].copy()
dw["vwap"] = vwap(dw)
dw["ema_fast"] = ema(dw["close"], EMA_FAST)
dw["ema_slow"] = ema(dw["close"], EMA_SLOW)

# Header metrics
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Symbol", SYMBOL)
c2.metric("Signal", sig["side"])
c3.metric("Stop (suggested)", sig["stop"])
c4.metric("TP (suggested)", sig["tp"])
c5.metric("Funding (last)", f"{prem['funding']:.6f}")
c6.metric("Basis", f"{prem['basis']:.6f}")

st.caption(sig["reason"])

l, r = st.columns(2)
with l:
    st.subheader("Price + VWAP + EMAs")
    st.altair_chart(multi(dw, ["close", "vwap", "ema_fast", "ema_slow"], "Price & Trend"), use_container_width=True)

with r:
    st.subheader("Volume")
    st.altair_chart(line(dw, "volume", "Volume"), use_container_width=True)

st.button("Refresh now")

if auto_refresh:
    time.sleep(float(refresh_sec))
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()
