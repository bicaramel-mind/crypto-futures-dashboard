# app.py
import json
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt
from websocket import WebSocketApp

# =========================
# Config
# =========================
FAPI_REST = "https://fapi.binance.com"
FAPI_WS_BASE = "wss://fstream.binance.com/stream?streams="
SYMBOL = "ETHUSDT"

def now_ms() -> int:
    return int(time.time() * 1000)

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def zscore(x: pd.Series, window: int) -> pd.Series:
    m = x.rolling(window).mean()
    s = x.rolling(window).std(ddof=0)
    return (x - m) / (s.replace(0, np.nan))

def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev_c = c.shift(1)
    return pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)

def atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(h, l, c)
    return tr.ewm(alpha=1/period, adjust=False).mean()

@dataclass
class LiqEvent:
    ts: int
    side: str  # BUY/SELL
    price: float
    qty: float

    @property
    def notional(self) -> float:
        return self.price * self.qty

# =========================
# Shared state
# =========================
class SharedState:
    def __init__(self, max_candles=30000, max_liqs=200000, max_oi_points=100000):
        self.lock = threading.Lock()
        self.candles_1m: Deque[dict] = deque(maxlen=max_candles)
        self.mark_price: Optional[float] = None
        self.last_mark_ts: Optional[int] = None
        self.liqs: Deque[LiqEvent] = deque(maxlen=max_liqs)

        self.oi_samples: Deque[dict] = deque(maxlen=max_oi_points)
        self.oi_last_poll_ms: int = 0
        self.oi_errors: Deque[str] = deque(maxlen=200)

        self.ws_errors: Deque[str] = deque(maxlen=300)
        self.ws_last_msg_ms: int = 0

        self.did_backfill: bool = False
        self.backfill_error: Optional[str] = None

STATE = SharedState()

# =========================
# REST helpers
# =========================
def rest_get_json(path: str, params: dict, timeout=10):
    url = FAPI_REST + path
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def rest_backfill_klines(symbol: str, interval: str, limit: int) -> List[dict]:
    data = rest_get_json("/fapi/v1/klines", {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": int(limit)
    })
    out = []
    for row in data:
        out.append({
            "ts": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "vol": float(row[5]),
        })
    return out

def rest_fetch_mark_price(symbol: str) -> Optional[float]:
    try:
        data = rest_get_json("/fapi/v1/premiumIndex", {"symbol": symbol.upper()})
        p = safe_float(data.get("markPrice"))
        return p if np.isfinite(p) else None
    except Exception:
        return None

def rest_fetch_open_interest(symbol: str) -> Optional[float]:
    try:
        data = rest_get_json("/fapi/v1/openInterest", {"symbol": symbol.upper()})
        oi = safe_float(data.get("openInterest"))
        return oi if np.isfinite(oi) else None
    except Exception as e:
        with STATE.lock:
            STATE.oi_errors.append(f"openInterest poll error: {repr(e)}")
        return None

def ensure_startup_backfill(minutes: int = 240):
    """
    Backfill more than 1h so 1m indicators have enough history.
    Default 240m (4h) -> still lightweight (240 rows).
    """
    with STATE.lock:
        if STATE.did_backfill:
            return
    try:
        candles = rest_backfill_klines(SYMBOL, "1m", limit=min(1500, minutes))
        mp = rest_fetch_mark_price(SYMBOL)
        candles = sorted(candles, key=lambda x: x["ts"])

        with STATE.lock:
            STATE.candles_1m.clear()
            for c in candles:
                STATE.candles_1m.append(c)
            if mp is not None:
                STATE.mark_price = mp
                STATE.last_mark_ts = now_ms()
            STATE.did_backfill = True
            STATE.backfill_error = None
    except Exception as e:
        with STATE.lock:
            STATE.did_backfill = True
            STATE.backfill_error = repr(e)

# =========================
# OI poller thread (fast OI)
# =========================
def oi_poller(symbol: str, poll_secs: int, stop_flag: threading.Event):
    while not stop_flag.is_set():
        oi = rest_fetch_open_interest(symbol)
        ts = pd.Timestamp.utcnow().tz_localize("UTC")
        with STATE.lock:
            if oi is not None:
                STATE.oi_samples.append({"ts": ts, "oi": float(oi)})
                STATE.oi_last_poll_ms = now_ms()
        # sleep in 1s chunks so stop_flag responds quickly
        for _ in range(max(1, poll_secs)):
            if stop_flag.is_set():
                break
            time.sleep(1)

# =========================
# Websocket worker (1m klines)
# =========================
def make_stream_url(symbol: str) -> str:
    s = symbol.lower()
    streams = [
        f"{s}@kline_1m",
        f"{s}@markPrice@1s",
        f"{s}@forceOrder",
    ]
    return FAPI_WS_BASE + "/".join(streams)

def upsert_candle_1m(candle: dict):
    ts = candle["ts"]
    if len(STATE.candles_1m) == 0:
        STATE.candles_1m.append(candle)
        return
    last_ts = STATE.candles_1m[-1]["ts"]
    if ts == last_ts:
        STATE.candles_1m[-1] = candle
    elif ts > last_ts:
        STATE.candles_1m.append(candle)
    else:
        return

def ws_on_message(ws: WebSocketApp, message: str):
    STATE.ws_last_msg_ms = now_ms()
    try:
        payload = json.loads(message)
        data = payload.get("data", {})
        stream = payload.get("stream", "")

        with STATE.lock:
            if "@kline_" in stream:
                k = data.get("k", {})
                if not k:
                    return
                candle = {
                    "ts": int(k["t"]),
                    "open": safe_float(k["o"]),
                    "high": safe_float(k["h"]),
                    "low": safe_float(k["l"]),
                    "close": safe_float(k["c"]),
                    "vol": safe_float(k["v"]),
                }
                if all(np.isfinite([candle["open"], candle["high"], candle["low"], candle["close"], candle["vol"]])):
                    upsert_candle_1m(candle)

            elif "@markPrice@" in stream:
                p = safe_float(data.get("p"))
                if np.isfinite(p):
                    STATE.mark_price = p
                    STATE.last_mark_ts = int(data.get("E", now_ms()))

            elif "@forceOrder" in stream:
                o = data.get("o", {})
                if not o:
                    return
                side = str(o.get("S", "")).upper()
                price = safe_float(o.get("p"))
                qty = safe_float(o.get("q"))
                ts = int(o.get("T", now_ms()))
                if side in ("BUY", "SELL") and np.isfinite(price) and np.isfinite(qty):
                    STATE.liqs.append(LiqEvent(ts=ts, side=side, price=price, qty=qty))

    except Exception as e:
        with STATE.lock:
            STATE.ws_errors.append(f"on_message error: {repr(e)}")

def ws_on_error(ws: WebSocketApp, error: Exception):
    with STATE.lock:
        STATE.ws_errors.append(f"ws error: {repr(error)}")

def ws_on_close(ws: WebSocketApp, close_status_code, close_msg):
    with STATE.lock:
        STATE.ws_errors.append(f"ws closed: {close_status_code} {close_msg}")

def ws_worker(symbol: str, stop_flag: threading.Event):
    backoff = 1.0
    while not stop_flag.is_set():
        ws = WebSocketApp(make_stream_url(symbol), on_message=ws_on_message, on_error=ws_on_error, on_close=ws_on_close)
        try:
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            with STATE.lock:
                STATE.ws_errors.append(f"run_forever error: {repr(e)}")
        if stop_flag.is_set():
            break
        time.sleep(backoff)
        backoff = min(backoff * 1.7, 30.0)

# =========================
# Liquidation zones + OI series
# =========================
def liquidation_features(ldf: pd.DataFrame, lookback_min: int, zone_bins: int, top_k: int) -> Dict:
    out = {"liq_buy": 0.0, "liq_sell": 0.0, "zones": []}
    if ldf.empty:
        return out
    cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(minutes=lookback_min)
    recent = ldf[ldf["ts"] >= cutoff].copy()
    if recent.empty:
        return out
    out["liq_buy"] = float(recent.loc[recent["side"] == "BUY", "notional"].sum())
    out["liq_sell"] = float(recent.loc[recent["side"] == "SELL", "notional"].sum())

    zones = []
    for side in ["BUY", "SELL"]:
        rs = recent[recent["side"] == side]
        if len(rs) < 15:
            continue
        prices = rs["price"].values
        w = rs["notional"].values
        pmin, pmax = float(np.min(prices)), float(np.max(prices))
        if not np.isfinite(pmin) or not np.isfinite(pmax) or pmin == pmax:
            continue
        hist, edges = np.histogram(prices, bins=zone_bins, range=(pmin, pmax), weights=w)
        idxs = np.argsort(hist)[::-1][:top_k]
        for i in idxs:
            center = 0.5 * (edges[i] + edges[i + 1])
            zones.append({"price": float(center), "strength": float(hist[i]), "side": side})
    out["zones"] = sorted(zones, key=lambda z: z["strength"], reverse=True)[:top_k * 2]
    return out

def nearest_zone(px: float, zones: List[Dict]) -> Tuple[Optional[Dict], Optional[float]]:
    if not zones:
        return None, None
    best, best_dist = None, None
    for z in zones:
        dist_bps = abs(px - z["price"]) / px * 1e4
        if best is None or dist_bps < best_dist:
            best, best_dist = z, dist_bps
    return best, float(best_dist) if best_dist is not None else None

def build_oi_1m(oi_samples: pd.DataFrame) -> pd.DataFrame:
    if oi_samples.empty:
        return pd.DataFrame(columns=["ts","oi","d_oi"])
    df = oi_samples.dropna(subset=["ts","oi"]).sort_values("ts").set_index("ts")
    oi_1m = df["oi"].resample("1min").last().ffill()
    out = oi_1m.to_frame("oi").reset_index()
    out["d_oi"] = out["oi"].diff()
    return out

# =========================
# 1m indicators + signal
# =========================
def indicators_1m(c1: pd.DataFrame) -> pd.DataFrame:
    if c1.empty:
        return pd.DataFrame(columns=["ts","open","high","low","close","vol",
                                     "ema_fast","ema_slow","ema_trend","macd","macd_sig","macd_hist",
                                     "rsi","vol_z","atr","trend_slope"])
    df = c1.copy()
    df["ema_fast"] = ema(df["close"], 12)
    df["ema_slow"] = ema(df["close"], 26)
    df["ema_trend"] = ema(df["close"], 200)  # ~3h20m trend anchor on 1m
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["macd_sig"] = ema(df["macd"], 9)
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    df["rsi"] = rsi(df["close"], 14)
    df["vol_z"] = zscore(df["vol"], 120)     # 2h vol impulse baseline
    df["atr"] = atr(df["high"], df["low"], df["close"], 14)
    df["trend_slope"] = df["ema_trend"].diff(30) / df["ema_trend"].shift(30)  # ~30m slope on trend anchor
    return df

def signal_1m(df1: pd.DataFrame, oi1m: pd.DataFrame, liq: Dict, px: float,
              zone_near_bps: float, liq_gate_usd: float) -> Dict:
    if df1.empty or len(df1) < 240:  # want a few hours for stable vol_z/trend
        return {"regime":"none","signal":"WAIT","confidence":0,"plan":{}, "debug":{"reason":"need more 1m history (backfill more)"}}

    last = df1.iloc[-1]
    li_total = float(liq["liq_buy"] + liq["liq_sell"])

    doi_1m = 0.0
    if oi1m is not None and not oi1m.empty and len(oi1m) >= 2:
        doi_1m = float(oi1m["d_oi"].iloc[-1])

    macd_hist = float(last.get("macd_hist", np.nan))
    trend_slope = float(last.get("trend_slope", np.nan))
    vol_z = float(last.get("vol_z", np.nan))
    rsi14 = float(last.get("rsi", np.nan))
    atr1 = float(last.get("atr", np.nan))

    if not np.isfinite(macd_hist) or not np.isfinite(trend_slope):
        return {"regime":"none","signal":"WAIT","confidence":0,"plan":{}, "debug":{"reason":"indicators warming up"}}

    macd_std = float(df1["macd_hist"].rolling(400).std(ddof=0).iloc[-1]) if len(df1) > 400 else float(df1["macd_hist"].std(ddof=0))
    macd_norm = float(np.clip(macd_hist / (macd_std + 1e-9), -3, 3))

    zbest, zdist = nearest_zone(px, liq.get("zones", []))
    near_zone = (zdist is not None) and (zdist <= zone_near_bps)

    oi_adding = doi_1m > 0
    oi_flushing = doi_1m < 0

    # hour-hold tuned gates on 1m
    momentum_up = (macd_norm > 0.35) and (trend_slope > 0)
    momentum_dn = (macd_norm < -0.35) and (trend_slope < 0)
    vol_impulse = np.isfinite(vol_z) and (vol_z > 0.9)

    cascade_up = (li_total >= liq_gate_usd) and vol_impulse and oi_adding and momentum_up
    cascade_dn = (li_total >= liq_gate_usd) and vol_impulse and oi_adding and momentum_dn

    prev_hist = df1["macd_hist"].iloc[-5] if len(df1) >= 5 else df1["macd_hist"].iloc[-2]
    momentum_decel = abs(macd_norm) < 0.35 or (np.sign(macd_norm) != np.sign(prev_hist))
    exhaustion = (li_total >= liq_gate_usd) and oi_flushing and momentum_decel and near_zone

    regime, sig = "none", "WAIT"
    if cascade_up:
        regime, sig = "cascade", "LONG"
    elif cascade_dn:
        regime, sig = "cascade", "SHORT"
    elif exhaustion:
        regime = "exhaustion"
        if zbest and zbest["side"] == "BUY" and np.isfinite(rsi14) and rsi14 > 55:
            sig = "SHORT"
        elif zbest and zbest["side"] == "SELL" and np.isfinite(rsi14) and rsi14 < 45:
            sig = "LONG"

    # Plan (ATR based, but 1m ATR)
    plan = {}
    if np.isfinite(atr1) and atr1 > 0 and sig in ("LONG","SHORT"):
        if regime == "cascade":
            sl_mult, tp_mult = 2.2, 4.0
        else:
            sl_mult, tp_mult = 1.6, 2.6
        if sig == "LONG":
            plan = {"entry": px, "stop": px - sl_mult*atr1, "tp": px + tp_mult*atr1, "atr1m": atr1, "time_stop_min": 360}
        else:
            plan = {"entry": px, "stop": px + sl_mult*atr1, "tp": px - tp_mult*atr1, "atr1m": atr1, "time_stop_min": 360}

    conf = 0.0
    conf += 20 * (1.0 if li_total >= liq_gate_usd else 0.0)
    conf += 14 * (max(0.0, min(3.0, vol_z)) if np.isfinite(vol_z) else 0.0)
    conf += 14 * abs(macd_norm)
    conf += 10 * (1.0 if (oi_adding and regime == "cascade") or (oi_flushing and regime == "exhaustion") else 0.0)
    conf += 8 * (1.0 if near_zone else 0.0)
    conf = float(np.clip(conf, 0, 100))

    return {
        "regime": regime,
        "signal": sig,
        "confidence": int(round(conf)),
        "plan": plan,
        "debug": {
            "px": px,
            "liq_total_usd": li_total,
            "doi_1m": doi_1m,
            "macd_norm": macd_norm,
            "trend_slope": trend_slope,
            "vol_z": vol_z,
            "rsi14": rsi14,
            "near_zone": near_zone,
            "zone_dist_bps": zdist,
            "zone_side": (zbest["side"] if zbest else None),
            "zone_px": (zbest["price"] if zbest else None),
        }
    }

# =========================
# Altair chart helpers
# =========================
def alt_candles_1m(df: pd.DataFrame, zones: List[Dict], title: str, window: int = 300):
    d = df.tail(window).copy()
    if d.empty:
        st.info("No candle data yet.")
        return

    y_min = float(d["low"].min())
    y_max = float(d["high"].max())
    pad = (y_max - y_min) * 0.02 if y_max > y_min else (y_max * 0.01 if y_max else 1.0)
    y_domain = [y_min - pad, y_max + pad]

    d["dir"] = np.where(d["close"] >= d["open"], "up", "down")

    base = alt.Chart(d).encode(
        x=alt.X("ts:T", axis=alt.Axis(title="Time", labelAngle=-45)),
    )

    wicks = base.mark_rule().encode(
        y=alt.Y("low:Q", scale=alt.Scale(domain=y_domain), axis=alt.Axis(title="Price")),
        y2="high:Q",
        tooltip=["ts:T","open:Q","high:Q","low:Q","close:Q","vol:Q"]
    )

    bodies = base.mark_bar().encode(
        y=alt.Y("open:Q", scale=alt.Scale(domain=y_domain)),
        y2="close:Q",
        color=alt.Color("dir:N", legend=None),
        tooltip=["ts:T","open:Q","high:Q","low:Q","close:Q","vol:Q","dir:N"]
    )

    chart = (wicks + bodies).properties(title=title, height=420).interactive()

    if zones:
        zdf = pd.DataFrame(zones[:6])
        z = alt.Chart(zdf).mark_rule(strokeDash=[4, 4]).encode(
            y=alt.Y("price:Q", scale=alt.Scale(domain=y_domain)),
            tooltip=["side:N","price:Q","strength:Q"]
        )
        chart = chart + z

    st.altair_chart(chart, use_container_width=True)

def alt_lines(df: pd.DataFrame, x_col: str, series: Dict[str, str], title: str, height: int = 220):
    if df.empty:
        st.info("No data yet.")
        return
    cols = [c for c in series.values() if c in df.columns]
    if not cols:
        st.info("Series not ready.")
        return
    tmp = df[[x_col] + cols].copy().melt(id_vars=[x_col], var_name="series", value_name="value")
    inv = {v: k for k, v in series.items()}
    tmp["series"] = tmp["series"].map(inv).fillna(tmp["series"])

    chart = alt.Chart(tmp).mark_line().encode(
        x=alt.X(f"{x_col}:T", axis=alt.Axis(title="Time", labelAngle=-45)),
        y=alt.Y("value:Q", axis=alt.Axis(title="Value")),
        color=alt.Color("series:N", legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip(f"{x_col}:T"), "series:N", alt.Tooltip("value:Q")]
    ).properties(title=title, height=height).interactive()
    st.altair_chart(chart, use_container_width=True)

def alt_liq_bars(ldf: pd.DataFrame, title: str, window_min: int = 240):
    if ldf.empty:
        st.info("No liquidations yet.")
        return
    cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(minutes=window_min)
    tmp = ldf[ldf["ts"] >= cutoff].copy()
    if tmp.empty:
        st.info("No liquidations in window.")
        return
    tmp["minute"] = tmp["ts"].dt.floor("min")
    g = tmp.groupby(["minute","side"], as_index=False)["notional"].sum()

    chart = alt.Chart(g).mark_bar().encode(
        x=alt.X("minute:T", axis=alt.Axis(title="Time", labelAngle=-45)),
        y=alt.Y("notional:Q", axis=alt.Axis(title="Notional (USD)")),
        color=alt.Color("side:N", legend=alt.Legend(orient="top")),
        tooltip=["minute:T","side:N",alt.Tooltip("notional:Q", format=",.0f")]
    ).properties(title=title, height=260).interactive()
    st.altair_chart(chart, use_container_width=True)

# =========================
# Snapshot
# =========================
def snapshot():
    with STATE.lock:
        c1 = pd.DataFrame(list(STATE.candles_1m))
        mp = STATE.mark_price
        ldf = pd.DataFrame([{
            "ts": pd.to_datetime(e.ts, unit="ms", utc=True),
            "side": e.side,
            "price": e.price,
            "qty": e.qty,
            "notional": e.notional
        } for e in list(STATE.liqs)])
        oi_s = pd.DataFrame(list(STATE.oi_samples))
        meta = {
            "ws_last_msg_ms": STATE.ws_last_msg_ms,
            "ws_errors": list(STATE.ws_errors)[-10:],
            "did_backfill": STATE.did_backfill,
            "backfill_error": STATE.backfill_error,
            "oi_last_poll_ms": STATE.oi_last_poll_ms,
            "oi_errors": list(STATE.oi_errors)[-8:],
        }

    if not c1.empty:
        c1["ts"] = pd.to_datetime(c1["ts"], unit="ms", utc=True)
        c1 = c1.sort_values("ts").reset_index(drop=True)
    if not ldf.empty:
        ldf = ldf.sort_values("ts").reset_index(drop=True)
    if not oi_s.empty and not pd.api.types.is_datetime64_any_dtype(oi_s["ts"]):
        oi_s["ts"] = pd.to_datetime(oi_s["ts"], utc=True)

    df1 = indicators_1m(c1)
    oi1m = build_oi_1m(oi_s) if not oi_s.empty else pd.DataFrame(columns=["ts","oi","d_oi"])
    return df1, ldf, mp, oi_s, oi1m, meta

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="ETHUSDT 1m LIQVOM-OI", layout="wide")
st.title("ETHUSDT — 1m Candles (Altair) + Liquidations + 1m OI (polled)")

with st.sidebar:
    st.header("Refresh")
    auto_refresh = st.checkbox("Auto-refresh UI", value=True)
    refresh_ms = st.slider("Refresh interval (ms)", 200, 5000, 750, 50)

    st.divider()
    st.header("Backfill")
    backfill_min = st.slider("Backfill minutes (for 1m indicators)", 60, 1500, 240, 30)

    st.divider()
    st.header("OI polling")
    oi_poll_secs = st.slider("Poll openInterest every (sec)", 2, 30, 5, 1)

    st.divider()
    st.header("Strategy knobs (1m)")
    liq_lookback_min = st.slider("Liquidation lookback (min)", 30, 240, 120, 10)
    liq_gate_usd = st.number_input("Min liquidation gate (USD)", 50_000, 50_000_000, 600_000, 50_000)
    zone_bins = st.slider("Zone bins", 20, 180, 70, 10)
    top_k = st.slider("Top zones per side", 1, 8, 3, 1)
    zone_near_bps = st.slider("Zone proximity (bps)", 5, 120, 30, 5)

    st.divider()
    st.header("Your sizing")
    equity = st.number_input("Account equity (USD)", min_value=1.0, value=100.0, step=10.0)
    leverage = st.number_input("Leverage", min_value=1, max_value=200, value=150, step=1)
    margin_frac = st.slider("Margin fraction / trade", 0.01, 0.95, 0.40, 0.01)

# Backfill (only once per session; if you change slider, you should restart app for now)
ensure_startup_backfill(minutes=int(backfill_min))

# Start threads
if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = threading.Event()

if "ws_thread" not in st.session_state:
    st.session_state.ws_thread = threading.Thread(target=ws_worker, args=(SYMBOL, st.session_state.stop_flag), daemon=True)
    st.session_state.ws_thread.start()

if "oi_poll_cfg" not in st.session_state:
    st.session_state.oi_poll_cfg = None

if st.session_state.oi_poll_cfg != oi_poll_secs:
    st.session_state.oi_poll_cfg = oi_poll_secs
    st.session_state.oi_thread = threading.Thread(
        target=oi_poller, args=(SYMBOL, int(oi_poll_secs), st.session_state.stop_flag), daemon=True
    )
    st.session_state.oi_thread.start()

# Snapshot
df1, ldf, mp, oi_s, oi1m, meta = snapshot()

px_now = float(mp) if (mp is not None and np.isfinite(mp)) else (float(df1["close"].iloc[-1]) if not df1.empty else np.nan)

liq = liquidation_features(ldf, lookback_min=liq_lookback_min, zone_bins=zone_bins, top_k=top_k)
sig = signal_1m(df1, oi1m, liq, px_now if np.isfinite(px_now) else 0.0, zone_near_bps, float(liq_gate_usd))

# Sizing
margin_used = equity * margin_frac
notional = margin_used * leverage
rough_margin_wipe_move_pct = (1.0 / leverage) * 100.0

col1, col2, col3 = st.columns([1.1, 1.0, 1.0])
with col1:
    st.subheader("Signal")
    st.metric("Regime", sig["regime"])
    st.metric("Action", sig["signal"])
    st.metric("Confidence", f"{sig['confidence']}/100")
    if sig.get("plan"):
        st.write("Plan:", sig["plan"])
    st.caption(f"Mark/Last: {px_now:.2f} | 1m rows: {len(df1)} | liqs: {len(ldf)} | OI samples: {len(oi_s)}")

with col2:
    st.subheader("Flow")
    st.metric("Liq BUY", f"{liq['liq_buy']:,.0f} USD")
    st.metric("Liq SELL", f"{liq['liq_sell']:,.0f} USD")
    if not oi1m.empty and len(oi1m) >= 2:
        st.metric("OI (1m)", f"{oi1m['oi'].iloc[-1]:,.0f}")
        st.metric("ΔOI (1m)", f"{oi1m['d_oi'].iloc[-1]:,.0f}")
    else:
        st.info("OI warming up (need a few polls).")

with col3:
    st.subheader("Sizing (150x / 40%)")
    st.metric("Margin used", f"${margin_used:,.2f}")
    st.metric("Notional exposure", f"${notional:,.2f}")
    st.caption(f"Rough: ~{rough_margin_wipe_move_pct:.2f}% adverse move ≈ 100% of margin at {leverage}x (ignores maintenance/fees).")

st.divider()

with st.expander("Health / diagnostics", expanded=False):
    ws_age = (now_ms() - meta["ws_last_msg_ms"]) if meta["ws_last_msg_ms"] else None
    oi_age = (now_ms() - meta["oi_last_poll_ms"]) if meta["oi_last_poll_ms"] else None
    st.write(f"Backfill done: {meta['did_backfill']}, error: {meta['backfill_error']}")
    st.write(f"WS last msg age (ms): {ws_age if ws_age is not None else '—'}")
    st.write(f"OI last poll age (ms): {oi_age if oi_age is not None else '—'}")
    if meta["ws_errors"]:
        st.error("Recent WS errors:\n" + "\n".join(meta["ws_errors"]))
    if meta["oi_errors"]:
        st.error("Recent OI errors:\n" + "\n".join(meta["oi_errors"]))
    if not meta["ws_errors"] and not meta["oi_errors"]:
        st.success("No recent thread errors captured.")

# Charts
alt_candles_1m(df1, liq.get("zones", []), title="ETHUSDT 1m Candles + Liquidation Zones", window=300)

cA, cB = st.columns(2)
with cA:
    alt_lines(df1.tail(800), "ts", {"MACD hist":"macd_hist", "RSI":"rsi"}, "Momentum (1m)", height=260)
with cB:
    alt_lines(df1.tail(800), "ts", {"Volume z":"vol_z", "ATR":"atr"}, "Impulse + Volatility (1m)", height=260)

cC, cD = st.columns(2)
with cC:
    alt_liq_bars(ldf, title="Liquidation Notional per Minute", window_min=240)
with cD:
    if not oi1m.empty:
        alt_lines(oi1m.tail(1200), "ts", {"OI (1m)":"oi", "ΔOI (1m)":"d_oi"}, "Open Interest (polled → 1m)", height=260)
    else:
        st.info("OI series not ready yet.")

st.subheader("Debug")
st.json(sig["debug"])

if st.button("Stop background threads (manual)"):
    st.session_state.stop_flag.set()
    st.warning("Stop flag set. Rerun the app to restart threads.")

# UI refresh heartbeat
if auto_refresh:
    time.sleep(refresh_ms / 1000.0)
    st.rerun()
