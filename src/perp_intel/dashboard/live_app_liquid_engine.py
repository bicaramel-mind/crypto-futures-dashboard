# app.py
# ETHUSDT Live Dashboard (Binance Futures)
# ✅ 1m live feed via WebSocket: kline_1m + markPrice@1s + forceOrder (liquidations)
# ✅ REST backfill for 1m candles on startup (so charts + resample work immediately)
# ✅ REST backfill for HTF candles (15m/30m/1h) for the past 3 days BEFORE WS starts
# ✅ OI history via REST /futures/data/openInterestHist (same call as before)
# ✅ Liquidation “liquidity levels” = liquidation heatmap zones (weighted price histogram)
# ✅ Signals computed on HTF (15m–1h) while UI ticks minute-to-minute
# ✅ Altair charts with properly padded y-domains + zones overlay
#
# Run:
#   pip install streamlit pandas numpy requests websocket-client altair
#   streamlit run app.py

import json
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt
from websocket import WebSocketApp

# =========================
# Config
# =========================
FAPI_REST = "https://fapi.binance.us"
FAPI_WS_BASE = "wss://fstream.binance.us/stream?streams="
SYMBOL = "ETHUSDT"

# =========================
# Time helpers
# =========================
def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")

def now_ms() -> int:
    return int(time.time() * 1000)

# =========================
# Math helpers
# =========================
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
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
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
    return tr.ewm(alpha=1 / period, adjust=False).mean()

# =========================
# Data classes
# =========================
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
    def __init__(self, max_candles=60000, max_liqs=300000, max_errors=300):
        self.lock = threading.Lock()

        self.candles_1m: Deque[dict] = deque(maxlen=max_candles)
        self.mark_price: Optional[float] = None
        self.last_mark_ts: Optional[int] = None

        self.liqs: Deque[LiqEvent] = deque(maxlen=max_liqs)

        # OI history
        self.oi_hist: Optional[pd.DataFrame] = None
        self.oi_last_refresh_ms: int = 0
        self.oi_errors: Deque[str] = deque(maxlen=max_errors)

        # OI HTTP debug
        self.oi_last_status: Optional[int] = None
        self.oi_last_body: Optional[str] = None
        self.oi_thread_running: bool = False

        # WS debug
        self.ws_last_msg_ms: int = 0
        self.ws_errors: Deque[str] = deque(maxlen=max_errors)

        # 1m backfill tracking
        self.backfill_minutes: Optional[int] = None
        self.backfill_error: Optional[str] = None

        # ✅ HTF seed backfill (REST) so signals work immediately
        self.htf_seed: Dict[str, pd.DataFrame] = {}         # tf -> df(ts, o,h,l,c,vol)
        self.htf_seed_meta: Dict[str, dict] = {}            # tf -> debug
        self.htf_backfill_error: Optional[str] = None

def get_state() -> SharedState:
    if "shared_state" not in st.session_state:
        st.session_state.shared_state = SharedState()
    return st.session_state.shared_state

# =========================
# REST helpers
# =========================
def rest_get_json(state: SharedState, path: str, params: dict, timeout=10) -> Any:
    url = FAPI_REST + path
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, params=params, timeout=timeout, headers=headers)

    if path == "/futures/data/openInterestHist":
        with state.lock:
            state.oi_last_status = r.status_code
            state.oi_last_body = (r.text or "")[:400]

    r.raise_for_status()
    return r.json()

def rest_backfill_klines(state: SharedState, symbol: str, interval: str, limit: int) -> List[dict]:
    data = rest_get_json(state, "/fapi/v1/klines", {"symbol": symbol.upper(), "interval": interval, "limit": int(limit)})
    out = []
    for row in data:
        out.append(
            {
                "ts": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "vol": float(row[5]),
            }
        )
    return out

def rest_fetch_mark_price(state: SharedState, symbol: str) -> Optional[float]:
    try:
        data = rest_get_json(state, "/fapi/v1/premiumIndex", {"symbol": symbol.upper()})
        p = safe_float(data.get("markPrice"))
        return p if np.isfinite(p) else None
    except Exception:
        return None

def ensure_backfill_1m(state: SharedState, minutes: int):
    with state.lock:
        if state.backfill_minutes == minutes and len(state.candles_1m) > 0:
            return

    try:
        candles = rest_backfill_klines(state, SYMBOL, "1m", limit=min(1500, int(minutes)))
        mp = rest_fetch_mark_price(state, SYMBOL)
        candles = sorted(candles, key=lambda x: x["ts"])
        with state.lock:
            state.candles_1m.clear()
            for c in candles:
                state.candles_1m.append(c)
            if mp is not None:
                state.mark_price = mp
                state.last_mark_ts = now_ms()
            state.backfill_minutes = minutes
            state.backfill_error = None
    except Exception as e:
        with state.lock:
            state.backfill_minutes = minutes
            state.backfill_error = repr(e)

# =========================
# ✅ HTF seed backfill (2–3 days)
# =========================
def tf_to_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    raise ValueError(f"Unsupported tf: {tf}")

def rest_backfill_klines_df(state: SharedState, symbol: str, interval: str, limit: int) -> pd.DataFrame:
    data = rest_get_json(state, "/fapi/v1/klines", {"symbol": symbol.upper(), "interval": interval, "limit": int(limit)})
    rows = []
    for r in data:
        rows.append({
            "ts": pd.to_datetime(int(r[0]), unit="ms", utc=True),
            "open": float(r[1]),
            "high": float(r[2]),
            "low":  float(r[3]),
            "close": float(r[4]),
            "vol":  float(r[5]),
        })
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return df

def ensure_htf_seed_backfill_days(state: SharedState, tf: str, days: int = 3):
    """
    Backfills last `days` of HTF candles via REST and stores into state.htf_seed[tf].
    For 15m: 3 days -> 288 bars (safe).
    Binance limit max 1500; we clamp.
    """
    tf = tf.strip().lower()
    if tf not in ("15m", "30m", "1h"):
        return

    mins = tf_to_minutes(tf)
    bars = int(days * 24 * 60 / mins)
    bars = max(80, min(1500, bars))

    # skip if already present with >= bars (avoid repeated REST on every rerun)
    with state.lock:
        existing = state.htf_seed.get(tf)
        if existing is not None and isinstance(existing, pd.DataFrame) and len(existing) >= bars:
            return

    try:
        df = rest_backfill_klines_df(state, SYMBOL, tf, limit=bars)
        with state.lock:
            state.htf_seed[tf] = df
            state.htf_backfill_error = None
            state.htf_seed_meta[tf] = {
                "rows": int(len(df)),
                "start": str(df["ts"].iloc[0]) if len(df) else None,
                "end": str(df["ts"].iloc[-1]) if len(df) else None,
                "days": int(days),
                "tf": tf,
                "bars_requested": int(bars),
            }
    except Exception as e:
        with state.lock:
            state.htf_backfill_error = repr(e)
            state.htf_seed_meta[tf] = {"rows": 0, "error": repr(e), "days": int(days), "tf": tf}

# =========================
# OI history
# =========================
def fetch_open_interest_hist(state: SharedState, symbol: str, period: str = "5m", limit: int = 200) -> pd.DataFrame:
    data = rest_get_json(
        state,
        "/futures/data/openInterestHist",
        {"symbol": symbol.upper(), "period": period, "limit": int(limit)},
    )
    if not isinstance(data, list):
        raise RuntimeError(f"openInterestHist returned non-list: {str(data)[:200]}")
    if len(data) == 0:
        raise RuntimeError("openInterestHist returned empty list")

    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["oi"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    df["oi_usd"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
    df = df[["ts", "oi", "oi_usd"]].dropna().sort_values("ts").reset_index(drop=True)

    df["d_oi"] = df["oi"].diff()
    df["d_oi_usd"] = df["oi_usd"].diff()
    df["d_oi_usd_z"] = zscore(df["d_oi_usd"].fillna(0.0), 60)
    return df

def oi_hist_refresher(state: SharedState, symbol: str, period: str, limit: int, refresh_secs: int, stop_flag: threading.Event):
    with state.lock:
        state.oi_thread_running = True

    while not stop_flag.is_set():
        try:
            df = fetch_open_interest_hist(state, symbol, period=period, limit=limit)
            with state.lock:
                state.oi_hist = df
                state.oi_last_refresh_ms = now_ms()
        except Exception as e:
            with state.lock:
                state.oi_errors.append(f"openInterestHist error: {repr(e)}")

        for _ in range(max(1, refresh_secs)):
            if stop_flag.is_set():
                break
            time.sleep(1)

    with state.lock:
        state.oi_thread_running = False

# =========================
# Websocket
# =========================
def make_stream_url(symbol: str) -> str:
    s = symbol.lower()
    return FAPI_WS_BASE + "/".join([f"{s}@kline_1m", f"{s}@markPrice@1s", f"{s}@forceOrder"])

def upsert_candle_1m(state: SharedState, candle: dict):
    ts = candle["ts"]
    if len(state.candles_1m) == 0:
        state.candles_1m.append(candle)
        return
    last_ts = state.candles_1m[-1]["ts"]
    if ts == last_ts:
        state.candles_1m[-1] = candle
    elif ts > last_ts:
        state.candles_1m.append(candle)

def make_ws_callbacks(state: SharedState):
    def on_message(ws: WebSocketApp, message: str):
        with state.lock:
            state.ws_last_msg_ms = now_ms()
        try:
            payload = json.loads(message)
            data = payload.get("data", {})
            stream = payload.get("stream", "")

            with state.lock:
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
                        upsert_candle_1m(state, candle)

                elif "@markPrice@" in stream:
                    p = safe_float(data.get("p"))
                    if np.isfinite(p):
                        state.mark_price = p
                        state.last_mark_ts = int(data.get("E", now_ms()))

                elif "@forceOrder" in stream:
                    o = data.get("o", {})
                    if not o:
                        return
                    side = str(o.get("S", "")).upper()
                    price = safe_float(o.get("p"))
                    qty = safe_float(o.get("q"))
                    ts = int(o.get("T", now_ms()))
                    if side in ("BUY", "SELL") and np.isfinite(price) and np.isfinite(qty):
                        state.liqs.append(LiqEvent(ts=ts, side=side, price=price, qty=qty))

        except Exception as e:
            with state.lock:
                state.ws_errors.append(f"on_message error: {repr(e)}")

    def on_error(ws: WebSocketApp, error: Exception):
        with state.lock:
            state.ws_errors.append(f"ws error: {repr(error)}")

    def on_close(ws: WebSocketApp, close_status_code, close_msg):
        with state.lock:
            state.ws_errors.append(f"ws closed: {close_status_code} {close_msg}")

    return on_message, on_error, on_close

def ws_worker(state: SharedState, symbol: str, stop_flag: threading.Event):
    backoff = 1.0
    on_message, on_error, on_close = make_ws_callbacks(state)

    while not stop_flag.is_set():
        ws = WebSocketApp(make_stream_url(symbol), on_message=on_message, on_error=on_error, on_close=on_close)
        try:
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            with state.lock:
                state.ws_errors.append(f"run_forever error: {repr(e)}")
        if stop_flag.is_set():
            break
        time.sleep(backoff)
        backoff = min(backoff * 1.7, 30.0)

# =========================
# Indicators (timeframe-agnostic)
# =========================
def indicators_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["ema_fast"] = ema(out["close"], 12)
    out["ema_slow"] = ema(out["close"], 26)
    out["ema_trend"] = ema(out["close"], 200)
    out["macd"] = out["ema_fast"] - out["ema_slow"]
    out["macd_sig"] = ema(out["macd"], 9)
    out["macd_hist"] = out["macd"] - out["macd_sig"]
    out["rsi"] = rsi(out["close"], 14)
    out["vol_z"] = zscore(out["vol"], 120)
    out["atr"] = atr(out["high"], out["low"], out["close"], 14)
    out["trend_slope"] = out["ema_trend"].diff(30) / out["ema_trend"].shift(30)
    return out

# =========================
# Build HTF from 1m
# =========================
def build_htf_from_1m(df1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df1m is None or df1m.empty:
        return pd.DataFrame()
    d = df1m.copy().set_index("ts").sort_index()
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    else:
        d.index = d.index.tz_convert("UTC")

    rule = tf
    o = d["open"].resample(rule, label="left", closed="left").first()
    h = d["high"].resample(rule, label="left", closed="left").max()
    l = d["low"].resample(rule, label="left", closed="left").min()
    c = d["close"].resample(rule, label="left", closed="left").last()
    v = d["vol"].resample(rule, label="left", closed="left").sum()

    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ["open", "high", "low", "close", "vol"]
    out = out.dropna().reset_index()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out

# =========================
# Liquidation features + zones
# =========================
def liquidation_features(ldf: pd.DataFrame, lookback_min: int, zone_bins: int, top_k: int) -> Dict:
    out = {"liq_buy": 0.0, "liq_sell": 0.0, "zones": [], "liq_per_min": pd.DataFrame(), "n_recent": 0}
    if ldf is None or ldf.empty:
        return out

    cutoff = utc_now() - pd.Timedelta(minutes=lookback_min)
    recent = ldf[ldf["ts"] >= cutoff].copy()
    out["n_recent"] = int(len(recent))
    if recent.empty:
        return out

    out["liq_buy"] = float(recent.loc[recent["side"] == "BUY", "notional"].sum())
    out["liq_sell"] = float(recent.loc[recent["side"] == "SELL", "notional"].sum())

    recent["minute"] = recent["ts"].dt.floor("min")
    out["liq_per_min"] = recent.groupby("minute", as_index=False)["notional"].sum()

    zones: List[Dict] = []
    for side in ["BUY", "SELL"]:
        rs = recent[recent["side"] == side]
        if len(rs) < 3:
            continue
        prices = rs["price"].values
        w = rs["notional"].values
        pmin, pmax = float(np.min(prices)), float(np.max(prices))
        if not np.isfinite(pmin) or not np.isfinite(pmax) or pmin == pmax:
            continue
        hist, edges = np.histogram(prices, bins=zone_bins, range=(pmin, pmax), weights=w)
        idxs = np.argsort(hist)[::-1][:top_k]
        for i in idxs:
            if hist[i] <= 0:
                continue
            center = 0.5 * (edges[i] + edges[i + 1])
            zones.append({"price": float(center), "strength": float(hist[i]), "side": side})

    out["zones"] = sorted(zones, key=lambda z: z["strength"], reverse=True)[: top_k * 2]
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

def dynamic_liq_gate(liq_per_min: pd.DataFrame, fallback_gate: float) -> float:
    if liq_per_min is None or liq_per_min.empty:
        return float(fallback_gate)
    x = liq_per_min["notional"].tail(60).values
    if len(x) < 10:
        return float(fallback_gate)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-9
    return max(float(fallback_gate), med + 3.0 * mad)

# =========================
# HTF signal
# =========================
def signal_scored_htf(
    htf: pd.DataFrame,
    oi_hist: pd.DataFrame,
    liq: Dict,
    px: float,
    zone_near_bps: float,
    fallback_liq_gate: float,
    use_oi: bool,
    oi_weight: float,
) -> Dict:
    if htf is None or htf.empty or len(htf) < 80:
        return {"signal": "WAIT", "confidence": 0, "why": ["warmup_htf"], "plan": {}, "debug": {}}

    last = htf.iloc[-1]
    vol_z = float(last.get("vol_z", np.nan))
    macd_hist = float(last.get("macd_hist", np.nan))
    trend_slope = float(last.get("trend_slope", np.nan))
    rsi14 = float(last.get("rsi", np.nan))
    atrv = float(last.get("atr", np.nan))

    macd_std = (
        float(htf["macd_hist"].rolling(120).std(ddof=0).iloc[-1])
        if len(htf) >= 120
        else float(htf["macd_hist"].std(ddof=0))
    )
    macd_norm = float(np.clip(macd_hist / (macd_std + 1e-9), -3, 3))

    doi_usd = 0.0
    doi_usd_z = 0.0
    oi_ready = False
    if use_oi and oi_hist is not None and not oi_hist.empty:
        if "d_oi_usd" in oi_hist.columns:
            v = oi_hist["d_oi_usd"].iloc[-1]
            doi_usd = float(v) if np.isfinite(v) else 0.0
            oi_ready = True
        if "d_oi_usd_z" in oi_hist.columns:
            vz = oi_hist["d_oi_usd_z"].iloc[-1]
            doi_usd_z = float(vz) if np.isfinite(vz) else 0.0

    li_total = float(liq["liq_buy"] + liq["liq_sell"])
    gate = dynamic_liq_gate(liq.get("liq_per_min"), fallback_liq_gate)
    liq_hot = li_total >= gate

    zbest, zdist = nearest_zone(px, liq.get("zones", []))
    near_zone = (zdist is not None) and (zdist <= zone_near_bps)

    score_long = 0.0
    score_short = 0.0
    why: List[str] = []
    contrib = {"oi": (0.0, 0.0)}

    if liq_hot:
        score_long += 10
        score_short += 10
        why.append("liq_hot")

    if liq["liq_buy"] > liq["liq_sell"]:
        score_long += 8
        why.append("liq_buy>liq_sell")
    elif liq["liq_sell"] > liq["liq_buy"]:
        score_short += 8
        why.append("liq_sell>liq_buy")

    if macd_norm > 0.25 and trend_slope > 0:
        score_long += 14
        why.append("HTF_mom_up")
    if macd_norm < -0.25 and trend_slope < 0:
        score_short += 14
        why.append("HTF_mom_dn")

    if np.isfinite(vol_z) and vol_z > 0.25:
        score_long += 6
        score_short += 6
        why.append("HTF_vol_impulse")

    oi_dl = oi_ds = 0.0
    if use_oi and oi_ready:
        if doi_usd > 0:
            oi_dl += 6.0 * oi_weight
            oi_ds += 4.0 * oi_weight
            why.append("OI_adding")
        elif doi_usd < 0:
            oi_dl += 4.0 * oi_weight
            oi_ds += 6.0 * oi_weight
            why.append("OI_flushing")
        if abs(doi_usd_z) > 1.0:
            oi_dl += 4.0 * oi_weight
            oi_ds += 4.0 * oi_weight
            why.append("dOI_z_high")

    score_long += oi_dl
    score_short += oi_ds
    contrib["oi"] = (oi_dl, oi_ds)

    if near_zone and liq_hot and abs(macd_norm) < 0.35:
        if zbest and zbest["side"] == "BUY" and np.isfinite(rsi14) and rsi14 > 55:
            score_short += 8
            why.append("near_buy_zone_exhaust")
        if zbest and zbest["side"] == "SELL" and np.isfinite(rsi14) and rsi14 < 45:
            score_long += 8
            why.append("near_sell_zone_exhaust")

    edge = score_long - score_short
    signal = "LONG" if edge > 10 else ("SHORT" if edge < -10 else "WAIT")

    conf = float(np.clip(50 + 3.0 * abs(edge), 0, 100))
    conf = int(round(conf)) if signal != "WAIT" else int(round(np.clip(conf - 20, 0, 100)))

    plan = {}
    if signal in ("LONG", "SHORT") and np.isfinite(atrv) and atrv > 0:
        if signal == "LONG":
            plan = {"entry": px, "stop": px - 2.0 * atrv, "tp": px + 3.2 * atrv, "time_stop_min": 360}
        else:
            plan = {"entry": px, "stop": px + 2.0 * atrv, "tp": px - 3.2 * atrv, "time_stop_min": 360}

    return {
        "signal": signal,
        "confidence": conf,
        "why": why[:12],
        "plan": plan,
        "debug": {
            "edge": edge,
            "score_long": score_long,
            "score_short": score_short,
            "macd_norm": macd_norm,
            "trend_slope": trend_slope,
            "vol_z": vol_z,
            "liq_total": li_total,
            "liq_gate": gate,
            "liq_hot": liq_hot,
            "near_zone": near_zone,
            "zone_dist_bps": zdist,
            "zone_side": (zbest["side"] if zbest else None),
            "zone_px": (zbest["price"] if zbest else None),
            "doi_usd": doi_usd,
            "doi_usd_z": doi_usd_z,
            "contrib": contrib,
            "htf_last_bar_ts": str(htf["ts"].iloc[-1]),
        },
    }

# =========================
# Altair charts
# =========================
def alt_candles(df: pd.DataFrame, zones: List[Dict], title: str, window: int = 300):
    d = df.tail(window).copy()
    if d.empty:
        st.info("No candle data yet.")
        return

    y_min = float(d["low"].min())
    y_max = float(d["high"].max())
    if zones:
        z_prices = [z["price"] for z in zones if "price" in z]
        if z_prices:
            y_min = min(y_min, float(min(z_prices)))
            y_max = max(y_max, float(max(z_prices)))

    pad = (y_max - y_min) * 0.02 if y_max > y_min else (y_max * 0.01 if y_max else 1.0)
    y_domain = [y_min - pad, y_max + pad]

    d["up"] = d["close"] >= d["open"]

    base = alt.Chart(d).encode(
        x=alt.X("ts:T", axis=alt.Axis(title="Time", labelAngle=-45))
    )

    wicks = base.mark_rule().encode(
        y=alt.Y("low:Q", scale=alt.Scale(domain=y_domain), axis=alt.Axis(title="Price")),
        y2="high:Q",
        tooltip=["ts:T", "open:Q", "high:Q", "low:Q", "close:Q", "vol:Q"],
    )

    bodies = base.mark_bar().encode(
        y=alt.Y("open:Q", scale=alt.Scale(domain=y_domain)),
        y2="close:Q",
        color=alt.condition(
            alt.datum.up,
            alt.value("green"),
            alt.value("red"),
        ),
        tooltip=["ts:T", "open:Q", "high:Q", "low:Q", "close:Q", "vol:Q"],
    )

    chart = (wicks + bodies).properties(title=title, height=420).interactive()

    if zones:
        zdf = pd.DataFrame(zones[:8]).copy()
        zdf["strength"] = pd.to_numeric(zdf["strength"], errors="coerce").fillna(0.0)

        z_rules = alt.Chart(zdf).mark_rule(strokeWidth=2, opacity=0.85).encode(
            y=alt.Y("price:Q", scale=alt.Scale(domain=y_domain)),
            strokeDash=alt.StrokeDash("side:N"),
            tooltip=["side:N", alt.Tooltip("price:Q", format=",.2f"), alt.Tooltip("strength:Q", format=",.0f")],
        )

        x_max = d["ts"].max()
        zdf["x_label"] = x_max
        z_text = alt.Chart(zdf).mark_text(align="left", dx=6, dy=-6).encode(
            x=alt.X("x_label:T"),
            y=alt.Y("price:Q", scale=alt.Scale(domain=y_domain)),
            text=alt.Text("side:N"),
            tooltip=["side:N", alt.Tooltip("price:Q", format=",.2f"), alt.Tooltip("strength:Q", format=",.0f")],
        )

        chart = chart + z_rules + z_text

    st.altair_chart(chart, use_container_width=True)


def alt_lines(df: pd.DataFrame, x_col: str, series: Dict[str, str], title: str, height: int = 260):
    if df is None or df.empty:
        st.info("No data yet.")
        return

    cols = [c for c in series.values() if c in df.columns]
    if not cols:
        st.info("Series not ready.")
        return

    tmp = df[[x_col] + cols].copy().melt(id_vars=[x_col], var_name="series", value_name="value")
    inv = {v: k for k, v in series.items()}
    tmp["series"] = tmp["series"].map(inv).fillna(tmp["series"])

    base = alt.Chart(tmp).encode(
        x=alt.X(f"{x_col}:T", axis=alt.Axis(title="Time", labelAngle=-45)),
        y=alt.Y("value:Q", axis=alt.Axis(title="Value")),
        detail="series:N",
        tooltip=[alt.Tooltip(f"{x_col}:T"), "series:N", alt.Tooltip("value:Q")],
    )

    pos = base.transform_filter(alt.datum.value >= 0).mark_line().encode(color=alt.value("green"))
    neg = base.transform_filter(alt.datum.value < 0).mark_line().encode(color=alt.value("red"))

    chart = (pos + neg).properties(title=title, height=height).interactive()
    st.altair_chart(chart, use_container_width=True)


def alt_liq_bars(ldf: pd.DataFrame, title: str, window_min: int = 240):
    """
    Net liquidation flow per minute:
      BUY  -> +notional
      SELL -> -notional
    Colors:
      >=0  -> green
      <0   -> red
    """
    if ldf is None or ldf.empty:
        st.info("No liquidations yet.")
        return

    cutoff = utc_now() - pd.Timedelta(minutes=window_min)
    tmp = ldf[ldf["ts"] >= cutoff].copy()
    if tmp.empty:
        st.info("No liquidations in window.")
        return

    tmp["minute"] = tmp["ts"].dt.floor("min")
    tmp["signed_notional"] = np.where(tmp["side"].astype(str).str.upper() == "BUY", tmp["notional"], -tmp["notional"])

    g = tmp.groupby("minute", as_index=False)["signed_notional"].sum()

    chart = (
        alt.Chart(g)
        .mark_bar()
        .encode(
            x=alt.X("minute:T", axis=alt.Axis(title="Time", labelAngle=-45)),
            y=alt.Y("signed_notional:Q", axis=alt.Axis(title="Net liq notional (BUY + / SELL -)")),
            color=alt.condition(
                alt.datum.signed_notional >= 0,
                alt.value("green"),
                alt.value("red"),
            ),
            tooltip=[
                alt.Tooltip("minute:T"),
                alt.Tooltip("signed_notional:Q", format=",.0f", title="net_notional"),
            ],
        )
        .properties(title=title, height=260)
        .interactive()
    )

    # Add a 0-line for readability
    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeWidth=1, opacity=0.6).encode(y="y:Q")

    st.altair_chart(chart + zero_line, use_container_width=True)


# =========================
# Snapshot
# =========================
def snapshot(state: SharedState):
    with state.lock:
        c1 = pd.DataFrame(list(state.candles_1m))
        mp = state.mark_price

        ldf = pd.DataFrame(
            [
                {
                    "ts": pd.to_datetime(e.ts, unit="ms", utc=True),
                    "side": e.side,
                    "price": e.price,
                    "qty": e.qty,
                    "notional": e.notional,
                }
                for e in list(state.liqs)
            ]
        )

        oi_hist = state.oi_hist.copy() if state.oi_hist is not None else pd.DataFrame()

        meta = {
            "ws_last_msg_ms": state.ws_last_msg_ms,
            "ws_errors": list(state.ws_errors)[-10:],
            "oi_last_refresh_ms": state.oi_last_refresh_ms,
            "oi_errors": list(state.oi_errors)[-8:],
            "oi_last_status": state.oi_last_status,
            "oi_last_body": state.oi_last_body,
            "oi_thread_running": state.oi_thread_running,
            "backfill_minutes": state.backfill_minutes,
            "backfill_error": state.backfill_error,
            "htf_seed_meta": dict(state.htf_seed_meta),
            "htf_backfill_error": state.htf_backfill_error,
        }

    if not c1.empty:
        c1["ts"] = pd.to_datetime(c1["ts"], unit="ms", utc=True)
        c1 = c1.sort_values("ts").reset_index(drop=True)
    if not ldf.empty:
        ldf = ldf.sort_values("ts").reset_index(drop=True)

    df1 = indicators_ohlcv(c1) if not c1.empty else pd.DataFrame()
    return df1, ldf, mp, oi_hist, meta

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="ETHUSDT 1m Feed + HTF Signals", layout="wide")
state = get_state()

with st.sidebar:
    st.header("UI refresh")
    auto_refresh = st.checkbox("Auto-refresh UI", value=True)
    refresh_ms = st.slider("Refresh interval (ms)", 200, 5000, 750, 50)

    st.divider()
    st.header("Startup backfill")
    backfill_min = st.slider("Backfill minutes (1m)", 120, 1500, 360, 30)

    st.divider()
    st.header("HTF seed backfill")
    seed_days = st.slider("Seed days (HTF)", 1, 7, 3, 1)
    seed_all_tfs = st.checkbox("Seed all HTFs (15m/30m/1h)", value=True)

    st.divider()
    st.header("OI history (Binance only)")
    oi_period = st.selectbox("OI period", ["5m", "15m", "30m", "1h"], index=0)
    oi_limit = st.slider("OI points", 60, 500, 200, 10)
    oi_refresh_secs = st.slider("OI refresh (sec)", 5, 120, 15, 5)

    st.divider()
    st.header("Liquidation zones")
    liq_lookback_min = st.slider("Liquidation lookback (min)", 30, 240, 120, 10)
    zone_bins = st.slider("Zone bins", 20, 180, 70, 10)
    top_k = st.slider("Top zones per side", 1, 8, 3, 1)
    zone_near_bps = st.slider("Zone proximity (bps)", 5, 120, 30, 5)

    st.divider()
    st.header("Signal timeframe")
    signal_tf = st.selectbox("Compute signals on", ["15m", "30m", "1h"], index=0)
    use_only_closed_htf = st.checkbox("Use only CLOSED HTF bars (more stable)", value=True)
    htf_window_bars = st.slider("HTF history bars", 120, 2000, 600, 60)

    st.divider()
    st.header("Signal weights")
    fallback_liq_gate = st.number_input("Fallback liq gate (USD)", 10_000, 50_000_000, 250_000, 10_000)
    use_oi = st.checkbox("Use OI in signal", value=True)
    oi_weight = st.slider("OI weight", 0.0, 2.0, 1.0, 0.1)

    st.divider()
    st.header("Sizing (as requested)")
    equity = st.number_input("Account equity (USD)", min_value=1.0, value=100.0, step=10.0)
    leverage = st.number_input("Leverage", min_value=1, max_value=200, value=150, step=1)
    margin_frac = st.slider("Margin fraction / trade", 0.01, 0.95, 0.40, 0.01)

st.title("ETHUSDT — 1m Feed + Liquidations + OI + HTF Signals")

# ✅ Backfill 1m candles so charts + resample work immediately
ensure_backfill_1m(state, int(backfill_min))

# ✅ Seed HTF history (REST) BEFORE WS starts (2–3 days recommended)
if seed_all_tfs:
    for tf in ("15m", "30m", "1h"):
        ensure_htf_seed_backfill_days(state, tf, days=int(seed_days))
else:
    ensure_htf_seed_backfill_days(state, signal_tf, days=int(seed_days))

# Thread control
if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = threading.Event()

# Start WS thread once
if "ws_thread" not in st.session_state:
    st.session_state.ws_thread = threading.Thread(
        target=ws_worker, args=(state, SYMBOL, st.session_state.stop_flag), daemon=True
    )
    st.session_state.ws_thread.start()

# Start OI thread once per config
if "oi_cfg" not in st.session_state:
    st.session_state.oi_cfg = None
    st.session_state.oi_thread = None

cfg = (oi_period, int(oi_limit), int(oi_refresh_secs))
need_new = (st.session_state.oi_cfg != cfg) and (
    st.session_state.oi_thread is None or not st.session_state.oi_thread.is_alive()
)
if need_new:
    st.session_state.oi_cfg = cfg
    st.session_state.oi_thread = threading.Thread(
        target=oi_hist_refresher,
        args=(state, SYMBOL, oi_period, int(oi_limit), int(oi_refresh_secs), st.session_state.stop_flag),
        daemon=True,
    )
    st.session_state.oi_thread.start()

# Snapshot current data
df1, ldf, mp, oi_hist, meta = snapshot(state)

px_now = float(mp) if (mp is not None and np.isfinite(mp)) else (float(df1["close"].iloc[-1]) if not df1.empty else np.nan)

# Liquidation features (zones)
liq = liquidation_features(ldf, lookback_min=liq_lookback_min, zone_bins=zone_bins, top_k=top_k)

# -------------------------
# ✅ Build HTF bars for signal
# Merge: REST seed + live resample from 1m
# -------------------------
htf_from_1m = build_htf_from_1m(df1[["ts", "open", "high", "low", "close", "vol"]].copy(), signal_tf) if not df1.empty else pd.DataFrame()

with state.lock:
    seed = state.htf_seed.get(signal_tf, pd.DataFrame()).copy()
    seed_meta = state.htf_seed_meta.get(signal_tf, {})
    seed_err = state.htf_backfill_error

if seed is not None and not seed.empty:
    htf_raw = (
        pd.concat([seed, htf_from_1m], ignore_index=True)
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )
else:
    htf_raw = htf_from_1m

htf_raw = htf_raw.tail(int(htf_window_bars)).reset_index(drop=True)
htf = indicators_ohlcv(htf_raw) if not htf_raw.empty else pd.DataFrame()

if use_only_closed_htf and not htf.empty:
    tf_delta = pd.Timedelta(signal_tf)
    last_open = htf["ts"].iloc[-1]
    if utc_now() < (last_open + tf_delta):
        htf = htf.iloc[:-1].copy()

# Compute HTF-based signal
sig = signal_scored_htf(
    htf=htf,
    oi_hist=oi_hist,
    liq=liq,
    px=px_now if np.isfinite(px_now) else 0.0,
    zone_near_bps=zone_near_bps,
    fallback_liq_gate=float(fallback_liq_gate),
    use_oi=use_oi,
    oi_weight=float(oi_weight),
)

# sizing summary
margin_used = equity * margin_frac
notional = margin_used * leverage
rough_wipe_pct = (1.0 / leverage) * 100.0

# =========================
# Header panels
# =========================
col1, col2, col3 = st.columns([1.25, 1.0, 1.0])

with col1:
    st.subheader("Signal (HTF)")
    st.metric("Action", sig["signal"])
    st.metric("Confidence", f"{sig['confidence']}/100")
    st.caption("Why: " + ", ".join(sig.get("why", [])[:10]) if sig.get("why") else "Why: —")
    if sig.get("plan"):
        st.write("Plan:", sig["plan"])

    if not htf_raw.empty:
        tf_delta = pd.Timedelta(signal_tf)
        last_open = htf_raw["ts"].iloc[-1]
        next_close = last_open + tf_delta
        secs = max(0, int((next_close - utc_now()).total_seconds()))
        st.caption(f"Signal TF: {signal_tf} | Next close in ~{secs//60}m {secs%60}s | Closed-only: {use_only_closed_htf}")

    st.caption(
        f"Mark/Last: {px_now:.2f} | 1m bars: {len(df1)} | liqs total: {len(ldf)} | liqs lookback: {liq.get('n_recent', 0)} "
        f"| zones: {len(liq.get('zones', []))} | OI rows: {len(oi_hist)} | HTF bars: {len(htf)}"
    )

with col2:
    st.subheader("Flow")
    st.metric("Mark Price", f"{px_now} USDT")
    st.metric("Liq BUY", f"{liq['liq_buy']:,.0f} USD")
    st.metric("Liq SELL", f"{liq['liq_sell']:,.0f} USD")
    if not oi_hist.empty:
        st.metric("OI USD (latest)", f"{oi_hist['oi_usd'].iloc[-1]:,.0f}")
        st.metric("ΔOI USD (latest)", f"{oi_hist['d_oi_usd'].iloc[-1]:,.0f}")
    else:
        st.warning("OI history empty — open diagnostics to see HTTP status/body.")

with col3:
    st.subheader("Sizing (150x / 40%)")
    st.metric("Margin used", f"${margin_used:,.2f}")
    st.metric("Notional exposure", f"${notional:,.2f}")
    st.caption(f"Rough: ~{rough_wipe_pct:.2f}% adverse move ≈ 100% margin at {leverage}x (ignores fees/maint.).")

st.divider()

# =========================
# Diagnostics
# =========================
with st.expander("Health / diagnostics", expanded=False):
    ws_age = (now_ms() - meta["ws_last_msg_ms"]) if meta["ws_last_msg_ms"] else None
    oi_age = (now_ms() - meta["oi_last_refresh_ms"]) if meta["oi_last_refresh_ms"] else None

    st.write("WS thread alive:", st.session_state.ws_thread.is_alive() if "ws_thread" in st.session_state else None)
    st.write("OI thread alive:", st.session_state.oi_thread.is_alive() if st.session_state.oi_thread else None)

    st.write(f"1m backfill minutes: {meta['backfill_minutes']} | error: {meta['backfill_error']}")
    st.write(f"WS last msg age (ms): {ws_age if ws_age is not None else '—'}")
    st.write(f"OI last refresh age (ms): {oi_age if oi_age is not None else '—'}")
    st.write("OI thread_running flag:", meta["oi_thread_running"])
    st.write("Last OI HTTP status:", meta["oi_last_status"])
    st.write("Last OI body snippet:", meta["oi_last_body"])

    st.write("HTF seed meta (all):", meta.get("htf_seed_meta", {}))
    st.write("HTF seed error:", meta.get("htf_backfill_error", None))
    st.write("Selected TF seed meta:", seed_meta)
    st.write("Selected TF seed error:", seed_err)

    if meta["ws_errors"]:
        st.error("Recent WS errors:\n" + "\n".join(meta["ws_errors"]))
    if meta["oi_errors"]:
        st.error("Recent OI errors:\n" + "\n".join(meta["oi_errors"]))

    if st.button("Probe openInterestHist now (sync)"):
        try:
            df_probe = fetch_open_interest_hist(state, SYMBOL, period=oi_period, limit=int(oi_limit))
            st.success(f"Probe OK: rows={len(df_probe)} latest_oi_usd={df_probe['oi_usd'].iloc[-1]:,.0f}")
        except Exception as e:
            st.error(f"Probe FAILED: {repr(e)}")

# =========================
# Charts
# =========================
if not df1.empty:
    alt_candles(df1, liq.get("zones", []), "ETHUSDT 1m Candles + Liquidation Zones", window=300)

if not htf_raw.empty:
    alt_candles(htf_raw, [], f"ETHUSDT {signal_tf} Candles (Signal TF)", window=220)

cA, cB = st.columns(2)
with cA:
    if not df1.empty:
        alt_lines(df1.tail(1200), "ts", {"MACD hist (1m)": "macd_hist", "RSI (1m)": "rsi"}, "Momentum (1m)", height=260)
with cB:
    if not htf.empty:
        alt_lines(htf.tail(800), "ts", {f"MACD hist ({signal_tf})": "macd_hist", f"RSI ({signal_tf})": "rsi"}, f"Momentum ({signal_tf})", height=260)
    else:
        st.info("HTF indicators warming up… (seed should prevent this unless REST failed)")

cC, cD = st.columns(2)
with cC:
    alt_liq_bars(ldf, "Liquidation Notional per Minute", window_min=240)
with cD:
    if not oi_hist.empty:
        alt_lines(oi_hist.tail(300), "ts", {"OI USD": "oi_usd", "ΔOI USD": "d_oi_usd"}, "Open Interest (openInterestHist)", height=260)
    else:
        st.info("OI chart waiting — check diagnostics for HTTP status/body.")

st.subheader("Signal debug")
st.json(sig.get("debug", {}))

# =========================
# Auto-refresh heartbeat
# =========================
if auto_refresh:
    time.sleep(refresh_ms / 1000.0)
    st.rerun()
