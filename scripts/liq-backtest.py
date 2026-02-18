# backtest_eth_htf15_liq_oi_grid.py
# ETHUSDT futures backtest + grid search (HTF=15m signals, LTF=1m execution)
#
# ✅ PATCHED (per your request):
# - Backtest window set to ~3 months (90 days) of 1m data + 15m warmup
# - Squeeze-continuation signal using:
#     * net liquidation z-score bursts (BUY +, SELL -)
#     * reclaim vs EMA50 + MACD_hist slope
#     * OI flush (ΔOI USD < 0) as soft requirement
# - Entry on 15m boundaries + persistence (configurable)
# - Parametric proxy liquidation builder (if LIQ_MODE="proxy") so grid can tune it
# - Full grid search + scoring + min-trade constraint
# - Writes: equity_curve.csv, trades.csv (for the *last run*), grid_results.csv (for the grid)
#
# Run:
#   pip install pandas numpy requests
#   python backtest_eth_htf15_liq_oi_grid.py
#
# Notes:
# - 90 days of 1m candles is large (~129,600 rows). REST paging is implemented with gentle pacing.
# - Best practice: use real forceOrder liquidation data (LIQ_MODE="file") if you have it.

import math
import time
from dataclasses import dataclass
from itertools import product
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import requests

FAPI_REST = "https://fapi.binance.com"
SYMBOL = "ETHUSDT"

# -----------------------------
# ✅ Backtest window (3 months)
# -----------------------------
DAYS_1M = 90
SEED_DAYS_15M = 14  # indicator warmup on 15m

# -----------------------------
# Default risk/costs (can be overridden by grid)
# -----------------------------
TAKER_FEE = 0.0004
LEVERAGE = 150
MARGIN_FRAC = 0.40
LIQ_MOVE = 1.0 / LEVERAGE

# Exits (hours, not days) (can be overridden by grid)
SL_ATR_MULT = 0.7
TP_ATR_MULT = 0.9
TIME_STOP_MIN = 240
MIN_HOLD_MIN = 0  # keep 0 for simplicity; you can add hold constraints later
COOLDOWN_AFTER_STOP_MIN = 20

# Risk floor
EQUITY_FLOOR = 10.0

# -----------------------------
# Signal logic defaults (can be overridden by grid)
# -----------------------------
LIQ_Z_WIN_MIN = 90
LIQ_Z_THRESH = 1.5
LIQ_Z_ROLL_MIN = 120
EDGE_PERSIST_BARS = 1
STRONG_BURST_Z = 2.2  # “very strong burst” threshold

# -----------------------------
# OI
# -----------------------------
OI_PERIOD = "5m"
OI_LIMIT = 500

# -----------------------------
# Liquidations source
# -----------------------------
LIQ_MODE = "proxy"          # "proxy" or "file"
LIQ_FILE = "liq_forceorder_ethusdt.csv"

# -----------------------------
# REST helpers
# -----------------------------
def rest_get_json(path: str, params: dict, timeout=20, max_tries=10):
    url = FAPI_REST + path
    headers = {"User-Agent": "Mozilla/5.0"}
    backoff = 1.0
    for _ in range(max_tries):
        r = requests.get(url, params=params, timeout=timeout, headers=headers)
        if r.status_code == 429:
            time.sleep(backoff)
            backoff = min(backoff * 1.7, 25.0)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Rate limited too long (429)")

def utc_ms(ts: pd.Timestamp) -> int:
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.value // 10**6)

def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")

def interval_ms(interval: str) -> int:
    interval = interval.strip().lower()
    if interval.endswith("m"):
        return int(interval[:-1]) * 60_000
    if interval.endswith("h"):
        return int(interval[:-1]) * 60 * 60_000
    raise ValueError(f"Unsupported interval: {interval}")

def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Paged fetch using startTime/limit. Each call returns up to 1500 bars.
    """
    out = []
    cur = int(start_ms)
    step = interval_ms(interval)
    calls = 0

    while cur < end_ms:
        data = rest_get_json("/fapi/v1/klines", {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1500
        })
        calls += 1
        if not data:
            break

        out.extend(data)
        last_open = int(data[-1][0])
        cur = last_open + step

        if len(data) < 1500:
            break

        # gentle pacing to reduce 429
        if calls % 8 == 0:
            time.sleep(0.25)
        else:
            time.sleep(0.06)

    df = pd.DataFrame(out, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base","taker_quote","ignore"
    ])
    if df.empty:
        return pd.DataFrame(columns=["ts","open","high","low","close","vol"])

    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["ts","open","high","low","close","volume"]].rename(columns={"volume":"vol"})
    df = df.dropna().sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    return df

def fetch_oi_hist(symbol: str, period: str, start_ms: int, end_ms: int, limit: int = 500) -> pd.DataFrame:
    data = rest_get_json("/futures/data/openInterestHist", {
        "symbol": symbol,
        "period": period,
        "limit": limit,
        "startTime": start_ms,
        "endTime": end_ms,
    })
    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame(columns=["ts","oi","oi_usd","d_oi_usd","d_oi_usd_z"])

    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["oi"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    df["oi_usd"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
    df = df[["ts","oi","oi_usd"]].dropna().sort_values("ts").reset_index(drop=True)
    df["d_oi_usd"] = df["oi_usd"].diff()

    roll = 60
    mu = df["d_oi_usd"].fillna(0.0).rolling(roll).mean()
    sd = df["d_oi_usd"].fillna(0.0).rolling(roll).std(ddof=0).replace(0, np.nan)
    df["d_oi_usd_z"] = (df["d_oi_usd"].fillna(0.0) - mu) / sd
    return df

# -----------------------------
# Indicators
# -----------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_dn.replace(0, np.nan))
    return 100 - (100/(1+rs))

def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev = c.shift(1)
    return pd.concat([(h-l).abs(), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)

def atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(h,l,c)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (s-m)/sd.replace(0, np.nan)

def indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["close"], 12)
    out["ema_slow"] = ema(out["close"], 26)
    out["ema_trend"] = ema(out["close"], 200)
    out["ema_50"] = ema(out["close"], 50)
    out["macd"] = out["ema_fast"] - out["ema_slow"]
    out["macd_sig"] = ema(out["macd"], 9)
    out["macd_hist"] = out["macd"] - out["macd_sig"]
    out["rsi"] = rsi(out["close"], 14)
    out["vol_z"] = zscore(out["vol"], 120)
    out["atr"] = atr(out["high"], out["low"], out["close"], 14)
    out["trend_slope"] = out["ema_trend"].diff(30) / out["ema_trend"].shift(30)
    return out

# -----------------------------
# Liquidations
# -----------------------------
def load_liqs_from_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ts" not in df.columns:
        raise ValueError("liq file must have column 'ts'")
    if np.issubdtype(df["ts"].dtype, np.number):
        df["ts"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms", utc=True)
    else:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["side"] = df["side"].astype(str).str.upper()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["notional"] = df["price"] * df["qty"]
    df = df.dropna(subset=["ts","side","price","qty","notional"]).sort_values("ts").reset_index(drop=True)
    return df[["ts","side","price","qty","notional"]]

def proxy_liqs_from_1m_parametric(df1m: pd.DataFrame, ret_q: float, vol_q: float, wick_q: float) -> pd.DataFrame:
    d = df1m.copy()
    d["ret_1m"] = d["close"].pct_change()
    d["wick_up"] = (d["high"] - d[["open","close"]].max(axis=1)) / d["close"]
    d["wick_dn"] = (d[["open","close"]].min(axis=1) - d["low"]) / d["close"]

    gate_ret = d["ret_1m"].abs().rolling(240).quantile(ret_q)
    gate_vol = d["vol"].rolling(240).quantile(vol_q)
    gate_wu  = d["wick_up"].rolling(240).quantile(wick_q)
    gate_wd  = d["wick_dn"].rolling(240).quantile(wick_q)

    cond = (d["vol"] > gate_vol) & (
        (d["ret_1m"].abs() > gate_ret) |
        (d["wick_up"] > gate_wu) |
        (d["wick_dn"] > gate_wd)
    )
    spikes = d[cond].copy()
    if spikes.empty:
        return pd.DataFrame(columns=["ts","side","price","qty","notional"])

    events = []
    for _, r in spikes.iterrows():
        notional = float(r["close"] * r["vol"])
        if r["ret_1m"] < 0 or r["wick_dn"] > r["wick_up"]:
            side = "SELL"
            price = float(r["low"])
        else:
            side = "BUY"
            price = float(r["high"])
        qty = max(1e-9, notional / max(price, 1e-9))
        events.append({"ts": r["ts"], "side": side, "price": price, "qty": qty, "notional": notional})

    return pd.DataFrame(events).sort_values("ts").reset_index(drop=True)

# -----------------------------
# Net liquidation z-score
# -----------------------------
def build_liq_min_z(liqs: pd.DataFrame) -> pd.DataFrame:
    if liqs is None or liqs.empty:
        return pd.DataFrame(columns=["minute","signed_notional","liq_z"]).set_index("minute")

    df = liqs.copy()
    df["minute"] = df["ts"].dt.floor("min")
    df["signed_notional"] = np.where(df["side"].astype(str).str.upper() == "BUY", df["notional"], -df["notional"])
    g = df.groupby("minute", as_index=False)["signed_notional"].sum().sort_values("minute").reset_index(drop=True)

    roll = int(LIQ_Z_ROLL_MIN)
    mu = g["signed_notional"].rolling(roll).mean()
    sd = g["signed_notional"].rolling(roll).std(ddof=0).replace(0, np.nan)
    g["liq_z"] = (g["signed_notional"] - mu) / sd
    return g.set_index("minute")

def liq_window_extrema(liq_min_z: pd.DataFrame, t: pd.Timestamp, win_min: int) -> Tuple[float, float]:
    if liq_min_z is None or liq_min_z.empty:
        return np.nan, np.nan
    end = t.floor("min")
    start = end - pd.Timedelta(minutes=win_min)
    w = liq_min_z.loc[(liq_min_z.index >= start) & (liq_min_z.index <= end)]
    if w.empty:
        return np.nan, np.nan
    return float(w["liq_z"].min()), float(w["liq_z"].max())

# -----------------------------
# Signal: relaxed squeeze-continuation (15m)
# -----------------------------
def htf_signal_continuation(
    htf: pd.DataFrame,
    oi_row: Optional[pd.Series],
    liq_min_z: pd.DataFrame,
    t: pd.Timestamp
) -> str:
    if htf is None or len(htf) < 3:
        return "WAIT"

    last = htf.iloc[-1]
    prev = htf.iloc[-2]

    # OI flush (soft)
    d_oi = np.nan
    if oi_row is not None:
        v = oi_row.get("d_oi_usd", np.nan)
        d_oi = float(v) if np.isfinite(v) else np.nan
    oi_flush = np.isfinite(d_oi) and (d_oi < 0)

    min_z, max_z = liq_window_extrema(liq_min_z, t, LIQ_Z_WIN_MIN)

    mom_up = (last["macd_hist"] > prev["macd_hist"])
    mom_dn = (last["macd_hist"] < prev["macd_hist"])
    reclaim_up = (last["close"] > last["ema_50"])
    reclaim_dn = (last["close"] < last["ema_50"])

    burst_dn = np.isfinite(min_z) and (min_z < -LIQ_Z_THRESH)
    burst_up = np.isfinite(max_z) and (max_z >  LIQ_Z_THRESH)

    # relaxed: (burst + reclaim + momentum) + (oi_flush OR very strong burst)
    long_ok = burst_dn and reclaim_up and mom_up and (oi_flush or (np.isfinite(min_z) and (min_z < -STRONG_BURST_Z)))
    short_ok = burst_up and reclaim_dn and mom_dn and (oi_flush or (np.isfinite(max_z) and (max_z >  STRONG_BURST_Z)))

    if long_ok:
        return "LONG"
    if short_ok:
        return "SHORT"
    return "WAIT"

# -----------------------------
# Backtest engine
# -----------------------------
@dataclass
class Trade:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    side: str
    entry_px: float
    exit_px: float
    pnl: float
    ret: float
    reason: str
    hold_min: int
    equity_before: float
    equity_after: float

def run_backtest(
    df1m: pd.DataFrame,
    df15m: pd.DataFrame,
    oi: pd.DataFrame,
    liq_min_z: pd.DataFrame,
    equity0: float = 10_000.0,
    warmup_15m: int = 120,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    global LIQ_MOVE

    equity = float(equity0)
    pos_side = None
    entry_px = None
    entry_ts = None
    entry_notional = None
    entry_fee = 0.0
    entry_margin = None
    cooldown_until = None

    trades: List[Trade] = []
    eq = np.full(len(df1m), np.nan, dtype=float)
    eq[0] = equity

    # Map latest 15m <= 1m
    htf_map = df15m.set_index("ts").sort_index()
    htf_idx = np.searchsorted(htf_map.index.values, df1m["ts"].values, side="right") - 1

    # OI to 1m
    oi_map = oi.set_index("ts").sort_index().reindex(df1m["ts"]).ffill()

    def is_15m_boundary(ts: pd.Timestamp) -> bool:
        return (ts.minute % 15 == 0)

    def persistent_signal(htf_slice: pd.DataFrame, t: pd.Timestamp, oi_row: Optional[pd.Series]) -> str:
        if EDGE_PERSIST_BARS <= 1:
            return htf_signal_continuation(htf_slice, oi_row, liq_min_z, t)

        if len(htf_slice) < (EDGE_PERSIST_BARS + 2):
            return "WAIT"

        sigs = []
        for k in range(EDGE_PERSIST_BARS):
            sub = htf_slice.iloc[:-(k)] if k > 0 else htf_slice
            sigs.append(htf_signal_continuation(sub, oi_row, liq_min_z, t - pd.Timedelta(minutes=15*k)))

        if all(s == sigs[0] for s in sigs) and sigs[0] in ("LONG","SHORT"):
            return sigs[0]
        return "WAIT"

    def close_trade(i: int, exit_px: float, reason: str):
        nonlocal equity, pos_side, entry_px, entry_ts, entry_notional, entry_fee, entry_margin, cooldown_until

        t = df1m["ts"].iloc[i]
        hold_min = int((t - entry_ts) / pd.Timedelta(minutes=1))
        equity_before = equity

        ret = (exit_px / entry_px - 1.0) if pos_side == "LONG" else (entry_px / exit_px - 1.0)
        pnl_gross = entry_notional * ret
        exit_fee = entry_notional * TAKER_FEE
        pnl = pnl_gross - entry_fee - exit_fee

        # cap to posted margin (simple)
        min_pnl = -entry_margin - entry_fee - exit_fee
        pnl = max(pnl, min_pnl)

        equity = max(equity + pnl, 0.0)

        trades.append(Trade(
            entry_ts=entry_ts, exit_ts=t,
            side=pos_side, entry_px=float(entry_px), exit_px=float(exit_px),
            pnl=float(pnl), ret=float(ret),
            reason=reason, hold_min=hold_min,
            equity_before=float(equity_before),
            equity_after=float(equity)
        ))

        if reason in ("SL","LIQ"):
            cooldown_until = t + pd.Timedelta(minutes=COOLDOWN_AFTER_STOP_MIN)

        pos_side = None
        entry_px = None
        entry_ts = None
        entry_notional = None
        entry_fee = 0.0
        entry_margin = None

    for i in range(1, len(df1m)):
        t = df1m["ts"].iloc[i]
        h = float(df1m["high"].iloc[i])
        l = float(df1m["low"].iloc[i])
        c = float(df1m["close"].iloc[i])

        if equity < EQUITY_FLOOR:
            eq[i] = equity
            continue

        hi = int(htf_idx[i])
        if hi < 0:
            eq[i] = equity
            continue

        htf_slice = htf_map.iloc[:hi+1].reset_index()
        if len(htf_slice) < warmup_15m:
            eq[i] = equity
            continue

        oi_row = oi_map.iloc[i] if len(oi_map) else None

        # Manage existing position
        if pos_side is not None:
            hold_min = int((t - entry_ts) / pd.Timedelta(minutes=1))

            # liquidation intrabar (simple)
            if pos_side == "LONG":
                liq_px = entry_px * (1.0 - LIQ_MOVE)
                if l <= liq_px:
                    close_trade(i, liq_px, "LIQ"); eq[i] = equity; continue
            else:
                liq_px = entry_px * (1.0 + LIQ_MOVE)
                if h >= liq_px:
                    close_trade(i, liq_px, "LIQ"); eq[i] = equity; continue

            atr15 = float(htf_slice["atr"].iloc[-1]) if np.isfinite(htf_slice["atr"].iloc[-1]) else np.nan
            if np.isfinite(atr15) and atr15 > 0:
                if pos_side == "LONG":
                    sl = entry_px - SL_ATR_MULT * atr15
                    tp = entry_px + TP_ATR_MULT * atr15
                    if l <= sl:
                        close_trade(i, sl, "SL"); eq[i] = equity; continue
                    if h >= tp:
                        close_trade(i, tp, "TP"); eq[i] = equity; continue
                else:
                    sl = entry_px + SL_ATR_MULT * atr15
                    tp = entry_px - TP_ATR_MULT * atr15
                    if h >= sl:
                        close_trade(i, sl, "SL"); eq[i] = equity; continue
                    if l <= tp:
                        close_trade(i, tp, "TP"); eq[i] = equity; continue

            if hold_min >= TIME_STOP_MIN:
                close_trade(i, c, "TIME"); eq[i] = equity; continue

        # Entry
        if pos_side is None and (cooldown_until is None or t >= cooldown_until):
            if is_15m_boundary(t):
                sig = persistent_signal(htf_slice, t, oi_row)
                if sig in ("LONG","SHORT"):
                    entry_margin = equity * MARGIN_FRAC
                    entry_notional = entry_margin * LEVERAGE
                    entry_fee = entry_notional * TAKER_FEE
                    pos_side = sig
                    entry_px = c
                    entry_ts = t

        eq[i] = equity

    # Force close
    if pos_side is not None and equity >= EQUITY_FLOOR:
        i = len(df1m) - 1
        close_trade(i, float(df1m["close"].iloc[i]), "EOD")
        eq[i] = equity

    curve = pd.DataFrame({"ts": df1m["ts"], "equity": pd.Series(eq).ffill().values})
    curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1.0

    trades_df = pd.DataFrame([t.__dict__ for t in trades])

    rets = curve["equity"].pct_change().fillna(0.0)
    sharpe = (rets.mean() / (rets.std(ddof=0) + 1e-12)) * math.sqrt(365 * 24 * 60)
    max_dd = float(curve["drawdown"].min())
    win_rate = float((trades_df["pnl"] > 0).mean()) if len(trades_df) else 0.0

    metrics = {
        "start_equity": float(equity0),
        "end_equity": float(curve["equity"].iloc[-1]),
        "net_pnl": float(curve["equity"].iloc[-1] - equity0),
        "return_pct": float(curve["equity"].iloc[-1] / equity0 - 1.0) if equity0 else np.nan,
        "trades": int(len(trades_df)),
        "win_rate": float(win_rate),
        "avg_hold_min": float(trades_df["hold_min"].mean()) if len(trades_df) else 0.0,
        "sharpe_1m_ann": float(sharpe),
        "max_drawdown": float(max_dd),
        "liq_exits": int((trades_df["reason"] == "LIQ").sum()) if len(trades_df) else 0,
        "sl_exits": int((trades_df["reason"] == "SL").sum()) if len(trades_df) else 0,
        "tp_exits": int((trades_df["reason"] == "TP").sum()) if len(trades_df) else 0,
        "flip_exits": int((trades_df["reason"] == "FLIP").sum()) if len(trades_df) else 0,
        "time_exits": int((trades_df["reason"] == "TIME").sum()) if len(trades_df) else 0,
    }
    return curve, trades_df, metrics

# -----------------------------
# Grid-run wrapper
# -----------------------------
def run_one_config(
    df1m: pd.DataFrame,
    df15m_ind: pd.DataFrame,
    oi: pd.DataFrame,
    liqs_base: pd.DataFrame,
    # signal params
    liq_z_thresh: float,
    liq_z_win_min: int,
    liq_z_roll_min: int,
    strong_burst_z: float,
    edge_persist_bars: int,
    # exit params
    sl_atr_mult: float,
    tp_atr_mult: float,
    time_stop_min: int,
    cooldown_min: int,
    # risk params
    leverage: int,
    margin_frac: float,
    taker_fee: float,
    # proxy params
    proxy_ret_q: float,
    proxy_vol_q: float,
    proxy_wick_q: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    global LIQ_Z_THRESH, LIQ_Z_WIN_MIN, LIQ_Z_ROLL_MIN, EDGE_PERSIST_BARS, STRONG_BURST_Z
    global SL_ATR_MULT, TP_ATR_MULT, TIME_STOP_MIN, COOLDOWN_AFTER_STOP_MIN
    global LEVERAGE, MARGIN_FRAC, TAKER_FEE, LIQ_MOVE

    LIQ_Z_THRESH = liq_z_thresh
    LIQ_Z_WIN_MIN = liq_z_win_min
    LIQ_Z_ROLL_MIN = liq_z_roll_min
    STRONG_BURST_Z = strong_burst_z
    EDGE_PERSIST_BARS = edge_persist_bars

    SL_ATR_MULT = sl_atr_mult
    TP_ATR_MULT = tp_atr_mult
    TIME_STOP_MIN = time_stop_min
    COOLDOWN_AFTER_STOP_MIN = cooldown_min

    LEVERAGE = int(leverage)
    MARGIN_FRAC = float(margin_frac)
    TAKER_FEE = float(taker_fee)
    LIQ_MOVE = 1.0 / max(1, LEVERAGE)

    # build liquidations for this config
    if LIQ_MODE == "proxy":
        liqs = proxy_liqs_from_1m_parametric(df1m, ret_q=proxy_ret_q, vol_q=proxy_vol_q, wick_q=proxy_wick_q)
    else:
        liqs = liqs_base

    liq_min_z = build_liq_min_z(liqs)

    # if z baseline is too weak, treat as no-trade
    if liq_min_z.empty or len(liq_min_z) < max(60, liq_z_roll_min // 2):
        curve = pd.DataFrame({"ts": df1m["ts"], "equity": np.nan})
        trades = pd.DataFrame()
        metrics = {
            "start_equity": 10_000.0, "end_equity": 10_000.0, "net_pnl": 0.0, "return_pct": 0.0,
            "trades": 0, "win_rate": 0.0, "avg_hold_min": 0.0, "sharpe_1m_ann": 0.0, "max_drawdown": 0.0,
            "liq_exits": 0, "sl_exits": 0, "tp_exits": 0, "flip_exits": 0, "time_exits": 0
        }
    else:
        curve, trades, metrics = run_backtest(
            df1m=df1m[["ts","open","high","low","close","vol"]],
            df15m=df15m_ind,
            oi=oi,
            liq_min_z=liq_min_z,
            equity0=10_000.0,
            warmup_15m=120
        )

    # attach config to metrics
    metrics = dict(metrics)
    metrics.update({
        "score": np.nan,
        "liq_z_thresh": liq_z_thresh,
        "liq_z_win_min": liq_z_win_min,
        "liq_z_roll_min": liq_z_roll_min,
        "strong_burst_z": strong_burst_z,
        "edge_persist_bars": edge_persist_bars,
        "sl_atr_mult": sl_atr_mult,
        "tp_atr_mult": tp_atr_mult,
        "time_stop_min": time_stop_min,
        "cooldown_min": cooldown_min,
        "leverage": leverage,
        "margin_frac": margin_frac,
        "taker_fee": taker_fee,
        "proxy_ret_q": proxy_ret_q,
        "proxy_vol_q": proxy_vol_q,
        "proxy_wick_q": proxy_wick_q,
        "liq_events": int(len(liqs)) if liqs is not None else 0,
        "liq_min_z_rows": int(len(liq_min_z)) if liq_min_z is not None else 0,
    })
    return curve, trades, metrics

def score_row(m: Dict) -> float:
    # Balanced score: return + sharpe - drawdown + mild trade encouragement + mild win-rate bonus
    return float(
        2.0 * m.get("return_pct", 0.0)
        + 0.4 * m.get("sharpe_1m_ann", 0.0)
        - 1.5 * abs(m.get("max_drawdown", 0.0))
        + 0.18 * np.log1p(max(0, int(m.get("trades", 0))))
        + 0.25 * (m.get("win_rate", 0.0) - 0.5)
    )

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    end = now_utc().floor("min")
    start_1m = end - pd.Timedelta(days=DAYS_1M)
    start_15m = end - pd.Timedelta(days=SEED_DAYS_15M)

    start_1m_ms = utc_ms(start_1m)
    start_15m_ms = utc_ms(start_15m)
    end_ms = utc_ms(end)

    print(f"Fetching {SYMBOL} 1m:  {start_1m} -> {end} ...")
    df1m = fetch_klines(SYMBOL, "1m", start_1m_ms, end_ms)
    print("1m bars:", len(df1m))

    print(f"Fetching {SYMBOL} 15m: {start_15m} -> {end} ...")
    df15m = fetch_klines(SYMBOL, "15m", start_15m_ms, end_ms)
    print("15m bars:", len(df15m))

    df15m_ind = indicators(df15m)

    print(f"Fetching OI hist ({OI_PERIOD}) ...")
    oi = fetch_oi_hist(SYMBOL, OI_PERIOD, start_1m_ms, end_ms, limit=OI_LIMIT)
    print("OI rows:", len(oi))

    # Base liquidations (for LIQ_MODE="file" we use this; for proxy, grid builds per-config)
    if LIQ_MODE == "file":
        print("Loading REAL forceOrder liqs from file:", LIQ_FILE)
        liqs_base = load_liqs_from_file(LIQ_FILE)
    else:
        print("LIQ_MODE=proxy: grid will build proxy liqs per config")
        liqs_base = pd.DataFrame(columns=["ts","side","price","qty","notional"])

    # -----------------------------
    # ✅ Grid Search
    # -----------------------------
    MIN_TRADES = 8  # enforce "take more trades" (tune)
    leverage_fixed = 150
    margin_fixed = 0.40
    taker_fee_fixed = 0.0004

    grid = {
        "liq_z_thresh":      [1.1, 1.3, 1.5, 1.8],
        "liq_z_win_min":     [60, 90, 120],
        "liq_z_roll_min":    [90, 120, 180],
        "strong_burst_z":    [1.8, 2.2, 2.6],
        "edge_persist_bars": [1, 2],
        "sl_atr_mult":       [0.5, 0.7, 0.9],
        "tp_atr_mult":       [0.7, 0.9, 1.1],
        "time_stop_min":     [120, 240, 360],
        "cooldown_min":      [10, 20, 40],
        # proxy sensitivity (ignored if LIQ_MODE="file")
        "proxy_ret_q":       [0.985, 0.99, 0.995],
        "proxy_vol_q":       [0.90, 0.95, 0.98],
        "proxy_wick_q":      [0.985, 0.99, 0.995],
    }

    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    total = 1
    for v in vals:
        total *= len(v)
    print(f"\nGrid size: {total} configs")

    results = []
    best_row = None

    # Optional: cap grid for quick iteration (set to None for full)
    MAX_CONFIGS = None  # e.g., 5000 to cap; None = run all

    n = 0
    for combo in product(*vals):
        n += 1
        if MAX_CONFIGS is not None and n > MAX_CONFIGS:
            break

        cfg = dict(zip(keys, combo))

        curve_df, trades_df, metrics = run_one_config(
            df1m=df1m[["ts","open","high","low","close","vol"]],
            df15m_ind=df15m_ind,
            oi=oi,
            liqs_base=liqs_base,
            leverage=leverage_fixed,
            margin_frac=margin_fixed,
            taker_fee=taker_fee_fixed,
            **cfg
        )
        metrics["score"] = score_row(metrics)
        results.append(metrics)

        if best_row is None or metrics["score"] > best_row["score"]:
            best_row = metrics

        if n % 200 == 0:
            print(f"  progress: {n}/{(MAX_CONFIGS or total)}  best_score={best_row['score']:.4f}  best_trades={best_row['trades']}")

    dfres = pd.DataFrame(results)

    dfok = dfres[dfres["trades"] >= MIN_TRADES].copy()
    if dfok.empty:
        print("\nNo configs met MIN_TRADES. Ranking without the trade filter.")
        dfok = dfres.copy()

    dfok = dfok.sort_values(["score"], ascending=False)
    dfok.to_csv("grid_results.csv", index=False)
    print("\nWrote: grid_results.csv")

    show_cols = [
        "score","return_pct","sharpe_1m_ann","max_drawdown","trades","win_rate",
        "liq_z_thresh","strong_burst_z","liq_z_win_min","liq_z_roll_min","edge_persist_bars",
        "sl_atr_mult","tp_atr_mult","time_stop_min","cooldown_min",
        "proxy_ret_q","proxy_vol_q","proxy_wick_q","liq_events","liq_min_z_rows"
    ]
    print("\nTop 15 configs:")
    print(dfok[show_cols].head(15).to_string(index=False))

    # -----------------------------
    # Run & export trades/equity for the best config
    # -----------------------------
    best = dfok.iloc[0].to_dict()
    print("\nRunning best config and exporting equity_curve.csv + trades.csv ...")

    curve_df, trades_df, metrics_best = run_one_config(
        df1m=df1m[["ts","open","high","low","close","vol"]],
        df15m_ind=df15m_ind,
        oi=oi,
        liqs_base=liqs_base,
        leverage=leverage_fixed,
        margin_frac=margin_fixed,
        taker_fee=taker_fee_fixed,
        liq_z_thresh=float(best["liq_z_thresh"]),
        liq_z_win_min=int(best["liq_z_win_min"]),
        liq_z_roll_min=int(best["liq_z_roll_min"]),
        strong_burst_z=float(best["strong_burst_z"]),
        edge_persist_bars=int(best["edge_persist_bars"]),
        sl_atr_mult=float(best["sl_atr_mult"]),
        tp_atr_mult=float(best["tp_atr_mult"]),
        time_stop_min=int(best["time_stop_min"]),
        cooldown_min=int(best["cooldown_min"]),
        proxy_ret_q=float(best["proxy_ret_q"]),
        proxy_vol_q=float(best["proxy_vol_q"]),
        proxy_wick_q=float(best["proxy_wick_q"]),
    )

    print("\n=== Best Metrics ===")
    for k, v in metrics_best.items():
        if isinstance(v, float):
            print(f"{k:16s}: {v:,.6f}" if abs(v) < 10 else f"{k:16s}: {v:,.2f}")
        else:
            print(f"{k:16s}: {v}")

    curve_df.to_csv("equity_curve.csv", index=False)
    trades_df.to_csv("trades.csv", index=False)
    print("\nWrote: equity_curve.csv, trades.csv")

    if len(trades_df):
        cols = ["entry_ts","exit_ts","side","entry_px","exit_px","pnl","reason","hold_min","equity_after"]
        print("\nLast 10 trades:")
        print(trades_df.tail(10)[cols].to_string(index=False))
