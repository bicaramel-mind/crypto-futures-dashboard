# backtest_eth_htf15_liq_oi.py  (PATCHED AGAIN: relaxed gates + proxy-liq loosening + debug)
# ETHUSDT futures backtest:
# - Signals computed on 15m (HTF), execution on 1m (LTF)
# - Strategy: squeeze-continuation using liquidation burst (net liq z-score) + reclaim + momentum
# - ✅ PATCH: relax conditions so it actually trades even with proxy liquidations:
#     * LIQ_Z_THRESH 2.5 -> 1.5
#     * LIQ_Z_ROLL_MIN 240 -> 120
#     * LIQ_Z_WIN_MIN 60 -> 90
#     * EDGE_PERSIST_BARS 2 -> 1
#     * OI flush is now "soft": (oi_flush OR very strong burst)
#     * Proxy liq detection loosened: quantiles 0.995/0.98 -> 0.99/0.95
# - Debug prints: liq_min_z rows + min/max
#
# Outputs:
# - equity_curve.csv
# - trades.csv
#
# Run:
#   pip install pandas numpy requests
#   python backtest_eth_htf15_liq_oi.py

import math
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import requests

FAPI_REST = "https://fapi.binance.com"
SYMBOL = "ETHUSDT"

# -----------------------------
# Backtest window
# -----------------------------
DAYS_1M = 30
SEED_DAYS_15M = 7

# -----------------------------
# Costs / risk
# -----------------------------
TAKER_FEE = 0.0004
LEVERAGE = 150
MARGIN_FRAC = 0.4

LIQ_MOVE = 1.0 / LEVERAGE

# Exits (hours, not days)
SL_ATR_MULT = 0.7
TP_ATR_MULT = 0.9
TIME_STOP_MIN = 240
MIN_HOLD_MIN = 30

# Risk control
EQUITY_FLOOR = 10.0
COOLDOWN_AFTER_STOP_MIN = 20

# -----------------------------
# ✅ Liquidation burst logic (relaxed)
# -----------------------------
LIQ_Z_WIN_MIN = 90     # was 60
LIQ_Z_THRESH = 1.5     # was 2.5
LIQ_Z_ROLL_MIN = 120   # was 240
EDGE_PERSIST_BARS = 1  # was 2

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
def rest_get_json(path: str, params: dict, timeout=15, max_tries=8):
    url = FAPI_REST + path
    headers = {"User-Agent": "Mozilla/5.0"}
    backoff = 1.0
    for _ in range(max_tries):
        r = requests.get(url, params=params, timeout=timeout, headers=headers)
        if r.status_code == 429:
            time.sleep(backoff)
            backoff = min(backoff * 1.7, 20.0)
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
    out = []
    cur = int(start_ms)
    step = interval_ms(interval)

    while cur < end_ms:
        data = rest_get_json("/fapi/v1/klines", {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1500
        })
        if not data:
            break

        out.extend(data)
        last_open = int(data[-1][0])
        cur = last_open + step

        if len(data) < 1500:
            break

        time.sleep(0.05)

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

def proxy_liqs_from_1m(df1m: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ PATCH: Loosen thresholds so proxy produces enough events:
      ret/wick quantile 0.995 -> 0.99
      vol quantile 0.98 -> 0.95
    """
    d = df1m.copy()
    d["ret_1m"] = d["close"].pct_change()
    d["wick_up"] = (d["high"] - d[["open","close"]].max(axis=1)) / d["close"]
    d["wick_dn"] = (d[["open","close"]].min(axis=1) - d["low"]) / d["close"]

    gate_ret = d["ret_1m"].abs().rolling(240).quantile(0.99)
    gate_vol = d["vol"].rolling(240).quantile(0.95)
    gate_wu  = d["wick_up"].rolling(240).quantile(0.99)
    gate_wd  = d["wick_dn"].rolling(240).quantile(0.99)

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
# ✅ Relaxed 15m squeeze-continuation signal
# -----------------------------
def htf_signal_continuation(htf: pd.DataFrame, oi_row: Optional[pd.Series],
                            liq_min_z: pd.DataFrame, t: pd.Timestamp) -> str:
    if htf is None or len(htf) < 3:
        return "WAIT"

    last = htf.iloc[-1]
    prev = htf.iloc[-2]

    # OI flush (deleveraging) - now soft
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

    # ✅ relaxed: (oi_flush OR very strong burst) + burst + reclaim + momentum
    long_ok = burst_dn and reclaim_up and mom_up and (oi_flush or (np.isfinite(min_z) and (min_z < -2.2)))
    short_ok = burst_up and reclaim_dn and mom_dn and (oi_flush or (np.isfinite(max_z) and (max_z >  2.2)))

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

def run_backtest(df1m: pd.DataFrame, df15m: pd.DataFrame, oi: pd.DataFrame,
                 liq_min_z: pd.DataFrame, equity0: float = 10_000.0):
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

    htf_map = df15m.set_index("ts").sort_index()
    htf_idx = np.searchsorted(htf_map.index.values, df1m["ts"].values, side="right") - 1

    oi_map = oi.set_index("ts").sort_index().reindex(df1m["ts"]).ffill()

    def close_trade(i: int, exit_px: float, reason: str):
        nonlocal equity, pos_side, entry_px, entry_ts, entry_notional, entry_fee, entry_margin, cooldown_until

        t = df1m["ts"].iloc[i]
        hold_min = int((t - entry_ts) / pd.Timedelta(minutes=1))
        equity_before = equity

        ret = (exit_px / entry_px - 1.0) if pos_side == "LONG" else (entry_px / exit_px - 1.0)
        pnl_gross = entry_notional * ret
        exit_fee = entry_notional * TAKER_FEE
        pnl = pnl_gross - entry_fee - exit_fee

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
        if len(htf_slice) < 120:
            eq[i] = equity
            continue

        oi_row = oi_map.iloc[i] if len(oi_map) else None

        # manage position
        if pos_side is not None:
            hold_min = int((t - entry_ts) / pd.Timedelta(minutes=1))

            # liquidation intrabar
            if pos_side == "LONG":
                liq_px = entry_px * (1.0 - LIQ_MOVE)
                if l <= liq_px:
                    close_trade(i, liq_px, "LIQ"); eq[i] = equity; continue
            else:
                liq_px = entry_px * (1.0 + LIQ_MOVE)
                if h >= liq_px:
                    close_trade(i, liq_px, "LIQ"); eq[i] = equity; continue

            # exits via 15m ATR
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

        # entry on 15m boundary
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

    # force close
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

    if LIQ_MODE == "file":
        print("Loading REAL forceOrder liqs from file:", LIQ_FILE)
        liqs = load_liqs_from_file(LIQ_FILE)
    else:
        print("Building PROXY liqs from 1m spikes (approx)")
        liqs = proxy_liqs_from_1m(df1m[["ts","open","high","low","close","vol"]].copy())
    print("Liq events:", len(liqs))

    liq_min_z = build_liq_min_z(liqs)

    # ✅ Debug to confirm why/when it trades
    print("liq_min_z rows:", len(liq_min_z))
    if len(liq_min_z):
        print("liq_z min/max:", float(liq_min_z["liq_z"].min()), float(liq_min_z["liq_z"].max()))

    print("Running backtest (signal=15m continuation relaxed, exec=1m) ...")
    curve_df, trades_df, metrics = run_backtest(
        df1m=df1m[["ts","open","high","low","close","vol"]],
        df15m=df15m_ind,
        oi=oi,
        liq_min_z=liq_min_z,
        equity0=10_000.0,
    )

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:16s}: {v:,.6f}" if abs(v) < 10 else f"{k:16s}: {v:,.2f}")
        else:
            print(f"{k:16s}: {v}")

    curve_df.to_csv("equity_curve.csv", index=False)
    trades_df.to_csv("trades.csv", index=False)
    print("\nWrote: equity_curve.csv, trades.csv")

    if len(trades_df):
        print("\nLast 10 trades:")
        cols = ["entry_ts","exit_ts","side","entry_px","exit_px","pnl","reason","hold_min","equity_after"]
        print(trades_df.tail(10)[cols].to_string(index=False))
