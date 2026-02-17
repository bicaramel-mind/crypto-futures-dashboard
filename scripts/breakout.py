"""
backtest_eth_5m_liq_sweep_v4_degenerate.py

ETHUSDT 5m "liquidation sweep reversal" — DEGNERATE V4 (survivable-overbet edition)

Key changes vs V3:
- Degenerate sizing: high risk per trade but with guardrails (leverage cap + fail-fast).
- Higher hit-rate bias: TP1 sooner + larger fraction, tighter stop, faster time-stop.
- Trail only activates once trade is green (prevents tiny churn + avoids giving back huge).
- Fail-fast exit in first N bars if reclaim fails (cuts many -1R into smaller losses).
- Still uses: sweep -> reclaim -> retest touch -> bullish bounce (quality entry)
- Still uses: capitulation filters (TR z-score + vol spike + wick), plus trend filters.

Run:
  pip install pandas numpy httpx
  python backtest_eth_5m_liq_sweep_v4_degenerate.py

Outputs:
  liq_v4_equity.csv
  liq_v4_trades.csv
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import httpx
import numpy as np
import pandas as pd

REST = "https://fapi.binance.com"
SYMBOL = "ETHUSDT"
INTERVAL = "5m"
DAYS = 90

# =========================
# DEGENERATE SETTINGS (V4)
# =========================
START_EQUITY = 100.0

# Overbet, but not suicidal
RISK_PER_TRADE = 0.4     # 15% risk per trade (degenerate)
MAX_LEVERAGE = 150.0       # hard cap to prevent insane notional explosions

# Strategy / signal
SWEEP_LOOKBACK = 72          # 6h
RECLAIM_WINDOW = 6           # 30m
WICK_MIN = 0.60              # stricter
VOL_SPIKE_Q = 0.90           # stricter (top 10%)

ATR_N = 14
STOP_ATR_MULT = 0.5         # tighter stop to raise hit rate

TR_Z_WIN = 288
TR_Z_MIN = 1.5               # stricter capitulation

EMA_FAST = 48
EMA_SLOW = 240
TREND_STRENGTH_MAX = 0.0035
MOMENTUM_1H_MAX = 0.012

TRENDINESS_WIN_6H = 72
TRENDINESS_MAX = 0.85        # stricter: avoid trend days

# Optional OI
OI_PERIOD_PRIMARY = "5m"
OI_Z_WIN = 96
OI_DROP_Z = -1.0             # slightly less strict than V3 (since we tightened other gates)

# Funding (historical)
FUNDING_BAD_LONG = -0.0012   # avoid catching knife on extreme negative funding

# Retest entry tuning
RETEST_ROLL_BARS = 3
TOUCH_ATR_TOL = 0.12

# Exits tuned for high hit rate
TP1_R = 0.6
TP1_FRACTION = 0.75
TRAIL_ACTIVATE_R = 0.15
TRAIL_ATR_MULT = 0.8
TIME_STOP_BARS = 12          # 60 minutes

# VWAP guard
VWAP_EXIT_MIN_R = 0.10
SCRATCH_R = -0.05

# Fail-fast (cuts many full stops)
FAIL_FAST_BARS = 2
FAIL_FAST_R = -0.35          # if <= -0.35R within first FAIL_FAST_BARS, exit now
FAIL_FAST_BREAK_LEVEL = True # also exit if close < prior_low_level within first bars

# Cooldown
COOLDOWN_AFTER_STOP_BARS = 48  # 4h cooldown after a full stop

# Costs
TAKER_FEE = 0.0004
SLIPPAGE_BP = 2.0
KLINE_LIMIT = 1500


# =========================
# Helpers
# =========================
def now_ms() -> int:
    import time
    return int(time.time() * 1000)


def interval_ms(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1]) * 60_000
    if interval.endswith("h"):
        return int(interval[:-1]) * 3_600_000
    if interval.endswith("d"):
        return int(interval[:-1]) * 86_400_000
    raise ValueError(interval)


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(5, n // 2)).mean()


def zscore(s: pd.Series, w: int) -> pd.Series:
    m = s.rolling(w, min_periods=max(10, w // 3)).mean()
    sd = s.rolling(w, min_periods=max(10, w // 3)).std(ddof=0).replace(0, np.nan)
    return (s - m) / sd


def apply_costs(notional: float) -> float:
    slip = SLIPPAGE_BP / 10_000.0
    return notional * (2 * TAKER_FEE + 2 * slip)


def wick_fraction(o: float, h: float, l: float, c: float) -> float:
    rng = max(h - l, 1e-12)
    if c >= o:
        return (o - l) / rng
    return (c - l) / rng


# =========================
# Data fetch
# =========================
def fetch_klines_paginated(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    step = interval_ms(interval)
    out = []
    cur = start_ms
    with httpx.Client(timeout=30) as c:
        while cur < end_ms:
            r = c.get(
                f"{REST}/fapi/v1/klines",
                params={"symbol": symbol, "interval": interval, "startTime": cur, "endTime": end_ms, "limit": KLINE_LIMIT},
            )
            r.raise_for_status()
            data = r.json()
            if not data:
                break
            out.extend(data)
            last_open = int(data[-1][0])
            nxt = last_open + step
            if nxt <= cur:
                break
            cur = nxt
            if len(data) < KLINE_LIMIT:
                break

    df = pd.DataFrame(out, columns=[
        "openTime","open","high","low","close","volume","closeTime","qv","n","tbbv","tbqv","ignore"
    ])
    df["ts"] = pd.to_datetime(df["openTime"].astype(int), unit="ms", utc=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df[["ts","open","high","low","close","volume"]].sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


def fetch_open_interest_hist(symbol: str, period: str, start_ms: int, end_ms: int, limit: int = 500) -> pd.DataFrame:
    with httpx.Client(timeout=30) as c:
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
    df["ts"] = df["ts"].dt.floor("5min")
    return df[["ts","oi"]].sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


def get_oi_5m_aligned(start_ms: int, end_ms: int) -> Tuple[pd.DataFrame, str]:
    # try 5m
    try:
        oi = fetch_open_interest_hist(SYMBOL, OI_PERIOD_PRIMARY, start_ms, end_ms)
        if not oi.empty:
            return oi, OI_PERIOD_PRIMARY
    except Exception:
        pass
    # fallback 15m
    try:
        oi15 = fetch_open_interest_hist(SYMBOL, "15m", start_ms, end_ms)
        if oi15.empty:
            return pd.DataFrame(columns=["ts","oi"]), "none"
        oi15["ts"] = oi15["ts"].dt.floor("15min")
        s = oi15.set_index("ts")["oi"].resample("5min").ffill()
        return pd.DataFrame({"ts": s.index, "oi": s.values}).reset_index(drop=True), "15m→5m"
    except Exception:
        return pd.DataFrame(columns=["ts","oi"]), "none"


def fetch_funding_resampled_5m(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    with httpx.Client(timeout=30) as c:
        r = c.get(
            f"{REST}/fapi/v1/fundingRate",
            params={"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000},
        )
        r.raise_for_status()
        data = r.json()
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["ts", "funding"])
    df["ts"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True)
    df["funding"] = df["fundingRate"].astype(float)
    s = df.set_index("ts")["funding"].sort_index().resample("5min").last().ffill()
    return pd.DataFrame({"ts": s.index, "funding": s.values}).reset_index(drop=True)


# =========================
# Features
# =========================
def compute_features(df: pd.DataFrame, oi: pd.DataFrame, funding_5m: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Trend filters
    d["ema_fast"] = ema(d["close"], EMA_FAST)
    d["ema_slow"] = ema(d["close"], EMA_SLOW)
    d["trend_strength"] = (d["ema_fast"] - d["ema_slow"]).abs() / d["close"].replace(0, np.nan)
    d["ret_1h"] = d["close"].pct_change(12).abs()

    d["atr"] = atr(d, ATR_N)

    # Trendiness 6h: abs move / sum ATR
    move_6h = (d["close"] - d["close"].shift(TRENDINESS_WIN_6H)).abs()
    atr_6h = d["atr"].rolling(TRENDINESS_WIN_6H, min_periods=TRENDINESS_WIN_6H // 2).sum()
    d["trendiness_6h"] = (move_6h / atr_6h.replace(0, np.nan))

    # VWAP full-series
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    d["vwap"] = (tp * d["volume"]).cumsum() / d["volume"].cumsum().replace(0, np.nan)

    # TR z-score (capitulation)
    tr = pd.concat([
        (d["high"] - d["low"]),
        (d["high"] - d["close"].shift()).abs(),
        (d["low"] - d["close"].shift()).abs()
    ], axis=1).max(axis=1)
    d["tr_z"] = zscore(tr, TR_Z_WIN)

    # Wick
    d["wick_frac"] = [wick_fraction(o, h, l, c) for o, h, l, c in zip(d["open"], d["high"], d["low"], d["close"])]

    # Sweep -> reclaim
    d["prior_low"] = d["low"].shift(1).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK // 2).min()
    d["swept_low"] = d["low"] < d["prior_low"]
    d["reclaim_low"] = d["close"] > d["prior_low"]

    d["last_sweep_ts"] = d["ts"].where(d["swept_low"]).ffill()
    d["bars_since_sweep"] = ((d["ts"] - d["last_sweep_ts"]).dt.total_seconds() / 300.0).round().astype("Int64")
    d["reclaim_ok"] = d["reclaim_low"] & d["bars_since_sweep"].between(0, RECLAIM_WINDOW)

    # Volume spike (stricter)
    d["vol_q"] = d["volume"].rolling(288, min_periods=60).quantile(VOL_SPIKE_Q)
    d["vol_spike"] = d["volume"] >= d["vol_q"]

    # OI (optional)
    if not oi.empty:
        d = d.merge(oi, on="ts", how="left")
        d["oi"] = d["oi"].ffill()
        d["oi_chg"] = d["oi"].diff()
        d["oi_z"] = zscore(d["oi_chg"], OI_Z_WIN)
    else:
        d["oi_z"] = np.nan

    # Funding
    if not funding_5m.empty:
        d = d.merge(funding_5m, on="ts", how="left")
        d["funding"] = d["funding"].ffill().fillna(0.0)
    else:
        d["funding"] = 0.0

    # Retest entry
    d["prior_low_level"] = d["prior_low"]
    d["touch_prior_low"] = d["low"] <= (d["prior_low_level"] + TOUCH_ATR_TOL * d["atr"])
    d["bull_bounce"] = (d["close"] > d["open"]) & (d["close"] > d["prior_low_level"])
    d["reclaim_recent"] = d["reclaim_ok"].rolling(RETEST_ROLL_BARS, min_periods=1).max().astype(bool)
    d["retest_entry"] = d["reclaim_recent"] & d["touch_prior_low"] & d["bull_bounce"]

    return d


def long_entry(row: pd.Series) -> bool:
    # Avoid trend days
    trend_ok = (row["trend_strength"] <= TREND_STRENGTH_MAX) and (row["ret_1h"] <= MOMENTUM_1H_MAX)
    trendiness_ok = (not np.isfinite(row["trendiness_6h"])) or (row["trendiness_6h"] <= TRENDINESS_MAX)

    # Must be violent exhaustion
    wick_ok = row["wick_frac"] >= WICK_MIN
    vol_ok = bool(row["vol_spike"])
    cap_ok = (np.isfinite(row["tr_z"]) and row["tr_z"] >= TR_Z_MIN)

    # OI optional but if present should suggest flush
    oi_ok = (not np.isfinite(row["oi_z"])) or (row["oi_z"] <= OI_DROP_Z)

    funding_ok = float(row["funding"]) >= FUNDING_BAD_LONG

    return bool(row.get("retest_entry", False)) and trend_ok and trendiness_ok and wick_ok and vol_ok and cap_ok and oi_ok and funding_ok


# =========================
# Backtest
# =========================
@dataclass
class Trade:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry: float
    exit: float
    notional: float
    pnl: float
    reason: str


def backtest(d: pd.DataFrame) -> Tuple[pd.DataFrame, List[Trade]]:
    equity = float(START_EQUITY)

    pos = 0
    entry_px = 0.0
    entry_ts = None
    entry_i = None
    entry_prior_low = np.nan

    notional = 0.0
    stop_px = 0.0
    r_unit = 0.0

    tp1_px = 0.0
    took_tp1 = False
    trail_px = np.nan
    bars_in = 0

    cooldown = 0

    trades: List[Trade] = []
    eq_rows = []

    start_i = 600  # warmup

    for i in range(start_i, len(d) - 1):
        row = d.iloc[i]
        nxt = d.iloc[i + 1]
        ts = row["ts"]

        if cooldown > 0:
            cooldown -= 1

        eq_rows.append({"ts": ts, "equity": equity, "pos": pos, "close": row["close"]})

        hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])

        # -------------
        # Manage
        # -------------
        if pos == 1:
            bars_in += 1

            # no same-bar exit
            if entry_i is not None and i <= entry_i:
                continue

            # R now
            r_now = (cl - entry_px) / max(r_unit, 1e-12)

            # FAIL-FAST: early rejection
            if bars_in <= FAIL_FAST_BARS:
                if r_now <= FAIL_FAST_R:
                    # exit at close (conservative: assume you can get close-ish)
                    exit_px = cl
                    rem_notional = notional if not took_tp1 else notional * (1.0 - TP1_FRACTION)
                    pnl = (exit_px - entry_px) / entry_px * rem_notional
                    equity += pnl - apply_costs(rem_notional)
                    trades.append(Trade(entry_ts, ts, entry_px, exit_px, notional, pnl, "FAIL_FAST"))
                    pos = 0
                    entry_px = 0.0
                    entry_ts = None
                    entry_i = None
                    entry_prior_low = np.nan
                    notional = 0.0
                    stop_px = 0.0
                    r_unit = 0.0
                    tp1_px = 0.0
                    took_tp1 = False
                    trail_px = np.nan
                    bars_in = 0
                    cooldown = max(cooldown, 8)  # small cooldown after fail-fast
                    continue

                if FAIL_FAST_BREAK_LEVEL and np.isfinite(entry_prior_low) and cl < float(entry_prior_low):
                    exit_px = cl
                    rem_notional = notional if not took_tp1 else notional * (1.0 - TP1_FRACTION)
                    pnl = (exit_px - entry_px) / entry_px * rem_notional
                    equity += pnl - apply_costs(rem_notional)
                    trades.append(Trade(entry_ts, ts, entry_px, exit_px, notional, pnl, "FAIL_BREAK"))
                    # reset
                    pos = 0
                    entry_px = 0.0
                    entry_ts = None
                    entry_i = None
                    entry_prior_low = np.nan
                    notional = 0.0
                    stop_px = 0.0
                    r_unit = 0.0
                    tp1_px = 0.0
                    took_tp1 = False
                    trail_px = np.nan
                    bars_in = 0
                    cooldown = max(cooldown, 8)
                    continue

            # Trail activation gate (only once green enough)
            if np.isfinite(row["atr"]) and float(row["atr"]) > 0 and r_now >= TRAIL_ACTIVATE_R:
                new_trail = float(cl - TRAIL_ATR_MULT * float(row["atr"]))
                trail_px = new_trail if not np.isfinite(trail_px) else max(float(trail_px), new_trail)

            # TP1 (fast)
            if not took_tp1 and hi >= tp1_px:
                pnl_tp1 = (tp1_px - entry_px) / entry_px * (notional * TP1_FRACTION)
                equity += pnl_tp1 - apply_costs(notional * TP1_FRACTION)
                took_tp1 = True

            rem_frac = (1.0 - TP1_FRACTION) if took_tp1 else 1.0
            rem_notional = notional * rem_frac

            hit = None
            exit_px = None

            # Hard stop
            if lo <= stop_px:
                hit, exit_px = "STOP", stop_px

            # Trail stop
            if hit is None and np.isfinite(trail_px) and lo <= float(trail_px):
                hit, exit_px = "TRAIL", float(trail_px)

            # VWAP exit (profit/scratch)
            if hit is None and np.isfinite(row["vwap"]):
                r_now = (cl - entry_px) / max(r_unit, 1e-12)
                if cl >= float(row["vwap"]) and (r_now >= VWAP_EXIT_MIN_R or r_now >= SCRATCH_R):
                    hit, exit_px = "VWAP", cl

            # Time stop (mean reversion should be quick)
            if hit is None and bars_in >= TIME_STOP_BARS:
                hit, exit_px = "TIME", cl

            if hit is not None:
                pnl = (exit_px - entry_px) / entry_px * rem_notional
                equity += pnl - apply_costs(rem_notional)
                trades.append(Trade(entry_ts, ts, entry_px, exit_px, notional, pnl, hit))

                if hit == "STOP":
                    cooldown = COOLDOWN_AFTER_STOP_BARS

                # reset
                pos = 0
                entry_px = 0.0
                entry_ts = None
                entry_i = None
                entry_prior_low = np.nan
                notional = 0.0
                stop_px = 0.0
                r_unit = 0.0
                tp1_px = 0.0
                took_tp1 = False
                trail_px = np.nan
                bars_in = 0

        # -------------
        # Enter
        # -------------
        if pos == 0 and cooldown == 0:
            if not np.isfinite(row["atr"]) or float(row["atr"]) <= 0:
                continue

            if long_entry(row):
                # stop distance as pct
                stop_dist = max(0.0025, STOP_ATR_MULT * (float(row["atr"]) / float(row["close"])))

                # risk dollars (degenerate)
                risk_usd = equity * RISK_PER_TRADE

                # notional implied by stop_dist
                notional = risk_usd / stop_dist

                # HARD leverage cap (prevents pathological sizing)
                notional = min(notional, equity * MAX_LEVERAGE)

                entry_px = float(nxt["open"])
                entry_ts = nxt["ts"]
                entry_i = i + 1
                entry_prior_low = float(row.get("prior_low_level", np.nan))

                pos = 1
                stop_px = float(entry_px * (1 - stop_dist))
                r_unit = float(entry_px - stop_px)
                tp1_px = float(entry_px + TP1_R * r_unit)

                took_tp1 = False
                trail_px = np.nan
                bars_in = 0

    return pd.DataFrame(eq_rows), trades


def summarize(eq: pd.Series) -> Dict[str, float]:
    eq = eq.dropna()
    rets = eq.pct_change().fillna(0.0)
    bars_per_year = 365 * 24 * 12
    mu, sd = rets.mean(), rets.std(ddof=0)
    sharpe = float("nan") if sd == 0 else float((mu / sd) * math.sqrt(bars_per_year))
    peak = eq.cummax()
    dd = (eq - peak) / peak.replace(0, np.nan)
    return {
        "end_equity": float(eq.iloc[-1]) if len(eq) else float("nan"),
        "net_pnl": float(eq.iloc[-1] - eq.iloc[0]) if len(eq) else float("nan"),
        "sharpe": sharpe,
        "max_dd_pct": float(dd.min() * 100.0) if len(dd) else 0.0,
        "bars": int(len(eq)),
    }


def main():
    end = now_ms()
    start = end - DAYS * 24 * 60 * 60 * 1000

    print(f"Fetching {SYMBOL} {INTERVAL} for ~{DAYS} days…")
    df = fetch_klines_paginated(SYMBOL, INTERVAL, start, end)
    print("Bars:", len(df), "|", df["ts"].min(), "→", df["ts"].max())

    print("Fetching OI…")
    oi, src = get_oi_5m_aligned(start, end)
    print("OI source:", src, "| rows:", len(oi))

    print("Fetching funding history…")
    funding_5m = fetch_funding_resampled_5m(SYMBOL, start, end)
    print("Funding rows:", len(funding_5m))

    d = compute_features(df, oi, funding_5m)

    warm = d.dropna(subset=["prior_low", "atr", "trend_strength", "ret_1h", "tr_z"])
    if len(warm):
        setups = int(warm.apply(long_entry, axis=1).sum())
        print(f"Diagnostics: candidate setup bars={setups} (post-warmup rows={len(warm)})")

    print("Running backtest…")
    eq, trades = backtest(d)
    stats = summarize(eq["equity"])

    print("\n===== LIQ SWEEP REVERSAL V4 — DEGENERATE (ETH 5m) =====")
    print(f"End equity:  ${stats['end_equity']:.2f}")
    print(f"Net PnL:     ${stats['net_pnl']:.2f}")
    print(f"Sharpe:      {stats['sharpe']:.2f}")
    print(f"Max DD:      {stats['max_dd_pct']:.2f}%")
    print(f"Trades:      {len(trades)}")
    print("=======================================================")

    eq.to_csv("liq_v4_equity.csv", index=False)
    pd.DataFrame([t.__dict__ for t in trades]).to_csv("liq_v4_trades.csv", index=False)
    print("Saved: liq_v4_equity.csv, liq_v4_trades.csv")


if __name__ == "__main__":
    main()
