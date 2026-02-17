#!/usr/bin/env python3
"""
liq_exhaustion_backtest_binance_only.py

Binance-only backtest approximating the live liquidation exhaustion strategy.

Important limitation:
- Binance does NOT offer historical forceOrder (liquidation prints) via REST.
- Therefore this backtest uses a "liq intensity proxy" built from:
  * kline volume + taker imbalance + impulse returns
  * open interest collapse from /futures/data/openInterestHist (Binance OI)

Data sources used (all Binance Futures REST):
- /fapi/v1/klines                 (OHLCV + taker buy volumes)
- /fapi/v1/fundingRate            (funding events; optional)
- /futures/data/openInterestHist  (historical OI; 5m if available, else 15m)

Implements:
- 1m bars default (configurable)
- Burst z-score on proxy intensity
- Peak tracking + decay (exhaustion)
- Cascade guard (avoid fading while burst is accelerating)
- Mean-reversion target to EMA (optional), with stop/TP/time stop
- Simple risk-based position sizing with leverage cap
- Fees + (optional) funding approximation

Run:
  python liq_exhaustion_backtest_binance_only.py --symbol ETHUSDT --days 90 --tf 1m
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import httpx


REST = "https://fapi.binance.com"

# --------------------------
# Timezone safety
# --------------------------
def to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Make sure ts is UTC tz-aware, without ever tz_localize-ing tz-aware timestamps.
    """
    if not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

# --------------------------
# Utilities
# --------------------------
def utc_ms(ts: pd.Timestamp) -> int:
    ts = to_utc(ts)
    return int(ts.value // 10**6)

def now_ms() -> int:
    return int(time.time() * 1000)

def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default

def minp_for(window: int, floor: int = 3) -> int:
    """
    Adaptive min_periods that is ALWAYS <= window.
    """
    window = int(max(1, window))
    # default heuristic: ~1/3 of window, but at least floor
    minp = max(floor, window // 3)
    return int(min(minp, window))

def rolling_zscore(s: pd.Series, window: int, minp: Optional[int] = None) -> pd.Series:
    if s.empty:
        return s
    window = int(max(3, window))
    if minp is None:
        minp = max(3, window // 3)
    minp = min(minp, window)
    m = s.rolling(window, min_periods=minp).mean()
    sd = s.rolling(window, min_periods=minp).std(ddof=0).replace(0, np.nan)
    return ((s - m) / sd).fillna(0.0)


def clip(a, lo, hi):
    return max(lo, min(hi, a))

# --------------------------
# Fetchers (Binance)
# --------------------------
def fetch_klines(
    client: httpx.Client,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1500,
) -> List[list]:
    r = client.get(
        f"{REST}/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": limit},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()

def fetch_funding_rate(
    client: httpx.Client,
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
) -> List[dict]:
    r = client.get(
        f"{REST}/fapi/v1/fundingRate",
        params={"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": limit},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()

def fetch_oi_hist(
    client: httpx.Client,
    symbol: str,
    period: str,
    start_ms: int,
    end_ms: int,
    limit: int = 500,
) -> List[dict]:
    r = client.get(
        f"{REST}/futures/data/openInterestHist",
        params={"symbol": symbol, "period": period, "startTime": start_ms, "endTime": end_ms, "limit": limit},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()

def fetch_klines_range(
    symbol: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Robust chunk fetch, handles rate limits by sleeping.
    """
    out = []
    start = to_utc(start)
    end = to_utc(end)
    start_ms = utc_ms(start)
    end_ms = utc_ms(end)

    with httpx.Client() as client:
        cur = start_ms
        tf_ms = {
            "1m": 60_000,
            "3m": 180_000,
            "5m": 300_000,
            "15m": 900_000,
            "30m": 1_800_000,
            "1h": 3_600_000,
        }[interval]
        max_chunk_ms = tf_ms * 1500

        while cur < end_ms:
            nxt = min(cur + max_chunk_ms, end_ms)
            try:
                data = fetch_klines(client, symbol, interval, cur, nxt, limit=1500)
                out.extend(data)
                if len(data) == 0:
                    cur = nxt
                else:
                    last_close = int(data[-1][6])
                    cur = last_close + 1
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                if code in (429, 418):
                    time.sleep(2.0)
                    continue
                raise

            time.sleep(0.05)

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(
        out,
        columns=[
            "openTime","open","high","low","close","volume","closeTime",
            "quoteVolume","nTrades","takerBuyBase","takerBuyQuote","ignore"
        ],
    )
    df["ts"] = pd.to_datetime(df["openTime"].astype(np.int64), unit="ms", utc=True)
    for c in ["open","high","low","close","volume","quoteVolume","takerBuyBase","takerBuyQuote"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    return df

def fetch_funding_range(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    out = []
    start = to_utc(start)
    end = to_utc(end)
    start_ms = utc_ms(start)
    end_ms = utc_ms(end)
    with httpx.Client() as client:
        cur = start_ms
        while cur < end_ms:
            try:
                data = fetch_funding_rate(client, symbol, cur, end_ms, limit=1000)
                out.extend(data)
                if not data:
                    break
                cur = int(data[-1]["fundingTime"]) + 1
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                if code in (429, 418):
                    time.sleep(2.0)
                    continue
                break
            time.sleep(0.05)

    if not out:
        return pd.DataFrame(columns=["ts","funding_rate"])

    df = pd.DataFrame(out)
    df["ts"] = pd.to_datetime(df["fundingTime"].astype(np.int64), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["ts","funding_rate"]].sort_values("ts").reset_index(drop=True)

def fetch_oi_range_with_fallback(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    primary_period: str = "5m",
    fallback_period: str = "15m",
    require_oi: bool = True,
    sleep_s: float = 0.25,
) -> Tuple[pd.DataFrame, str]:
    """
    Fetch OI hist from Binance Futures REST with retries/backoff.
    If blocked (403) repeatedly, returns empty (or raises if require_oi=True).
    """
    start_ms = utc_ms(start)
    end_ms = utc_ms(end)

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }

    def _fetch(period: str) -> pd.DataFrame:
        rows = []
        minutes = int(period[:-1])
        step_ms = minutes * 60_000 * 500  # 500 points max per call

        with httpx.Client(headers=HEADERS) as client:
            cur = start_ms
            tries_403 = 0

            while cur < end_ms:
                nxt = min(cur + step_ms, end_ms)

                for attempt in range(6):
                    try:
                        data = fetch_oi_hist(client, symbol, period, cur, nxt, limit=500)
                        rows.extend(data)
                        if not data:
                            cur = nxt
                        else:
                            cur = int(data[-1]["timestamp"]) + 1
                        break
                    except httpx.HTTPStatusError as e:
                        code = e.response.status_code

                        if code in (418, 429):
                            time.sleep(2.0 + 1.5 * attempt)
                            continue

                        if code == 403:
                            tries_403 += 1
                            time.sleep(3.0 + 2.0 * attempt)
                            if tries_403 >= 3:
                                return pd.DataFrame()
                            continue

                        if code == 400:
                            # sometimes "invalid range" - skip this chunk and move on
                            cur = nxt
                            break

                        # other hard errors: stop this period
                        return pd.DataFrame()

                time.sleep(sleep_s)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms", utc=True)
        df["oi"] = df["sumOpenInterest"].astype(float)
        df = df[["ts", "oi"]].sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
        return df

    df = _fetch(primary_period)
    if not df.empty:
        return df, primary_period

    df2 = _fetch(fallback_period)
    if not df2.empty:
        return df2, fallback_period

    if require_oi:
        raise SystemExit(
            "Binance OI history could not be fetched (403/blocked or empty). "
            "You said 'stick to Binance OI' — so aborting instead of silently running without OI.\n"
            "Try: --tf 5m and --oi_period 15m (fewer calls), or run on a network that can access Binance futures endpoints."
        )

    return pd.DataFrame(columns=["ts", "oi"]), "none"


# --------------------------
# Strategy / Backtest
# --------------------------
@dataclass
class Params:
    symbol: str = "ETHUSDT"
    tf: str = "1m"
    days: int = 90

    # signal windows (in bars; derived from seconds logic)
    burst_win: int = 30
    peak_win: int = 90
    exh_win: int = 25
    vel_win: int = 15
    zwin: int = 1800

    # thresholds
    burst_z_th: float = 3.0
    impulse_th: float = 0.003
    rate_frac: float = 0.35
    vel_decay_frac: float = 0.45
    z_accel_th: float = 0.35

    # risk/exits
    start_equity: float = 10_000.0
    risk_per_trade: float = 0.02
    leverage_cap: float = 15.0
    max_pos_notional: float = 250_000

    fee_bps: float = 4.0
    slippage_bps: float = 1.0

    stop_atr_mult: float = 2.5
    atr_len: int = 14

    tp1_r: float = 1.5
    tp1_fraction: float = 0.6

    trail_activate_r: float = 0.8
    trail_atr_mult: float = 2.0

    time_stop_bars: int = 60

    use_ema_target: bool = True
    ema_span: int = 180

    # OI integration
    oi_drop_weight: float = 1.0
    taker_imb_weight: float = 0.6
    vol_weight: float = 0.8
    impulse_weight: float = 1.0

    min_stop_dist_pct: float = 0.0012     # 0.12% minimum stop distance (avoid micro churn)
    cooldown_bars: int = 5                # wait N bars after exit before re-entering
    min_liq_rate_z: float = 2.2           # stronger burst requirement for proxy
    min_expected_r: float = 0.35          # require at least ~0.35R “room” vs costs

def infer_tf_minutes(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    raise ValueError(tf)

def compute_features(df: pd.DataFrame, oi_df: pd.DataFrame, oi_period: str, p: Params) -> pd.DataFrame:
    d = df.copy()

    # returns
    d["ret1"] = d["close"].pct_change()
    d["impulse"] = d["close"].pct_change(p.peak_win).fillna(0.0)

    # taker imbalance proxy
    d["taker_buy"] = d["takerBuyBase"]
    d["taker_sell"] = (d["volume"] - d["takerBuyBase"]).clip(lower=0.0)
    denom = (d["taker_buy"] + d["taker_sell"]).replace(0, np.nan)
    d["taker_imb"] = ((d["taker_buy"] - d["taker_sell"]) / denom).fillna(0.0)

    d["abs_ret"] = d["ret1"].abs().fillna(0.0)

    # ATR for stops
    high = d["high"]
    low = d["low"]
    prev_close = d["close"].shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(p.atr_len, min_periods=minp_for(p.atr_len, floor=5)).mean()

    # ---- OI merge & collapse proxy
    oi = oi_df.copy()
    if oi.empty:
        d["oi"] = np.nan
    else:
        oi = oi.sort_values("ts").drop_duplicates("ts")
        oi = oi.set_index("ts")["oi"]
        oi = oi.resample(p.tf).last().ffill()
        d = d.merge(oi.rename("oi"), left_on="ts", right_index=True, how="left")
        d["oi"] = d["oi"].ffill()

    d["oi_chg"] = d["oi"].diff()
    d["oi_collapse"] = (-d["oi_chg"]).clip(lower=0.0)
    d["oi_collapse_z"] = rolling_zscore(d["oi_collapse"], p.zwin)

    # ---- liquidation intensity proxy
    vol_z = rolling_zscore(d["volume"], p.zwin)
    imp_z = rolling_zscore(d["impulse"].abs(), p.zwin)
    imb_z = rolling_zscore(d["taker_imb"].abs(), p.zwin)

    d["liq_proxy_z"] = (
        p.vol_weight * np.maximum(vol_z, 0)
        + p.impulse_weight * np.maximum(imp_z, 0)
        + p.taker_imb_weight * np.maximum(imb_z, 0)
        + p.oi_drop_weight * np.maximum(d["oi_collapse_z"], 0)
    )

    # rolling sums/maxes need min_periods <= window
    bw = int(max(1, p.burst_win))
    pw = int(max(bw + 1, p.peak_win))
    vw = int(max(1, p.vel_win))
    ew = int(max(1, p.exh_win))

    d["liq_rate"] = (
        np.maximum(d["liq_proxy_z"], 0)
        .rolling(bw, min_periods=minp_for(bw, floor=3))
        .sum()
        .fillna(0.0)
    )
    d["liq_rate_z"] = rolling_zscore(d["liq_rate"], p.zwin)

    d["vel"] = d["ret1"].rolling(vw, min_periods=minp_for(vw, floor=3)).sum().fillna(0.0)

    d["liq_rate_peak"] = d["liq_rate"].rolling(pw, min_periods=minp_for(pw, floor=3)).max()
    d["vel_peak"] = d["vel"].abs().rolling(pw, min_periods=minp_for(pw, floor=3)).max()

    d["exh_rate"] = (d["liq_rate"] <= (p.rate_frac * d["liq_rate_peak"].replace(0, np.nan))).astype(float).fillna(0.0)
    d["exh_vel"] = (d["vel"].abs() <= (p.vel_decay_frac * d["vel_peak"].replace(0, np.nan))).astype(float).fillna(0.0)
    d["exhausted"] = ((d["exh_rate"] > 0.5) & (d["exh_vel"] > 0.5)).astype(int)

    d["z_d1"] = d["liq_rate_z"].diff().fillna(0.0)
    d["cascade_guard"] = ((d["z_d1"] > p.z_accel_th) & (d["liq_rate_z"] > p.burst_z_th)).astype(int)

    d["bias"] = "NONE"
    d.loc[(d["liq_rate_z"] >= p.burst_z_th) & (d["taker_imb"] < 0) & (d["impulse"] < -p.impulse_th), "bias"] = "LONG"
    d.loc[(d["liq_rate_z"] >= p.burst_z_th) & (d["taker_imb"] > 0) & (d["impulse"] >  p.impulse_th), "bias"] = "SHORT"

    d["exh_recent"] = d["exhausted"].rolling(ew, min_periods=1).max().fillna(0).astype(int)

    d["ema"] = d["close"].ewm(span=p.ema_span, adjust=False).mean()

    d.attrs["oi_period_used"] = oi_period
    return d

@dataclass
class Trade:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    side: str
    entry: float
    exit: float
    qty: float
    notional: float
    pnl: float
    reason: str

def fee_cost(notional: float, fee_bps: float, slippage_bps: float) -> float:
    bps_total = (fee_bps + slippage_bps) * 2.0
    return notional * (bps_total / 10_000.0)

def backtest(d: pd.DataFrame, p: Params) -> Tuple[pd.DataFrame, List[Trade], dict]:

    cooldown = 0
    equity = p.start_equity
    trades: List[Trade] = []

    in_pos = False
    side = None
    entry_px = None
    entry_ts = None
    qty = 0.0
    stop_px = None
    trail_px = None
    tp1_px = None
    tp1_done = False
    bars_in_trade = 0
    r_unit = 0.0

    eq_curve = []

    for i in range(len(d)):
        row = d.iloc[i]
        ts = row["ts"]
        px = float(row["close"])
        atr = float(row["atr"]) if np.isfinite(row["atr"]) else np.nan

        if in_pos:
            if side == "LONG":
                unreal = (px - entry_px) * qty
            else:
                unreal = (entry_px - px) * qty
            eq_curve.append((ts, equity + unreal))
        else:
            eq_curve.append((ts, equity))

        if in_pos:
            bars_in_trade += 1

            if r_unit > 0:
                r_now = (px - entry_px) / r_unit if side == "LONG" else (entry_px - px) / r_unit
            else:
                r_now = 0.0

            if np.isfinite(atr) and r_now >= p.trail_activate_r:
                if side == "LONG":
                    new_trail = px - p.trail_atr_mult * atr
                    trail_px = max(trail_px, new_trail) if trail_px is not None else new_trail
                else:
                    new_trail = px + p.trail_atr_mult * atr
                    trail_px = min(trail_px, new_trail) if trail_px is not None else new_trail

            if (not tp1_done) and (r_unit > 0):
                hit = (px >= tp1_px) if side == "LONG" else (px <= tp1_px)
                if hit:
                    close_qty = qty * p.tp1_fraction
                    close_notional = close_qty * px
                    pnl_part = (px - entry_px) * close_qty if side == "LONG" else (entry_px - px) * close_qty
                    pnl_part -= fee_cost(close_notional, p.fee_bps, p.slippage_bps)
                    equity += pnl_part
                    qty -= close_qty
                    tp1_done = True
                    if side == "LONG":
                        stop_px = max(stop_px, entry_px) if stop_px is not None else entry_px
                    else:
                        stop_px = min(stop_px, entry_px) if stop_px is not None else entry_px

            exit_reason = None
            exit_px = None

            if stop_px is not None:
                if side == "LONG" and px <= stop_px:
                    exit_reason, exit_px = "STOP", px
                if side == "SHORT" and px >= stop_px:
                    exit_reason, exit_px = "STOP", px

            if exit_reason is None and trail_px is not None:
                if side == "LONG" and px <= trail_px:
                    exit_reason, exit_px = "TRAIL", px
                if side == "SHORT" and px >= trail_px:
                    exit_reason, exit_px = "TRAIL", px

            if exit_reason is None and bars_in_trade >= p.time_stop_bars:
                exit_reason, exit_px = "TIME", px

            if exit_reason is None and p.use_ema_target:
                ema = float(row["ema"])
                if np.isfinite(ema):
                    if side == "LONG" and px >= ema:
                        exit_reason, exit_px = "EMA", px
                    if side == "SHORT" and px <= ema:
                        exit_reason, exit_px = "EMA", px

            if exit_reason is not None:
                exit_notional = qty * exit_px
                pnl = (exit_px - entry_px) * qty if side == "LONG" else (entry_px - exit_px) * qty
                pnl -= fee_cost(exit_notional, p.fee_bps, p.slippage_bps)
                equity += pnl

                trades.append(
                    Trade(
                        entry_ts=entry_ts,
                        exit_ts=ts,
                        side=side,
                        entry=entry_px,
                        exit=exit_px,
                        qty=qty,
                        notional=exit_notional,
                        pnl=pnl,
                        reason=exit_reason,
                    )
                )

                in_pos = False
                cooldown = p.cooldown_bars
                if cooldown > 0:
                    cooldown -= 1
                    continue
                side = None
                entry_px = None
                entry_ts = None
                qty = 0.0
                stop_px = None
                trail_px = None
                tp1_px = None
                tp1_done = False
                bars_in_trade = 0
                r_unit = 0.0

            continue

        # friction as fraction of price per roundtrip
        friction_frac = (p.fee_bps + p.slippage_bps) * 2.0 / 10_000.0
        # convert friction into R (stop_dist in price)
        friction_r = (friction_frac * px) / stop_dist
        if friction_r > (p.min_expected_r * 0.5):
            # if fees are a big chunk of your 1R, skip
            continue

        # entry logic
        if i < max(p.zwin, p.peak_win, p.burst_win, p.vel_win) or not np.isfinite(row["atr"]):
            continue
        if int(row["cascade_guard"]) == 1:
            continue
        if row["bias"] not in ("LONG", "SHORT"):
            continue
        if float(row["liq_rate_z"]) < p.min_liq_rate_z:
            continue

        # accept exhaustion OR (liq_rate_z has started falling)
        if not (int(row["exh_recent"]) == 1 or float(row["z_d1"]) < 0):
            continue

        atr = float(row["atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue

        stop_dist = p.stop_atr_mult * atr
        if (stop_dist / px) < p.min_stop_dist_pct:
            continue

        risk_usd = equity * p.risk_per_trade
        qty = risk_usd / stop_dist

        notional = qty * px
        max_notional = min(p.max_pos_notional, equity * p.leverage_cap)
        if notional > max_notional:
            qty = max_notional / px
            notional = qty * px

        if qty <= 0:
            continue

        in_pos = True
        side = row["bias"]
        entry_px = px
        entry_ts = ts
        bars_in_trade = 0
        tp1_done = False
        r_unit = stop_dist

        if side == "LONG":
            stop_px = entry_px - stop_dist
            tp1_px = entry_px + p.tp1_r * stop_dist
        else:
            stop_px = entry_px + stop_dist
            tp1_px = entry_px - p.tp1_r * stop_dist

        equity -= fee_cost(notional, p.fee_bps, p.slippage_bps)

    eq = pd.DataFrame(eq_curve, columns=["ts","equity"]).drop_duplicates("ts").reset_index(drop=True)
    if not eq.empty:
        eq["ret"] = eq["equity"].pct_change().fillna(0.0)

    stats = {}
    stats["start_equity"] = p.start_equity
    stats["end_equity"] = float(eq["equity"].iloc[-1]) if not eq.empty else p.start_equity
    stats["net_pnl"] = stats["end_equity"] - stats["start_equity"]
    stats["trades"] = len(trades)
    if stats["trades"] > 0:
        wins = sum(1 for t in trades if t.pnl > 0)
        stats["win_rate"] = wins / stats["trades"]
    else:
        stats["win_rate"] = np.nan

    if not eq.empty and eq["ret"].std(ddof=0) > 0:
        stats["sharpe"] = (eq["ret"].mean() / eq["ret"].std(ddof=0)) * math.sqrt(
            365 * 24 * 60 / infer_tf_minutes(p.tf)
        )
    else:
        stats["sharpe"] = np.nan

    if not eq.empty:
        peak = eq["equity"].cummax()
        dd = (eq["equity"] / peak) - 1.0
        stats["max_dd"] = float(dd.min())
    else:
        stats["max_dd"] = 0.0

    return eq, trades, stats

def trades_to_df(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["entry_ts","exit_ts","side","entry","exit","qty","notional","pnl","reason"])
    return pd.DataFrame([{
        "entry_ts": t.entry_ts,
        "exit_ts": t.exit_ts,
        "side": t.side,
        "entry": t.entry,
        "exit": t.exit,
        "qty": t.qty,
        "notional": t.notional,
        "pnl": t.pnl,
        "reason": t.reason,
    } for t in trades])

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, default="ETHUSDT")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--tf", type=str, default="1m", choices=["1m","3m","5m","15m","30m","1h"])
    ap.add_argument("--start_equity", type=float, default=10_000.0)
    ap.add_argument("--risk", type=float, default=0.02)
    ap.add_argument("--levcap", type=float, default=15.0)
    ap.add_argument("--fee_bps", type=float, default=4.0)
    ap.add_argument("--slip_bps", type=float, default=1.0)
    args = ap.parse_args()

    end = to_utc(pd.Timestamp.now(tz="UTC"))
    start = end - pd.Timedelta(days=int(args.days))

    p = Params(symbol=args.symbol, tf=args.tf, days=args.days)
    p.start_equity = float(args.start_equity)
    p.risk_per_trade = float(args.risk)
    p.leverage_cap = float(args.levcap)
    p.fee_bps = float(args.fee_bps)
    p.slippage_bps = float(args.slip_bps)

    p.burst_z_th = 1.8      # 3.0 is too strict for proxy z
    p.impulse_th = 0.0012   # 0.12% instead of 0.30%
    p.rate_frac = 0.55      # exhaustion condition less strict
    p.vel_decay_frac = 0.65
    p.z_accel_th = 0.60     # reduce false “cascade_guard”

    tf_min = infer_tf_minutes(p.tf)

    def min_to_bars(minutes: int) -> int:
        return max(3, int(round(minutes / tf_min)))

    # These are sane “regime” windows for bar data
    p.burst_win = min_to_bars(15)   # 15 minutes
    p.peak_win  = min_to_bars(60)   # 60 minutes peak tracking
    p.exh_win   = min_to_bars(20)   # 20 minutes exhaustion lookback
    p.vel_win   = min_to_bars(10)   # 10 minutes velocity
    p.zwin      = min_to_bars(360)  # 6 hours zscore baseline


    p.burst_win = clip(p.burst_win, 3, 60)
    p.peak_win = clip(p.peak_win, p.burst_win + 2, 120)
    p.exh_win = clip(p.exh_win, 3, 60)
    p.vel_win = clip(p.vel_win, 3, 60)

    print(f"[Data] Fetching klines {p.tf} from {start} to {end} …")
    df = fetch_klines_range(p.symbol, p.tf, start, end)
    if df.empty:
        raise SystemExit("No kline data returned.")

    df = df[["ts","open","high","low","close","volume","takerBuyBase","takerBuyQuote"]].copy()

    print("[Data] Fetching OI history (Binance only) …")
    oi_df, oi_period = fetch_oi_range_with_fallback(
        p.symbol, start, end,
        primary_period="5m",
        fallback_period="15m",
        require_oi=True,   # IMPORTANT
    )
    print(f"[Data] OI period used: {oi_period} | points={len(oi_df)}")

    d = compute_features(df, oi_df, oi_period, p)

    eq, trades, stats = backtest(d, p)
    tdf = trades_to_df(trades)

    print("\n=== RESULTS ===")
    print(f"Start equity:  ${stats['start_equity']:,.2f}")
    print(f"End equity:    ${stats['end_equity']:,.2f}")
    print(f"Net PnL:       ${stats['net_pnl']:,.2f}")
    print(f"Sharpe:        {stats['sharpe'] if np.isfinite(stats['sharpe']) else np.nan:.2f}")
    print(f"Max DD:        {stats['max_dd']*100:.2f}%")
    print(f"Trades:        {stats['trades']}")
    if np.isfinite(stats["win_rate"]):
        print(f"Win rate:      {stats['win_rate']*100:.1f}%")
    print(f"OI used:       {d.attrs.get('oi_period_used')}")

    out_prefix = f"{p.symbol}_{p.tf}_{p.days}d"
    eq_path = f"{out_prefix}_equity.csv"
    tr_path = f"{out_prefix}_trades.csv"
    feat_path = f"{out_prefix}_features.csv"

    eq.to_csv(eq_path, index=False)
    tdf.to_csv(tr_path, index=False)
    d.to_csv(feat_path, index=False)

    print("\nSaved:")
    print(" ", eq_path)
    print(" ", tr_path)
    print(" ", feat_path)

    if not tdf.empty:
        print("\nFirst 10 trades:")
        print(tdf.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
