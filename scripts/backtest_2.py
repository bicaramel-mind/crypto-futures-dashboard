#!/usr/bin/env python3
"""
ECC — ETHUSDT Extreme Crowding Cascade (VERY aggressive)
PATCHED to FIX "0 trades":
1) Robust alignment: OI and funding are RESAMPLED/REINDEXED to 15m grid (no exact-ts merge/map brittleness).
2) Adaptive extremes: use rolling percentile thresholds (top/bottom ~1.5%) instead of fixed z thresholds.
3) Aggressive gate relaxation: require (vol_compress OR near_24h_extreme), not both.
4) Built-in diagnostics: prints which conditions are binding and how many candidate bars exist.

Notes:
- OI history on Binance often limited (~30d). We run:
    * FULL window backtest (OI may be missing earlier; strategy degrades gracefully)
    * LAST_30D backtest (OI generally available)
- Risk per trade = 10% (as requested). Expect huge variance.

Install:
  pip install pandas numpy httpx

Run:
  python ecc_ethusdt_backtest.py

Outputs:
  ecc_ethusdt_equity_curve_full.csv
  ecc_ethusdt_trades_full.csv
  ecc_ethusdt_equity_curve_30d.csv
  ecc_ethusdt_trades_30d.csv
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

REST = "https://fapi.binance.com"

SYMBOL = "ETHUSDT"
INTERVAL = "15m"
TF_15M = "15min"
TF_8H = "8H"

# Windows
DAYS_MAIN = 90
DAYS_OI_MAX = 30

# Very aggressive sizing
START_EQUITY = 1000.0
RISK_PER_TRADE = 0.1  # 10% risk per trade (requested)
MAX_EFFECTIVE_LEV = 80.0  # cap to avoid insane sizing from tiny stops

# Costs (simple)
TAKER_FEE = 0.0004
SLIPPAGE_BP = 2.0  # per side

# Lookbacks
LOOKBACK_BASIS_15M = 192
LOOKBACK_OI_15M = 192
LOOKBACK_FUND_8H = 21
VOL_WINDOW_15M = 96  # 24h

# Stops & exits
STOP_MIN_PCT = 0.007
STOP_ATR_MULT = 2
TP1_PCT = 0.1
TRAIL_PCT = 0.015

# Adaptive extreme window
QWIN = 7 * 24 * 4  # 7 days of 15m bars
QHI = 0.985
QLO = 0.015

# Entry helpers
NEAR_EXTREME_PCT = 0.015  # 1.5% of 24h high/low
EXIT_NORMALIZE_FUND_Z = 1.0
EXIT_OI_COLLAPSE_Z = 0.5


def now_ms() -> int:
    return int(time.time() * 1000)


def ms_days_ago(days: int) -> int:
    return now_ms() - days * 24 * 60 * 60 * 1000


def rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    if x.empty:
        return x
    minp = max(10, window // 3)
    m = x.rolling(window, min_periods=minp).mean()
    s = x.rolling(window, min_periods=minp).std(ddof=0).replace(0, np.nan)
    return (x - m) / s


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def fetch_paginated_klines(client: httpx.Client, endpoint: str, params: Dict, start_ms: int, end_ms: int, limit: int = 1500) -> List:
    out = []
    cur = start_ms
    while True:
        p = dict(params)
        p["startTime"] = cur
        p["endTime"] = end_ms
        p["limit"] = limit
        r = client.get(f"{REST}{endpoint}", params=p, timeout=30.0)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        out.extend(data)
        last_open = int(data[-1][0])
        nxt = last_open + 1
        if nxt <= cur:
            break
        cur = nxt
        if last_open >= end_ms:
            break
        if len(data) < 5:
            break
    return out


def fetch_funding_rate_events(client: httpx.Client, symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> pd.DataFrame:
    rows = []
    cur = start_ms
    while True:
        params = {"symbol": symbol, "startTime": cur, "endTime": end_ms, "limit": limit}
        r = client.get(f"{REST}/fapi/v1/fundingRate", params=params, timeout=30.0)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        rows.extend(data)
        last_ts = int(data[-1]["fundingTime"])
        nxt = last_ts + 1
        if nxt <= cur:
            break
        cur = nxt
        if last_ts >= end_ms:
            break
        if len(data) < 5:
            break
    if not rows:
        return pd.DataFrame(columns=["ts", "fundingRate"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    return df[["ts", "fundingRate"]].sort_values("ts").drop_duplicates("ts")


def fetch_open_interest_hist(client: httpx.Client, symbol: str, period: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    out = []
    cur = start_ms
    while True:
        params = {"symbol": symbol, "period": period, "startTime": cur, "endTime": end_ms, "limit": 500}
        r = client.get(f"{REST}/futures/data/openInterestHist", params=params, timeout=30.0)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        out.extend(data)
        last_ts = int(data[-1]["timestamp"])
        nxt = last_ts + 1
        if nxt <= cur:
            break
        cur = nxt
        if last_ts >= end_ms:
            break
        if len(data) < 10:
            break
    df = pd.DataFrame(out)
    if df.empty:
        return pd.DataFrame(columns=["ts", "oi_last"])
    df["ts"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df["oi_last"] = df["sumOpenInterest"].astype(float)
    return df[["ts", "oi_last"]].sort_values("ts").drop_duplicates("ts")


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=max(5, period // 2)).mean()


@dataclass
class Trade:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    side: str
    entry: float
    exit: float
    size_notional: float
    pnl_usd: float
    reason: str


def apply_costs(notional: float) -> float:
    slip = SLIPPAGE_BP / 10_000.0
    return notional * (2 * TAKER_FEE + 2 * slip)


def load_data(days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start_ms = ms_days_ago(days)
    end_ms = now_ms()
    oi_start_ms = max(start_ms, ms_days_ago(DAYS_OI_MAX))

    with httpx.Client() as client:
        fut = fetch_paginated_klines(
            client, "/fapi/v1/klines",
            {"symbol": SYMBOL, "interval": INTERVAL},
            start_ms, end_ms
        )
        df_trade = pd.DataFrame(
            fut,
            columns=["openTime","open","high","low","close","volume","closeTime","qv","n","tbbv","tbqv","ignore"]
        )
        df_trade["ts"] = pd.to_datetime(df_trade["openTime"].astype(int), unit="ms", utc=True)
        for c in ["open","high","low","close"]:
            df_trade[c] = df_trade[c].astype(float)
        df_trade = df_trade[["ts","open","high","low","close"]].sort_values("ts").drop_duplicates("ts")

        mk = fetch_paginated_klines(
            client, "/fapi/v1/markPriceKlines",
            {"symbol": SYMBOL, "interval": INTERVAL},
            start_ms, end_ms
        )
        df_mk = pd.DataFrame(
            mk,
            columns=["openTime","open","high","low","close","volume","closeTime","qv","n","tbbv","tbqv","ignore"]
        )
        df_mk["ts"] = pd.to_datetime(df_mk["openTime"].astype(int), unit="ms", utc=True)
        df_mk["mark_close"] = df_mk["close"].astype(float)
        df_mk = df_mk[["ts","mark_close"]].sort_values("ts").drop_duplicates("ts")

        ix = fetch_paginated_klines(
            client, "/fapi/v1/indexPriceKlines",
            {"pair": SYMBOL, "interval": INTERVAL},
            start_ms, end_ms
        )
        df_ix = pd.DataFrame(
            ix,
            columns=["openTime","open","high","low","close","volume","closeTime","qv","n","tbbv","tbqv","ignore"]
        )
        df_ix["ts"] = pd.to_datetime(df_ix["openTime"].astype(int), unit="ms", utc=True)
        df_ix["index_close"] = df_ix["close"].astype(float)
        df_ix = df_ix[["ts","index_close"]].sort_values("ts").drop_duplicates("ts")

        df_fr = fetch_funding_rate_events(client, SYMBOL, start_ms, end_ms)
        df_oi = fetch_open_interest_hist(client, SYMBOL, INTERVAL, oi_start_ms, end_ms)

    # Base frame
    df_feat = df_trade.merge(df_mk, on="ts", how="inner").merge(df_ix, on="ts", how="inner")
    df_feat = df_feat.sort_values("ts").reset_index(drop=True)
    idx = pd.DatetimeIndex(df_feat["ts"])

    # --- Robust funding alignment (REINDEX + FFILL) ---
    if df_fr.empty:
        df_feat["funding_last"] = np.nan
        df_feat["funding_z"] = np.nan
    else:
        s15 = df_fr.set_index("ts")["fundingRate"].sort_index().resample(TF_15M).last().ffill()
        s8 = df_fr.set_index("ts")["fundingRate"].sort_index().resample(TF_8H).last().ffill()
        z8 = rolling_zscore(s8, LOOKBACK_FUND_8H)
        z15 = z8.resample(TF_15M).ffill()

        df_feat["funding_last"] = s15.reindex(idx, method="ffill").to_numpy()
        df_feat["funding_z"] = z15.reindex(idx, method="ffill").to_numpy()

    # --- Robust OI alignment (RESAMPLE to 15m + FFILL) ---
    if df_oi.empty:
        df_feat["oi_last"] = np.nan
    else:
        oi15 = df_oi.set_index("ts")["oi_last"].sort_index().resample(TF_15M).last().ffill()
        df_feat["oi_last"] = oi15.reindex(idx, method="ffill").to_numpy()

    return df_feat, df_fr


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values("ts").reset_index(drop=True)

    d["basis"] = (d["mark_close"] - d["index_close"]) / d["index_close"].replace(0, np.nan)
    d["basis_z"] = rolling_zscore(d["basis"], LOOKBACK_BASIS_15M)

    d["ret"] = d["close"].pct_change()
    d["vol15"] = d["ret"].rolling(VOL_WINDOW_15M, min_periods=30).std()
    d["vol_med24h"] = d["vol15"].rolling(VOL_WINDOW_15M, min_periods=30).median()
    d["vol_compress"] = d["vol15"] < d["vol_med24h"]

    d["hi24"] = d["high"].rolling(VOL_WINDOW_15M, min_periods=30).max()
    d["lo24"] = d["low"].rolling(VOL_WINDOW_15M, min_periods=30).min()
    d["near_hi24"] = (d["hi24"] - d["close"]) / d["hi24"].replace(0, np.nan) <= NEAR_EXTREME_PCT
    d["near_lo24"] = (d["close"] - d["lo24"]) / d["lo24"].replace(0, np.nan) <= NEAR_EXTREME_PCT

    # Exhaustion proxy (weakening momentum)
    d["ret_1"] = d["close"].pct_change(1)
    d["ret_2"] = d["close"].pct_change(2)
    d["mom_weak_up"] = (d["ret_2"] > 0) & (d["ret_1"] > 0) & (d["ret_1"] < d["ret_2"])
    d["mom_weak_dn"] = (d["ret_2"] < 0) & (d["ret_1"] < 0) & (d["ret_1"] > d["ret_2"])

    d["atr"] = compute_atr(d, 14)

    # OI notional features
    d["oi_notional"] = d["oi_last"] * d["mark_close"]
    d["oi_notional_chg"] = d["oi_notional"].diff()
    d["oi_notional_z"] = rolling_zscore(d["oi_notional_chg"], LOOKBACK_OI_15M)

    # Adaptive thresholds (rolling percentiles)
    d["fund_hi"] = d["funding_z"].rolling(QWIN, min_periods=200).quantile(QHI)
    d["fund_lo"] = d["funding_z"].rolling(QWIN, min_periods=200).quantile(QLO)
    d["basis_hi"] = d["basis_z"].rolling(QWIN, min_periods=200).quantile(QHI)
    d["basis_lo"] = d["basis_z"].rolling(QWIN, min_periods=200).quantile(QLO)
    d["oi_hi"] = d["oi_notional_z"].rolling(QWIN, min_periods=200).quantile(QHI)

    return d


def ecc_entry_side(row: pd.Series) -> Optional[str]:
    fz = row.get("funding_z", np.nan)
    bz = row.get("basis_z", np.nan)
    oz = row.get("oi_notional_z", np.nan)

    if not (np.isfinite(fz) and np.isfinite(bz)):
        return None

    # Adaptive extremes
    fund_hi = row.get("fund_hi", np.nan)
    fund_lo = row.get("fund_lo", np.nan)
    basis_hi = row.get("basis_hi", np.nan)
    basis_lo = row.get("basis_lo", np.nan)

    short_crowd = (np.isfinite(fund_hi) and np.isfinite(basis_hi) and (fz >= fund_hi) and (bz >= basis_hi))
    long_crowd = (np.isfinite(fund_lo) and np.isfinite(basis_lo) and (fz <= fund_lo) and (bz <= basis_lo))

    # OI: if available, require extreme; if not available, allow (degraded)
    oi_hi = row.get("oi_hi", np.nan)
    oi_ok = (np.isfinite(oz) and np.isfinite(oi_hi) and (oz >= oi_hi)) or (not np.isfinite(oz))

    # Aggressive relaxation: require (vol_compress OR near extreme)
    vol_ok = bool(row.get("vol_compress", False))
    near_hi = bool(row.get("near_hi24", False))
    near_lo = bool(row.get("near_lo24", False))
    loc_ok = vol_ok or near_hi or near_lo

    if not (oi_ok and loc_ok):
        return None

    # Timing: exhaustion + location
    if short_crowd and near_hi and bool(row.get("mom_weak_up", False)):
        return "SHORT"
    if long_crowd and near_lo and bool(row.get("mom_weak_dn", False)):
        return "LONG"

    return None


def diag_counts(d: pd.DataFrame, name: str) -> None:
    x = d.copy()
    # Finite checks
    fin_f = np.isfinite(x["funding_z"]).sum()
    fin_b = np.isfinite(x["basis_z"]).sum()
    fin_o = np.isfinite(x["oi_notional_z"]).sum()

    short_crowd = (np.isfinite(x["fund_hi"]) & np.isfinite(x["basis_hi"]) &
                   (x["funding_z"] >= x["fund_hi"]) & (x["basis_z"] >= x["basis_hi"]))
    long_crowd = (np.isfinite(x["fund_lo"]) & np.isfinite(x["basis_lo"]) &
                  (x["funding_z"] <= x["fund_lo"]) & (x["basis_z"] <= x["basis_lo"]))

    oi_ok = ((np.isfinite(x["oi_notional_z"]) & np.isfinite(x["oi_hi"]) & (x["oi_notional_z"] >= x["oi_hi"])) |
             (~np.isfinite(x["oi_notional_z"])))

    loc_ok = x["vol_compress"].fillna(False) | x["near_hi24"].fillna(False) | x["near_lo24"].fillna(False)

    short_timing = x["near_hi24"].fillna(False) & x["mom_weak_up"].fillna(False)
    long_timing = x["near_lo24"].fillna(False) & x["mom_weak_dn"].fillna(False)

    short_all = short_crowd & oi_ok & loc_ok & short_timing
    long_all = long_crowd & oi_ok & loc_ok & long_timing

    print(f"\n=== DIAG {name} ===")
    print("rows:", len(x))
    print("funding_z finite:", int(fin_f))
    print("basis_z finite:", int(fin_b))
    print("oi_notional_z finite:", int(fin_o))
    print("short_crowd:", int(short_crowd.sum()), "long_crowd:", int(long_crowd.sum()))
    print("oi_ok:", int(oi_ok.sum()), "loc_ok:", int(loc_ok.sum()))
    print("short_timing:", int(short_timing.sum()), "long_timing:", int(long_timing.sum()))
    print("SHORT ALL:", int(short_all.sum()), "LONG ALL:", int(long_all.sum()))
    print("===================")


def backtest(df: pd.DataFrame, df_funding_events: pd.DataFrame) -> Tuple[pd.DataFrame, List[Trade]]:
    equity = START_EQUITY
    pos = 0
    entry_px = 0.0
    entry_ts = None
    size_notional = 0.0
    stop_px = 0.0
    took_tp1 = False
    trail_active = False
    trail_px = 0.0

    trades: List[Trade] = []
    rows = []

    # Funding events mapped to 15m bar timestamps
    fund_map = {}
    if not df_funding_events.empty:
        fr = df_funding_events.copy()
        fr["bar_ts"] = fr["ts"].dt.floor(TF_15M)
        fund_map = fr.groupby("bar_ts")["fundingRate"].last().to_dict()

    for i in range(200, len(df)):  # warmup for quantiles
        r = df.iloc[i]
        ts = r["ts"]
        px_open = float(r["open"])
        px_close = float(r["close"])
        hi = float(r["high"])
        lo = float(r["low"])

        # Funding cashflow at this bar if event hits
        funding_pnl = 0.0
        if ts in fund_map and pos != 0 and size_notional > 0:
            fr = float(fund_map[ts])
            # long pays when fr>0, short receives when fr>0
            funding_pnl = (-pos * fr) * size_notional
            equity += funding_pnl

        rows.append({
            "ts": ts,
            "equity": equity,
            "pos": pos,
            "entry_px": entry_px if pos != 0 else np.nan,
            "stop_px": stop_px if pos != 0 else np.nan,
            "trail_px": trail_px if trail_active else np.nan,
            "size_notional": size_notional if pos != 0 else 0.0,
            "funding_pnl": funding_pnl,
            "close": px_close,
        })

        # Manage open position
        if pos != 0:
            # update trailing stop (after TP1)
            if trail_active:
                if pos == 1:
                    trail_px = max(trail_px, px_close * (1 - TRAIL_PCT))
                else:
                    trail_px = min(trail_px, px_close * (1 + TRAIL_PCT))

            # TP1: take 50% at +6%
            if not took_tp1:
                if pos == 1 and (hi - entry_px) / entry_px >= TP1_PCT:
                    exit_px = entry_px * (1 + TP1_PCT)
                    pnl = (exit_px - entry_px) / entry_px * (size_notional * 0.5)
                    fee = apply_costs(size_notional * 0.5)
                    equity += pnl - fee
                    took_tp1 = True
                    trail_active = True
                    trail_px = px_close * (1 - TRAIL_PCT)
                elif pos == -1 and (entry_px - lo) / entry_px >= TP1_PCT:
                    exit_px = entry_px * (1 - TP1_PCT)
                    pnl = (entry_px - exit_px) / entry_px * (size_notional * 0.5)
                    fee = apply_costs(size_notional * 0.5)
                    equity += pnl - fee
                    took_tp1 = True
                    trail_active = True
                    trail_px = px_close * (1 + TRAIL_PCT)

            # Stops
            hit_stop = False
            exit_px = None

            if pos == 1 and lo <= stop_px:
                hit_stop = True
                exit_px = stop_px
            elif pos == -1 and hi >= stop_px:
                hit_stop = True
                exit_px = stop_px

            hit_trail = False
            if trail_active and exit_px is None:
                if pos == 1 and lo <= trail_px:
                    hit_trail = True
                    exit_px = trail_px
                elif pos == -1 and hi >= trail_px:
                    hit_trail = True
                    exit_px = trail_px

            # Optional state exits on remainder (after TP1)
            state_exit = False
            if trail_active and exit_px is None:
                fz = float(r.get("funding_z", np.nan))
                oz = float(r.get("oi_notional_z", np.nan))
                if np.isfinite(fz) and abs(fz) < EXIT_NORMALIZE_FUND_Z:
                    state_exit = True
                    exit_px = px_close
                if np.isfinite(oz) and oz < EXIT_OI_COLLAPSE_Z:
                    state_exit = True
                    exit_px = px_close

            if exit_px is not None:
                rem_notional = size_notional * (0.5 if took_tp1 else 1.0)
                if pos == 1:
                    pnl = (exit_px - entry_px) / entry_px * rem_notional
                else:
                    pnl = (entry_px - exit_px) / entry_px * rem_notional
                fee = apply_costs(rem_notional)
                equity += pnl - fee

                reason = "STOP" if hit_stop else "TRAIL" if hit_trail else "STATE" if state_exit else "EXIT"
                trades.append(Trade(
                    entry_ts=entry_ts,
                    exit_ts=ts,
                    side="LONG" if pos == 1 else "SHORT",
                    entry=float(entry_px),
                    exit=float(exit_px),
                    size_notional=float(size_notional),
                    pnl_usd=float(pnl - fee),
                    reason=reason,
                ))

                pos = 0
                entry_px = 0.0
                entry_ts = None
                size_notional = 0.0
                stop_px = 0.0
                took_tp1 = False
                trail_active = False
                trail_px = 0.0

        # Enter (use current bar features; fill at next bar open approx)
        if pos == 0:
            side = ecc_entry_side(r)
            if side is not None:
                atr = float(r.get("atr", np.nan))
                if not np.isfinite(atr) or atr <= 0:
                    continue

                stop_dist = max(STOP_MIN_PCT, STOP_ATR_MULT * (atr / px_close))
                stop_dist = clamp(stop_dist, 0.004, 0.06)

                risk_usd = equity * RISK_PER_TRADE
                notional = risk_usd / stop_dist
                eff_lev = notional / max(equity, 1e-9)
                if eff_lev > MAX_EFFECTIVE_LEV:
                    notional = MAX_EFFECTIVE_LEV * equity

                entry_px = px_open
                entry_ts = ts
                size_notional = float(notional)

                if side == "LONG":
                    pos = 1
                    stop_px = entry_px * (1 - stop_dist)
                else:
                    pos = -1
                    stop_px = entry_px * (1 + stop_dist)

    out = pd.DataFrame(rows)
    return out, trades


def summarize(eq: pd.Series) -> Dict[str, float]:
    eq = eq.dropna()
    rets = eq.pct_change().fillna(0.0)
    bars_per_year = 365 * 24 * 4
    mu = rets.mean()
    sd = rets.std(ddof=0)
    sharpe = float("nan") if sd == 0 else float((mu / sd) * math.sqrt(bars_per_year))
    peak = eq.cummax()
    dd = (eq - peak) / peak.replace(0, np.nan)
    return {
        "end_equity": float(eq.iloc[-1]) if len(eq) else float("nan"),
        "net_pnl": float(eq.iloc[-1] - eq.iloc[0]) if len(eq) else float("nan"),
        "sharpe": sharpe,
        "max_dd_pct": float(dd.min() * 100.0) if len(dd) else 0.0,
    }


def main():
    print(f"Loading {SYMBOL} {INTERVAL} data…")
    df_raw, df_fr = load_data(DAYS_MAIN)
    df = compute_features(df_raw)

    # Diagnostics
    diag_counts(df, "FULL")

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=DAYS_OI_MAX)
    df_30 = df[df["ts"] >= cutoff].copy()
    diag_counts(df_30, "LAST_30D")

    # Backtest full
    print("\nRunning ECC backtest (FULL window; OI may be missing earlier)…")
    eq_full, trades_full = backtest(df, df_fr)
    sf = summarize(eq_full["equity"])
    print("===== ECC FULL WINDOW =====")
    print(f"End equity:  ${sf['end_equity']:.2f}")
    print(f"Net PnL:     ${sf['net_pnl']:.2f}")
    print(f"Sharpe:      {sf['sharpe']:.2f}")
    print(f"Max DD:      {sf['max_dd_pct']:.2f}%")
    print(f"Trades:      {len(trades_full)}")
    print("===========================")
    eq_full.to_csv("ecc_ethusdt_equity_curve_full.csv", index=False)
    pd.DataFrame([t.__dict__ for t in trades_full]).to_csv("ecc_ethusdt_trades_full.csv", index=False)

    # Backtest last 30d
    print("\nRunning ECC backtest (LAST ~30 DAYS; OI usually present)…")
    eq_30, trades_30 = backtest(df_30, df_fr)
    s30 = summarize(eq_30["equity"])
    print("===== ECC LAST_30D =====")
    print(f"End equity:  ${s30['end_equity']:.2f}")
    print(f"Net PnL:     ${s30['net_pnl']:.2f}")
    print(f"Sharpe:      {s30['sharpe']:.2f}")
    print(f"Max DD:      {s30['max_dd_pct']:.2f}%")
    print(f"Trades:      {len(trades_30)}")
    print("========================")
    eq_30.to_csv("ecc_ethusdt_equity_curve_30d.csv", index=False)
    pd.DataFrame([t.__dict__ for t in trades_30]).to_csv("ecc_ethusdt_trades_30d.csv", index=False)

    print("\nSaved CSVs:")
    print("  ecc_ethusdt_equity_curve_full.csv")
    print("  ecc_ethusdt_trades_full.csv")
    print("  ecc_ethusdt_equity_curve_30d.csv")
    print("  ecc_ethusdt_trades_30d.csv")


if __name__ == "__main__":
    main()
