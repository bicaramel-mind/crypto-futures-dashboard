#!/usr/bin/env python3
"""
Improved backtest for ETHUSDT perp strategy (15m) with:
1) OI used as SOFT confirmation (no hard gate that filters trades away).
2) OI normalized to NOTIONAL: oi_notional = sumOpenInterest * mark_close.
3) Funding handled as a SLOW regime signal (z-score computed on 8h grid, then forward-filled to 15m).
4) Funding PnL INCLUDED (cashflows at funding timestamps).

Binance USDⓈ-M endpoints used:
- Futures klines (execution):        GET /fapi/v1/klines
- Mark price klines (signals):       GET /fapi/v1/markPriceKlines
- Index price klines (basis):        GET /fapi/v1/indexPriceKlines   (NOTE: pair= not symbol=)
- Open interest history (signals):   GET /futures/data/openInterestHist (max lookback ~30d, max limit 500)
- Funding rate events:               GET /fapi/v1/fundingRate

Important limitation:
- openInterestHist only provides ~last 30 days. For a 3-month backtest:
  - We merge OI for last 30d and leave earlier OI as NaN.
  - When OI features are NaN, the strategy still runs (OI confirmation just contributes 0).

Execution assumptions:
- Compute signal on bar t close
- Enter/flip at bar t+1 open (no lookahead)
- Held signal: once LONG/SHORT chosen, stays until opposite entry condition triggers
  (no "neutral" flips)

Run:
  pip install pandas numpy httpx
  python backtest_ethusdt_15m_improved.py

Outputs:
- backtest_results_ethusdt_15m_improved.csv
- prints summary for full ~3 months and last 30 days
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

REST_BASE = "https://fapi.binance.com"

SYMBOL = "ETHUSDT"
INTERVAL = "15m"
TF_15M = "15min"
TF_8H = "8H"

# Backtest window
DAYS_BACK_MAIN = 92      # ~3 months
DAYS_BACK_OI_MAX = 30    # OI availability constraint

# Capital / costs
CAPITAL_USD = 10_000.0
LEVERAGE = 1.0
TAKER_FEE = 0.0004       # 0.04% per side
SLIPPAGE_BP = 1.0        # per entry/exit

# Indicator lookbacks
LOOKBACK_BASIS_15M = 192     # 2 days of 15m bars
LOOKBACK_OI_15M = 192        # 2 days of 15m bars (only effective in last 30d)
LOOKBACK_FUND_8H = 21        # ~7 days on 8h grid (3 points/day)

# Entry / flip thresholds (tuned to be less brittle than before)
ENTRY_SCORE = 1.4            # score needed to enter from FLAT
FLIP_SCORE = 1.6             # score magnitude needed to flip from LONG to SHORT or vice versa
# If you want fewer trades: raise these (e.g., 1.8 / 2.0)

# Feature weights (OI is soft confirmation)
W_BASIS = 0.7
W_FUND = 0.9
W_OI = 0.4
W_MOM = 0.5


# -----------------------------
# Utilities
# -----------------------------
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


# -----------------------------
# Binance pagination
# -----------------------------
def fetch_paginated_klines(
    client: httpx.Client,
    endpoint: str,
    params: Dict,
    start_ms: int,
    end_ms: int,
    limit: int = 1500,
) -> List:
    out: List = []
    cur = start_ms
    while True:
        p = dict(params)
        p["startTime"] = cur
        p["endTime"] = end_ms
        p["limit"] = limit

        r = client.get(f"{REST_BASE}{endpoint}", params=p, timeout=20.0)
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        out.extend(data)
        last_open = int(data[-1][0])
        next_cur = last_open + 1
        if next_cur <= cur:
            break
        cur = next_cur
        if last_open >= end_ms:
            break
        if len(data) < 5:
            break
    return out


def fetch_paginated_open_interest_hist(
    client: httpx.Client,
    symbol: str,
    period: str,
    start_ms: int,
    end_ms: int,
) -> List[dict]:
    """
    openInterestHist:
    - max limit = 500
    - data only for latest ~30 days / 1 month
    """
    out: List[dict] = []
    cur = start_ms
    limit = 500

    while True:
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": cur,
            "endTime": end_ms,
            "limit": limit,
        }
        r = client.get(f"{REST_BASE}/futures/data/openInterestHist", params=params, timeout=20.0)
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        out.extend(data)
        last_ts = int(data[-1]["timestamp"])
        next_cur = last_ts + 1
        if next_cur <= cur:
            break
        cur = next_cur
        if last_ts >= end_ms:
            break
        if len(data) < 10:
            break

    return out


def fetch_funding_rate_events(
    client: httpx.Client,
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
) -> pd.DataFrame:
    rows: List[dict] = []
    cur = start_ms
    while True:
        params = {"symbol": symbol, "startTime": cur, "endTime": end_ms, "limit": limit}
        r = client.get(f"{REST_BASE}/fapi/v1/fundingRate", params=params, timeout=20.0)
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        rows.extend(data)
        last_ts = int(data[-1]["fundingTime"])
        next_cur = last_ts + 1
        if next_cur <= cur:
            break
        cur = next_cur
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


# -----------------------------
# Load data
# -----------------------------
def load_all_data(symbol: str, days_back: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_feat: 15m feature table (mark_close, index_close, basis, oi_notional (last 30d), funding_8h_z ffilled to 15m)
      df_trade: 15m futures execution prices (open/close)
      df_funding_events: discrete funding events for cashflows
    """
    start_ms = ms_days_ago(days_back)
    end_ms = now_ms()

    # clamp OI window to last 30 days
    oi_start_ms = max(start_ms, ms_days_ago(DAYS_BACK_OI_MAX))

    with httpx.Client() as client:
        # Execution klines
        fut = fetch_paginated_klines(
            client,
            endpoint="/fapi/v1/klines",
            params={"symbol": symbol, "interval": INTERVAL},
            start_ms=start_ms,
            end_ms=end_ms,
            limit=1500,
        )
        df_trade = pd.DataFrame(
            fut,
            columns=["openTime","open","high","low","close","volume","closeTime","qv","n","tbbv","tbqv","ignore"],
        )
        df_trade["ts"] = pd.to_datetime(df_trade["openTime"].astype(int), unit="ms", utc=True)
        df_trade["open"] = df_trade["open"].astype(float)
        df_trade["close"] = df_trade["close"].astype(float)
        df_trade = df_trade[["ts","open","close"]].sort_values("ts").drop_duplicates("ts")

        # Mark price klines
        mk = fetch_paginated_klines(
            client,
            endpoint="/fapi/v1/markPriceKlines",
            params={"symbol": symbol, "interval": INTERVAL},
            start_ms=start_ms,
            end_ms=end_ms,
            limit=1500,
        )
        df_mk = pd.DataFrame(
            mk,
            columns=["openTime","open","high","low","close","volume","closeTime","qv","n","tbbv","tbqv","ignore"],
        )
        df_mk["ts"] = pd.to_datetime(df_mk["openTime"].astype(int), unit="ms", utc=True)
        df_mk["mark_close"] = df_mk["close"].astype(float)
        df_mk = df_mk[["ts","mark_close"]].sort_values("ts").drop_duplicates("ts")

        # Index price klines (pair=)
        ix = fetch_paginated_klines(
            client,
            endpoint="/fapi/v1/indexPriceKlines",
            params={"pair": symbol, "interval": INTERVAL},
            start_ms=start_ms,
            end_ms=end_ms,
            limit=1500,
        )
        df_ix = pd.DataFrame(
            ix,
            columns=["openTime","open","high","low","close","volume","closeTime","qv","n","tbbv","tbqv","ignore"],
        )
        df_ix["ts"] = pd.to_datetime(df_ix["openTime"].astype(int), unit="ms", utc=True)
        df_ix["index_close"] = df_ix["close"].astype(float)
        df_ix = df_ix[["ts","index_close"]].sort_values("ts").drop_duplicates("ts")

        # Open interest (last 30d only)
        oi_rows = fetch_paginated_open_interest_hist(client, symbol=symbol, period=INTERVAL, start_ms=oi_start_ms, end_ms=end_ms)
        df_oi = pd.DataFrame(oi_rows)
        if df_oi.empty:
            df_oi = pd.DataFrame(columns=["ts","oi_last"])
        else:
            df_oi["ts"] = pd.to_datetime(df_oi["timestamp"].astype(int), unit="ms", utc=True)
            df_oi["oi_last"] = df_oi["sumOpenInterest"].astype(float)
            df_oi = df_oi[["ts","oi_last"]].sort_values("ts").drop_duplicates("ts")

        # Funding events (for pnl + regime)
        df_fr = fetch_funding_rate_events(client, symbol=symbol, start_ms=start_ms, end_ms=end_ms)

    # Merge base features
    df_feat = df_mk.merge(df_ix, on="ts", how="inner").merge(df_oi, on="ts", how="left")
    df_feat = df_feat.sort_values("ts").reset_index(drop=True)

    # Basis
    df_feat["basis_close"] = (df_feat["mark_close"] - df_feat["index_close"]) / df_feat["index_close"].replace(0, np.nan)

    # Momentum (15m)
    df_feat["ret_1"] = df_feat["mark_close"].pct_change()
    df_feat["ret_4"] = df_feat["mark_close"].pct_change(4)   # 1h drift

    # OI notional (soft confirmation)
    df_feat["oi_notional"] = df_feat["oi_last"] * df_feat["mark_close"]
    df_feat["oi_notional_chg"] = df_feat["oi_notional"].diff()

    # Funding regime: compute funding z-score on 8H grid, ffill to 15m
    if df_fr.empty:
        df_feat["funding_last"] = np.nan
        df_feat["funding_z"] = np.nan
    else:
        # funding_last (piecewise constant) at 15m for reporting, but z-score computed on 8h
        s15 = df_fr.set_index("ts")["fundingRate"].sort_index().resample(TF_15M).last().ffill()
        df_feat["funding_last"] = df_feat["ts"].map(s15).astype(float)

        s8 = df_fr.set_index("ts")["fundingRate"].sort_index().resample(TF_8H).last().ffill()
        z8 = rolling_zscore(s8, LOOKBACK_FUND_8H)
        z15 = z8.resample(TF_15M).ffill()
        df_feat["funding_z"] = df_feat["ts"].map(z15).astype(float)

    return df_feat, df_trade, df_fr


# -----------------------------
# Indicators + signal (OI is SOFT)
# -----------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values("ts").reset_index(drop=True)

    d["basis_z"] = rolling_zscore(d["basis_close"], LOOKBACK_BASIS_15M)

    # OI z-score only where available; elsewhere NaN -> contributes 0 to score
    d["oi_notional_z"] = rolling_zscore(d["oi_notional_chg"], LOOKBACK_OI_15M)

    # Momentum proxy: use ret_4 (1h) as mild directional prior
    d["mom"] = d["ret_4"].fillna(0.0)

    return d


def score_row(r: pd.Series) -> float:
    """
    Positive score => LONG bias, negative => SHORT bias.
    - Funding: contrarian regime component (very positive funding => long crowding => short bias)
    - Basis:   contrarian crowding component
    - OI:      soft confirmation: only adds when it agrees with direction
    - Momentum: mild trend prior (prevents always fading)
    """
    fz = float(r.get("funding_z", np.nan))
    bz = float(r.get("basis_z", np.nan))
    oz = float(r.get("oi_notional_z", np.nan))
    mom = float(r.get("mom", 0.0))

    # Contrarian components (crowding)
    # If funding_z is +, crowding long => score more negative
    fund_term = 0.0 if not np.isfinite(fz) else (-fz)
    basis_term = 0.0 if not np.isfinite(bz) else (-bz)

    # Momentum term (small)
    mom_term = clamp(mom * 10.0, -2.0, 2.0)  # scale 1h return into rough z-like range

    # OI term: only reinforces when it aligns (soft confirmation)
    # High positive OI change means "participation / fuel". We use it to reinforce the CURRENT score direction.
    oi_term = 0.0
    if np.isfinite(oz):
        # map oz to [-2,2] for stability
        oz_c = clamp(oz, -2.0, 2.0)
        # if crowding score suggests long (positive), and oz positive, reinforce; if suggests short and oz positive, reinforce too
        # i.e. OI is fuel for moves; we don't use its sign as direction, only as confidence in the direction we already have.
        oi_term = abs(oz_c)

    # Base score from crowding + momentum
    base = W_FUND * fund_term + W_BASIS * basis_term + W_MOM * mom_term

    # Apply oi_term as magnitude booster (soft confirmation)
    score = base + W_OI * (oi_term if abs(base) > 0.3 else 0.0)

    # Mild saturation to avoid single-factor blowups
    return float(clamp(score, -6.0, 6.0))


def held_position_from_scores(scores: pd.Series) -> pd.Series:
    """
    Stateful rule:
    - Start FLAT
    - Enter LONG if score >= ENTRY_SCORE
    - Enter SHORT if score <= -ENTRY_SCORE
    - If LONG, only flip to SHORT if score <= -FLIP_SCORE
    - If SHORT, only flip to LONG if score >= FLIP_SCORE
    - Otherwise HOLD last position (no neutral churn)
    """
    pos = 0
    out = []
    for s in scores:
        if pos == 0:
            if s >= ENTRY_SCORE:
                pos = 1
            elif s <= -ENTRY_SCORE:
                pos = -1
        elif pos == 1:
            if s <= -FLIP_SCORE:
                pos = -1
        elif pos == -1:
            if s >= FLIP_SCORE:
                pos = 1
        out.append(pos)
    return pd.Series(out, index=scores.index, name="pos_signal")


# -----------------------------
# Backtest + funding pnl
# -----------------------------
def backtest(df_feat: pd.DataFrame, df_trade: pd.DataFrame, df_funding_events: pd.DataFrame) -> pd.DataFrame:
    df = df_feat.merge(df_trade, on="ts", how="inner").sort_values("ts").reset_index(drop=True)
    df = compute_indicators(df)

    df["score"] = df.apply(score_row, axis=1)
    df["pos_signal"] = held_position_from_scores(df["score"])

    # Execute at next bar open
    df["pos_exec"] = df["pos_signal"].shift(1).fillna(0).astype(int)

    # Price return (close-to-close)
    df["ret_bar"] = df["close"].pct_change().fillna(0.0)
    df["pnl_gross_1x"] = df["pos_exec"] * df["ret_bar"]

    # Trading costs on changes in position
    df["pos_change"] = df["pos_exec"].diff().fillna(df["pos_exec"]).abs()
    slip = SLIPPAGE_BP / 10_000.0
    df["cost_1x"] = df["pos_change"] * (TAKER_FEE + slip)

    df["pnl_net_1x_ex_funding"] = df["pnl_gross_1x"] - df["cost_1x"]

    # Funding cashflows
    df["funding_cf_1x"] = 0.0
    if not df_funding_events.empty:
        fr = df_funding_events.copy()
        fr["bar_ts"] = fr["ts"].dt.floor(TF_15M)

        pos_map = df.set_index("ts")["pos_exec"]
        fr["pos"] = fr["bar_ts"].map(pos_map).fillna(0).astype(int)

        # Funding cashflow per $1 notional:
        # long pays when fundingRate > 0; short receives.
        fr["cf_1x"] = -fr["pos"] * fr["fundingRate"].astype(float)

        cf_bar = fr.groupby("bar_ts")["cf_1x"].sum()
        df["funding_cf_1x"] = df["ts"].map(cf_bar).fillna(0.0)

    scale = CAPITAL_USD * LEVERAGE
    df["pnl_net_usd_ex_funding"] = df["pnl_net_1x_ex_funding"] * scale
    df["funding_cf_usd"] = df["funding_cf_1x"] * scale
    df["pnl_net_usd"] = df["pnl_net_usd_ex_funding"] + df["funding_cf_usd"]

    df["equity_usd"] = CAPITAL_USD + df["pnl_net_usd"].cumsum()
    return df


# -----------------------------
# Stats
# -----------------------------
def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(dd.min()) if len(dd) else 0.0


def sharpe(returns: pd.Series, bars_per_year: float) -> float:
    r = returns.dropna()
    if len(r) < 500:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=0)
    if sd == 0:
        return float("nan")
    return float((mu / sd) * math.sqrt(bars_per_year))


def summarize(tag: str, df: pd.DataFrame) -> None:
    equity = df["equity_usd"]
    rets = equity.pct_change().fillna(0.0)
    sh = sharpe(rets, bars_per_year=365 * 24 * 4)  # 15m bars/year
    mdd = max_drawdown(equity)
    total = float(equity.iloc[-1] - equity.iloc[0])
    trades = int((df["pos_exec"].diff().fillna(0) != 0).sum())

    print(f"\n===== {tag} =====")
    print(f"Net PnL (incl funding): ${total:,.2f}")
    print(f"End equity:             ${equity.iloc[-1]:,.2f}")
    print(f"Sharpe (15m equity):    {sh:.2f}")
    print(f"Max drawdown:           {mdd*100:.2f}%")
    print(f"Trades (approx):        {trades}")
    print("====================\n")


# -----------------------------
# Main
# -----------------------------
def main():
    print("Downloading data from Binance (15m)...")
    df_feat, df_trade, df_fr = load_all_data(SYMBOL, DAYS_BACK_MAIN)

    if df_feat.empty or df_trade.empty:
        raise RuntimeError("No data returned. Check symbol/interval or Binance availability.")

    df_bt = backtest(df_feat, df_trade, df_fr)

    out_cols = [
        "ts","open","close",
        "mark_close","index_close","basis_close","basis_z",
        "funding_last","funding_z",
        "oi_last","oi_notional","oi_notional_z",
        "ret_1","ret_4","mom",
        "score","pos_signal","pos_exec",
        "ret_bar","pnl_net_usd_ex_funding","funding_cf_usd","pnl_net_usd","equity_usd",
    ]
    df_bt[out_cols].to_csv("backtest_results_ethusdt_15m_improved.csv", index=False)

    summarize("3-MONTH BACKTEST (OI soft; funding pnl included)", df_bt)
    print("Saved: backtest_results_ethusdt_15m_improved.csv")

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=DAYS_BACK_OI_MAX)
    df_30 = df_bt[df_bt["ts"] >= cutoff].copy()
    if not df_30.empty:
        summarize("30-DAY SLICE (OI available throughout slice)", df_30)


if __name__ == "__main__":
    main()
