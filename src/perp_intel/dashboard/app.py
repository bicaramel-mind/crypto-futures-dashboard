from __future__ import annotations

import duckdb
import pandas as pd
import streamlit as st
import altair as alt
from perp_intel.config import settings
from perp_intel.features.funding import funding_crowding_score
from perp_intel.charting.charts import altair_line_chart

alt.data_transformers.disable_max_rows()

st.set_page_config(page_title="Perp Intel", layout="wide")
st.title("Perp Intel — Funding / OI / Basis")

con = duckdb.connect(settings.duckdb_path, read_only=True)

sym = st.text_input("Symbol (e.g., BTCUSDT)", "BTCUSDT").upper().strip()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Mark Price + Funding (live-ish)")
    df = con.execute(
        """
        SELECT ts_ms, symbol, mark_price, index_price, funding_rate, next_funding_ts_ms
        FROM mark_price
        WHERE symbol = ?
        ORDER BY ts_ms DESC
        LIMIT 2000
        """,
        [sym],
    ).df()
    if len(df) == 0:
        st.info("No live markPrice rows yet. Run realtime streamer first.")
    else:
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df = df.sort_values("ts")
        df["crowding"] = funding_crowding_score(df["funding_rate"], window=240)
        st.altair_chart(altair_line_chart(df, "ts", "Time", "mark_price", "Mark Price", ["ts:T", "mark_price:Q"]))
        st.line_chart(df.set_index("ts")[["funding_rate"]])
        st.line_chart(df.set_index("ts")[["crowding"]])

with col2:
    st.subheader("Funding History")
    fh = con.execute(
        """
        SELECT funding_ts_ms, funding_rate, mark_price
        FROM funding_history
        WHERE symbol = ?
        ORDER BY funding_ts_ms DESC
        LIMIT 1000
        """,
        [sym],
    ).df()
    if len(fh) == 0:
        st.info("No funding history yet. Run backfill.")
    else:
        fh["ts"] = pd.to_datetime(fh["funding_ts_ms"], unit="ms", utc=True)
        fh = fh.sort_values("ts")
        st.line_chart(fh.set_index("ts")[["funding_rate"]])

st.subheader("Open Interest History (1h)")
oi = con.execute(
    """
    SELECT ts_ms, sum_open_interest, sum_open_interest_value
    FROM open_interest_hist
    WHERE symbol = ?
    ORDER BY ts_ms DESC
    LIMIT 1000
    """,
    [sym],
).df()
if len(oi) > 0:
    oi["ts"] = pd.to_datetime(oi["ts_ms"], unit="ms", utc=True)
    oi = oi.sort_values("ts")
    st.line_chart(oi.set_index("ts")[["sum_open_interest_value"]])

st.subheader("Basis History (PERPETUAL)")
bh = con.execute(
    """
    SELECT ts_ms, basis, basis_rate, futures_price, index_price
    FROM basis_hist
    WHERE pair = ?
    ORDER BY ts_ms DESC
    LIMIT 1000
    """,
    [sym],
).df()
if len(bh) > 0:
    bh["ts"] = pd.to_datetime(bh["ts_ms"], unit="ms", utc=True)
    bh = bh.sort_values("ts")
    st.line_chart(bh.set_index("ts")[["basis", "basis_rate"]])
