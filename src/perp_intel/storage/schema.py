SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS mark_price (
  ts_ms BIGINT,
  symbol VARCHAR,
  mark_price DOUBLE,
  index_price DOUBLE,
  funding_rate DOUBLE,
  next_funding_ts_ms BIGINT
);

CREATE TABLE IF NOT EXISTS funding_history (
  funding_ts_ms BIGINT,
  symbol VARCHAR,
  funding_rate DOUBLE,
  mark_price DOUBLE
);

CREATE TABLE IF NOT EXISTS open_interest_hist (
  ts_ms BIGINT,
  symbol VARCHAR,
  sum_open_interest DOUBLE,
  sum_open_interest_value DOUBLE
);

CREATE TABLE IF NOT EXISTS basis_hist (
  ts_ms BIGINT,
  pair VARCHAR,
  futures_price DOUBLE,
  index_price DOUBLE,
  basis DOUBLE,
  basis_rate DOUBLE
);

CREATE TABLE IF NOT EXISTS force_orders (
  ts_ms BIGINT,
  symbol VARCHAR,
  side VARCHAR,
  price DOUBLE,
  qty DOUBLE
);
"""
