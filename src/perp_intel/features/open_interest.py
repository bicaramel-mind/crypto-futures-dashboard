from __future__ import annotations

import pandas as pd
import numpy as np

def oi_price_bucket(price: pd.Series, oi: pd.Series) -> pd.Series:
    dp = price.diff()
    doi = oi.diff()

    def lab(x):
        dp_, doi_ = x
        if pd.isna(dp_) or pd.isna(doi_):
            return None
        if dp_ > 0 and doi_ > 0: return "price_up_oi_up"
        if dp_ > 0 and doi_ < 0: return "price_up_oi_down"
        if dp_ < 0 and doi_ > 0: return "price_down_oi_up"
        if dp_ < 0 and doi_ < 0: return "price_down_oi_down"
        return "flat"

    return pd.DataFrame({"dp": dp, "doi": doi}).apply(lambda r: lab((r["dp"], r["doi"])), axis=1)

def impulse_z(oi: pd.Series, window: int = 240) -> pd.Series:
    d = oi.diff()
    m = d.rolling(window).mean()
    s = d.rolling(window).std(ddof=0)
    return (d - m) / s.replace(0, np.nan)
