from __future__ import annotations

import numpy as np
import pandas as pd

def zscore(series: pd.Series, window: int = 240) -> pd.Series:
    m = series.rolling(window).mean()
    s = series.rolling(window).std(ddof=0)
    return (series - m) / s.replace(0, np.nan)

def funding_crowding_score(funding_rate: pd.Series, window: int = 240) -> pd.Series:
    """
    Simple crowding: abs(z) mapped into 0..100 with clipping.
    """
    z = zscore(funding_rate, window=window).abs()
    score = (z.clip(0, 4) / 4.0) * 100.0
    return score
