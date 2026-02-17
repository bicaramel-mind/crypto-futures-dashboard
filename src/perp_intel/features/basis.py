from __future__ import annotations

import pandas as pd
import numpy as np

def basis_z(basis: pd.Series, window: int = 240) -> pd.Series:
    m = basis.rolling(window).mean()
    s = basis.rolling(window).std(ddof=0)
    return (basis - m) / s.replace(0, np.nan)
