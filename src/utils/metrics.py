import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple


def fit_trend_line(dates: pd.Series, values: pd.Series) -> Tuple[pd.Series, float]:
    clean = values.dropna()
    if len(clean) < 2:
        return pd.Series([np.nan] * len(dates), index=dates.index), 0.0
    x = np.arange(len(dates))
    slope, intercept, _, _, _ = stats.linregress(x, values.fillna(values.mean()).values)
    fitted = pd.Series(slope * x + intercept, index=dates.index)
    return fitted, float(slope)


def trend_direction(series: pd.Series, min_pts: int = 3) -> str:
    valid = series.dropna()
    if len(valid) < min_pts:
        return "insufficient data"
    x = np.arange(len(valid))
    slope, _, _, p, _ = stats.linregress(x, valid.values)
    if p > 0.10:
        return "maintaining"
    return "increasing" if slope > 0 else "decreasing"
