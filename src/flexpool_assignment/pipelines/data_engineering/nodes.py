from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd

def preprocessing(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    df = (
        df.copy()
        .assign(datetime_utc_from=lambda d: pd.to_datetime(d[dt_col], utc=True))
        .sort_values("datetime_utc_from")
    ) # ensures that df.datetime_utc_from.is_monotonic_increasing

    # duplicates - only 2 and no clear pattern
    duplicates = df[df["datetime_utc_from"].duplicated(keep=False)]

    # averaging - minimal impact and conservative solution
    df = (
        df.groupby("datetime_utc_from", as_index=False)
        .mean(numeric_only=True)
        .sort_values("datetime_utc_from")
    )

    # 1 hour unique delta - check passed
    delta = df["datetime_utc_from"].diff().value_counts()

    # convert to zurich just to be safe 
    # drop na() after having checked fractions of NAs are immaterial, if any
    df = (
        df.assign(datetime_local=lambda d: d["datetime_utc_from"].dt.tz_convert("Europe/Zurich"))
        .sort_values("datetime_local")
        .dropna()
        .reset_index(drop=True)
    )

    return df

# hour of the day is 1 over 24, day of the week 1 over 7 
# preserve cyclicality in a continuous variable as opposed 
# to dummy variables
def _add_cyclical(df: pd.DataFrame,
                  col: str,
                  period: float) ->pd.DataFrame:
    return df.assign(
        **{
            f"{col}_sin": lambda d: np.sin(2*np.pi *d[col]/period),
            f"{col}_cos": lambda d: np.cos(2*np.pi *d[col]/period)
        } 
    )


def make_supervised_table(
        df: pd.DataFrame, 
        params: Dict[str, Any]) -> pd.DataFrame: 
    
    lags = params["lags_hours"] #[48, 72, 168] - choice to be safe that we are actually bidding at 10 using 
    # consumption data that belong to previous day only
    if 24 in lags: 
        raise ValueError("24 lag violates the bidding mechansim - consumption up untill D 00:00")
    
    base = (
        df.sort_values("datetime_local")
        .assign(
            hour=lambda d: d["datetime_local"].dt.hour.astype("int16"),
            dow=lambda d: d["datetime_local"].dt.dayofweek.astype("int16"),
            month=lambda d: d["datetime_local"].dt.month.astype("int16"),
            is_weekend=lambda d: (d["dow"] >= 5).astype("int8"),
        )
        .pipe(_add_cyclical, col = "hour", period = 24.0)#apply the cyclical sin & cos funcs
        .pipe(_add_cyclical, col = "dow", period = 7)#to day of the week and hour of day
    )
    lag_features = {
        f"lag_{h}h": (lambda hh: lambda d: d["consumption_MWh"].shift(hh))(h)
        for h in lags
    } # look at lagged consumption for the list [48, 72, 168] as abov proposed

    return ( base.assign(**lag_features).reset_index(drop = True))