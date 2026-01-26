from __future__ import annotations

from typing import Any, Dict, Tuple, List
import numpy as np 
import pandas as pd 

import torch 

from flexpool_assignment.common.ml_utils import TorchLinear

def _predict(
        model_bundle: Dict[str,Any],
        X: np.ndarray)-> np.ndarray:
     
    mtype = model_bundle["best_model_type"]

    if mtype == "torch_linear":
        scaler = model_bundle["scaler"]
        Xs = scaler.transform(X).astype(np.float32)

        torch_art = model_bundle["torch"]
        n_features = int(torch_art["n_features"])
        model = TorchLinear(n_features=n_features)
        model.load_state_dict(torch_art["state_dict"])
        model.eval()

        with torch.no_grad():
            pred = model(torch.from_numpy(Xs)).numpy().astype(np.float32)
        return pred

    if mtype == "sklearn_hgbr":
        model = model_bundle["sklearn"]["model"]
        pred = model.predict(X).astype(np.float32)
        return pred

    raise ValueError(f"Unknown best_model_type: {mtype}")

def backtest_winner_model(
        supervised_table: pd.DataFrame,
        model_artifacts: Dict[str,Any],
        params: Dict[str,Any]) -> Tuple[pd.DataFrame,pd.DataFrame]:
    df = (
        supervised_table.
        sort_values("datetime_local").
        reset_index(drop=True)
    )

    feature_cols: List[str] = model_artifacts["feature_cols"]
    ycol = "consumption_MWh"

    # We backtest over forecast DAYS (the D+1 day). We need full 24 rows.
    df = df.dropna(subset=feature_cols + [ycol]).reset_index(drop=True)

    # build local date column (tz-aware -> .dt.date is fine)
    df = df.assign(date_local=lambda d: d["datetime_local"].dt.date)

    all_days = np.array(sorted(df["date_local"].unique()))
    if len(all_days) < 3:
        raise ValueError("Not enough days in supervised_table for backtesting.")

    # choose last N forecast days where we can simulate an origin day (needs previous day present)
    n = int(params["backtest_days"])
    chosen_forecast_days = all_days[-min(n, len(all_days)):]  # last N days available

    hourly_rows = []
    daily_rows = []

    for fday in chosen_forecast_days:
        # forecast day = fday; origin day is fday - 1 day (forecast made at origin 10:00)
        # we don't actually use origin 10:00 in features because you designed lags >=48h (safe),
        # but we keep the narrative consistent.
        day_mask = df["date_local"] == fday
        df_day = df[day_mask].reset_index(drop=True)

        # require full 24 hours
        if len(df_day) != 24:
            continue

        X = df_day[feature_cols].to_numpy(dtype=np.float32)
        y_true = df_day[ycol].to_numpy(dtype=np.float32)
        y_pred = _predict(model_artifacts, X)

        mae = float(np.mean(np.abs(y_pred - y_true)))

        daily_rows.append(
            {
                "forecast_day": str(fday),
                "n_hours": int(len(df_day)),
                "mae": mae,
            }
        )

        hourly_rows.append(
            df_day[["datetime_local", ycol]]
            .assign(
                forecast_day=str(fday),
                y_pred=y_pred,
                abs_error=np.abs(y_pred - y_true),
            )
        )
    backtest_daily = (
        pd.DataFrame(daily_rows)
        .sort_values("forecast_day")
        .reset_index(drop=True)
        .assign(
            mae_mean=lambda d: d["mae"].mean() if len(d) else np.nan,
            mae_median=lambda d: d["mae"].median() if len(d) else np.nan,
            mae_p90=lambda d: d["mae"].quantile(0.9) if len(d) else np.nan,
        )
    )

    backtest_hourly = (
        pd.concat(hourly_rows, ignore_index=True)
        if len(hourly_rows) else
        pd.DataFrame(columns=["datetime_local", ycol, "forecast_day", "y_pred", "abs_error"])
    )

    return backtest_daily, backtest_hourly
