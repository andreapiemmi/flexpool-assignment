from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from flexpool_assignment.common.ml_utils import set_seed, StandardScaler, fit_torch_linear
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor


def split_train_val(
        supervised_table: pd.DataFrame, 
        params: Dict[str,Any]) -> Dict[str,Any]:
    test_size = float(params['test_size'])

    df = supervised_table.sort_values('datetime_local').reset_index(drop=True)

    ycol = "consumption_MWh"

    feature_cols = (df.columns.
                    difference(["datetime_utc_from","datetime_local", ycol]).
                    tolist())
    
    df_model = (
        df.dropna(subset = feature_cols + [ycol]).
        reset_index(drop = True)
    )

    X_train, X_val, Y_train, Y_val = train_test_split(
        df_model[feature_cols].to_numpy(dtype=np.float32),
        df_model[ycol].to_numpy(dtype=np.float32),
        test_size=test_size,
        shuffle=False
    )

    baseline_cols = ["datetime_local", ycol] + [c for c in df_model.columns if c.startswith("lag_")]
    df_for_baselines = df_model[baseline_cols].reset_index(drop=True)

    return {
        "feature_cols": feature_cols,
        "X_train": X_train,
        "Y_train": Y_train,
        "X_val": X_val,
        "Y_val": Y_val,
        "df_for_baselines": df_for_baselines,
    }

def train_linear_ols_torch(
        split_data : Dict[str,Any],
        params: Dict[str,Any]) -> Dict[str,Any]:
    
    set_seed(int(params['seed']))

    X_train, Y_train = split_data['X_train'], split_data['Y_train']
    X_val, Y_val = split_data["X_val"], split_data['Y_val']
    feature_cols = split_data["feature_cols"]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train).astype(np.float32)
    Xva = scaler.transform(X_val).astype(np.float32)

    t = params['torch']

    art, val_mae = fit_torch_linear(
        Xtr, Y_train.astype(np.float32),
        Xva, Y_val.astype(np.float32),
        l1=0.0, l2=0.0,
        lr=float(t["lr"]),
        max_epochs=int(t["max_epochs"]),
        patience=int(t["patience"]),
        seed=int(params["seed"]),
    )

    return {
        "model_name": "torch_ols",
        "model_type": "torch_linear",
        "val_mae": float(val_mae),
        "feature_cols": feature_cols,
        "scaler": scaler,
        "torch": art,
    }

def train_ridge_lasso_enet_torch(
        split_data : Dict[str,Any],
        params : Dict[str,Any]) -> Dict[str,Any]:
    
    set_seed(int(params['seed']))

    X_train, Y_train = split_data["X_train"], split_data["Y_train"]
    X_val, Y_val = split_data["X_val"], split_data["Y_val"]
    feature_cols = split_data["feature_cols"]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train).astype(np.float32)
    Xva = scaler.transform(X_val).astype(np.float32)

    Ytr = Y_train.astype(np.float32)
    Yva = Y_val.astype(np.float32)

    t = params["torch"]
    alphas = t["enet_alpha"]
    ratios = t["enet_l1_ratio"]

    rows: List[Dict[str, Any]] = []
    best = {"name": None, "mae": float("inf"), "torch_art": None}

    for alpha in alphas:
        for r in ratios:
            l1 = float(alpha) * float(r)
            l2 = float(alpha) * float(1.0 - float(r))

            torch_art, val_mae = fit_torch_linear(
                Xtr, Ytr, Xva, Yva,
                l1=l1, l2=l2,
                lr=float(t["lr"]),
                max_epochs=int(t["max_epochs"]),
                patience=int(t["patience"]),
                seed=int(params["seed"]),
            )

            name = f"torch_enet_alpha={alpha}_l1r={r}"
            rows.append({"model": name, "val_mae": float(val_mae), "alpha": float(alpha), "l1_ratio": float(r)})

            if val_mae < best["mae"]:
                best = {"name": name, "mae": float(val_mae), "torch_art": torch_art}

    return {
        "model_name": best["name"],
        "model_type": "torch_linear",
        "val_mae": best["mae"],
        "feature_cols": feature_cols,
        "scaler": scaler,
        "torch": best["torch_art"],
        "grid_results": rows,
    }


def train_hgbr_sklearn(
        split_data: Dict[str,Any], 
        params: Dict[str,Any])-> Dict[str,Any]:
    
    seed = int(params['seed'])
    X_train, Y_train = split_data["X_train"], split_data["Y_train"]
    X_val, Y_val = split_data["X_val"], split_data["Y_val"]
    feature_cols = split_data["feature_cols"]

    h = params['hgbr']
    model = HistGradientBoostingRegressor(
        max_depth=int(h['max_depth']),
        learning_rate=float(h['learning_rate']),
        max_iter=int(h["max_iter"]),
        min_samples_leaf=int(h["min_samples_leaf"]),
        l2_regularization=float(h["l2_regularization"]),
        random_state=seed,
    )

    model.fit(X_train, Y_train)
    pred = model.predict(X_val).astype(np.float32)
    val_mae = float(np.mean(np.abs(pred-Y_val)))

    return {
        "model_name": "sklearn_hgbr",
        "model_type": "sklearn_hgbr",
        "val_mae": val_mae,
        "feature_cols": feature_cols,
        "sklearn": {"model": model}
    }

def compare_and_select(
        ols_artifacts : Dict[str,Any],
        enet_artifacts: Dict[str, Any],
        hgbr_artifacts: Dict[str,Any],) -> Tuple[Dict[str,Any], pd.DataFrame]:
    
    rows = [
        {"model": ols_artifacts["model_name"], "val_mae": float(ols_artifacts["val_mae"]), "type": ols_artifacts["model_type"]},
        {"model": enet_artifacts["model_name"], "val_mae": float(enet_artifacts["val_mae"]), "type": enet_artifacts["model_type"]},
        {"model": hgbr_artifacts["model_name"], "val_mae": float(hgbr_artifacts["val_mae"]), "type": hgbr_artifacts["model_type"]},
    ]

    metrics = (
        pd.DataFrame(rows)
        .sort_values("val_mae", ascending=True)
        .reset_index(drop=True)
        .assign(rank=lambda d: np.arange(1, len(d) + 1))
        [["rank", "model", "type", "val_mae"]]
    )

    winner_name = metrics.iloc[0]["model"]
    if winner_name == ols_artifacts["model_name"]:
        best = ols_artifacts
    elif winner_name == enet_artifacts["model_name"]:
        best = enet_artifacts
    else:
        best = hgbr_artifacts

    bundle: Dict[str, Any] = {
        "best_model_name": best["model_name"],
        "best_model_type": best["model_type"],
        "val_mae": float(best["val_mae"]),
        "feature_cols": best["feature_cols"],
    }

    if best["model_type"] == "torch_linear":
        bundle.update(
            {
                "scaler": best.get("scaler"),
                "torch": best.get("torch"),
            }
        )
    else:
        bundle.update(
            {
                "sklearn": best.get("sklearn"),
            }
        )
    return bundle, metrics