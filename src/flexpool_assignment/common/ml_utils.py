from __future__ import annotations

from dataclasses import dataclass 
from typing import Any 

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler

def to_np_float32(df: pd.DataFrame) -> np.ndarray:
    return df.to_numpy(dtype=np.float32)

def fit_scaler(X: pd.DataFrame) -> dict[str, Any]:
    scaler = StandardScaler().fit( to_np_float32(X) )
    return{'mean_': scaler.mean_.astype(np.float32),
           'sd_': scaler.scale_.astype(np.float32)}

def apply_scaler(X: pd.DataFrame, scaler: dict[str,Any])->np.ndarray:
    mean, sd = np.asarray(scaler['mean_'], dtype=np.float32), np.asarray(scaler['sd_'], dtype=np.float32)
    return (to_np_float32(X) - mean) / sd

