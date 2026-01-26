from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
import torch

import flexpool_assignment.common.ml_utils as ml_utils 

def _train_one_epoch(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module) -> float: 
    model.train() # the model is now in training mode
    total_loss, total_n = 0.0, 0
    for xb, yb in loader: 
        # 1) Foward üass: calcuate predictions given the params
        y_hat = model(xb)
        # 2) how bad ae these with respect to a subset of the data train set?
        loss = loss_fn(y_hat, yb)

        # 3) given answer to 2, compute gradients of loss func. wrt param space \theta
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # recording (not): prevnt the tracking of graients in the plot 
        bs = xb.shape[0]
        total_loss += float(loss.detach().cpu().item()) * bs
        total_n += bs
    
    return total_loss / total_n 

def train_torch_linear(
        X_train: pd.DataFrame,
        Y_train: pd.DataFrame, 
        feature_scaler: dict[str, Any],
        seed: int, 
        batch_size: int, 
        epochs: int, 
        lr: float,
        weight_decay: float) -> dict:
     torch.manual_seed(seed) 
     device = torch.device("cpu")

     X_train_t = ml_utils.apply_scaler(X_train, feature_scaler)
     Y_train_t = ml_utils.to_np_float32(Y_train)

     model = torch.nn.Linear(X_train_t.shape[1],1).to(device)
     loss_fn = torch.nn.MSELoss()
     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

     dataset = torch.utils.data.TensorDataset(
         torch.tensor(X_train_t, device=device),
         torch.tensor(Y_train_t, device=device))
     loader = torch.utils.data.DataLoader(
         dataset, batch_size=batch_size, shuffle=True, drop_last=False)
     
     for _ in range(epochs):
         _ = _train_one_epoch(model = model, loader = loader, 
                              optimizer = optimizer, loss_fn = loss_fn)    
     return model.state_dict()

def predict_baseline_torch_linear(
        X_train: pd.DataFrame,
        Y_train: pd.DataFrame,
        X_val: pd.DataFrame, 
        Y_val: pd.DataFrame,
        X_test: pd.DataFrame,
        Y_test: pd.DataFrame,
        features_scaler: dict[str, Any],
        torch_linear_state: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    Y_train_t = ml_utils.to_np_float32(Y_train)
    mu = float(Y_train_t.mean())

    Yva, Yve = ml_utils.to_np_float32(Y_val), ml_utils.to_np_float32(Y_test)
    
    mean_val, mean_test = np.full_like(Yva, mu),  np.full_like(Yve, mu)

    Xva, Xte = ml_utils.apply_scaler(X_val, features_scaler), ml_utils.apply_scaler(X_test, features_scaler)

    model = torch.nn.Linear(Xva.shape[1],1)
    model.load_state_dict(torch_linear_state)
    model.eval()

    with torch.no_grad():
        lin_val = model(torch.tensor(Xva)).numpy()
        lin_test = model(torch.tensor(Xte)).numpy()

    preds_val = pd.DataFrame({
        "Y_true": Yva.ravel(), 
        "mean_baseline": mean_val.ravel(),
        "torch_linear": lin_val.ravel()
    })

    preds_test = pd.DataFrame({
        "Y_true": Yve.ravel(),
        "mean_baseline": mean_test.ravel(),
        "torch_linear": lin_test.ravel()
    })

    return preds_val, preds_test