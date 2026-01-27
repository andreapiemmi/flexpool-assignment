from __future__ import annotations

from typing import Tuple, Dict, Any 
import numpy as np 
import torch 
import torch.nn as nn 

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class StandardScaler: 

    def __init__(self)->None: 
        self.mean_ : np.ndarray | None = None
        self.std_: np.ndarray | None = None 
    
    def fit(self, X:np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis = 0)
        self.std_ = X.std(axis = 0) + 1e-12
        return self 
    
    def transform(self, X: np.ndarray) -> np.ndarray: 
        if self.mean_ is None or self.std_ is None: 
            raise RuntimeError('You have to fit the scaler before calling transform methods')
        return (X-self.mean_) / self.std_
    
    def fit_transform(self, X : np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class TorchLinear(nn.Module):

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def fit_torch_linear(
        X_train : np.ndarray, 
        Y_train: np.ndarray, 
        X_val: np.ndarray, 
        Y_val: np.ndarray, 
        *, 
        l1: float = 0.0,
        l2: float = 0.0, 
        lr: float = 0.05, 
        max_epochs: int = 400, 
        patience: int = 25, 
        seed: int = 42) -> Tuple[Dict[str,Any], float]:
    
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TorchLinear(X_train.shape[1]).to(device)

    Xtr = torch.from_numpy(X_train).to(device)
    Ytr = torch.from_numpy(Y_train).to(device)
    Xva = torch.from_numpy(X_val).to(device)
    Yva = torch.from_numpy(Y_val).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    best_val = float("inf")
    best_state = None
    patience_left = patience

    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()

        pred = model(Xtr)
        loss = loss_fn(pred, Ytr)

        # penalties (exclude bias)
        w = model.linear.weight
        if l2 > 0: #ridge
            loss = loss + l2 * w.pow(2).sum()
        if l1 > 0: #lasso
            loss = loss + l1 * w.abs().sum()

        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xva)
            val_mae = float(loss_fn(val_pred, Yva).item())

        if val_mae < best_val - 1e-5:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

            
    artifacts = {
            "model_type": "torch_linear",
            "n_features": X_train.shape[1],
            "l1": float(l1),
            "l2": float(l2),
            "state_dict": best_state,
        }
    return artifacts, best_val


