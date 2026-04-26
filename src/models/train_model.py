"""Módulo contendo a rotina de treinamento isolada do PyTorch."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import average_precision_score
from typing import Tuple

def train_advanced_model(
    model: nn.Module,
    criterion: nn.Module,
    loader_tr: DataLoader,
    dataset_val,
    device: torch.device,
    focal_gamma: float,
    focal_alpha: float,
    max_lr: float,
    weight_decay: float,
    n_epochs: int = 150,
    patience: int = 15
) -> Tuple[nn.Module, float]:
    """
    Executa o loop de treinamento da rede neural com Early Stopping e OneCycleLR.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = OneCycleLR(
        optimizer, max_lr=max_lr, steps_per_epoch=len(loader_tr), epochs=n_epochs, pct_start=0.3
    )

    best_pr_auc = 0.0
    patience_cnt = 0
    best_state = None

    X_num_val = dataset_val.X_num.to(device)
    X_cat_val = dataset_val.X_cat.to(device)
    y_val_t = dataset_val.y.to(device)

    for epoch in range(1, n_epochs + 1):
        model.train()
        for X_num, X_cat, yb in loader_tr:
            X_num, X_cat, yb = X_num.to(device), X_cat.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_num, X_cat), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_num_val, X_cat_val)).cpu().numpy()
            val_pr_auc = average_precision_score(y_val_t.cpu().numpy(), val_probs)

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            patience_cnt = 0
            best_state = model.state_dict()
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_pr_auc
