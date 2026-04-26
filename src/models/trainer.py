"""Módulo contendo a rotina de treinamento para a Focal Loss e OneCycleLR."""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score
import pandas as pd
from typing import Tuple

def train_focal_model(
    model: nn.Module,
    criterion: nn.Module,
    X_tr_t: torch.Tensor,
    y_tr_t: torch.Tensor,
    X_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    device: torch.device,
    max_lr: float,
    weight_decay: float,
    n_epochs: int = 150,
    batch_size: int = 64,
    patience: int = 15
) -> Tuple[nn.Module, list]:
    """
    Executa o loop de treinamento avançado com Focal Loss, Early Stopping, AdamW e OneCycleLR.
    """
    dataset_tr = TensorDataset(X_tr_t, y_tr_t)
    loader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, steps_per_epoch=len(loader_tr), epochs=n_epochs, pct_start=0.3
    )

    best_pr_auc = 0.0
    patience_cnt = 0
    best_state = None
    history = []

    X_val_t = X_val_t.to(device)
    # Important: squeeze to match shape [batch_size]
    y_val_t = y_val_t.to(device).view(-1)

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in loader_tr:
            xb = xb.to(device)
            # Ensure target shape is [batch_size] to match our ChurnMLP squeeze output
            yb = yb.to(device).view(-1)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * len(xb)

        train_loss /= len(loader_tr.dataset)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_pr_auc = average_precision_score(y_val_t.cpu().numpy(), val_probs)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_pr_auc': val_pr_auc
        })

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

    return model, history
