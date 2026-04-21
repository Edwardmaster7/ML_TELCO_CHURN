# src/ml_telco_churn/model_nn.py
import torch
import torch.nn as nn
from typing import List

class ChurnMLP(nn.Module):
    """MLP para classificação binária de churn.

    Blocos: Linear -> BatchNorm1d -> ReLU -> Dropout
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)
