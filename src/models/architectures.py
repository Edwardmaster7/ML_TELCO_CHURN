"""
Módulo contendo as arquiteturas baseadas em ChurnMLP com suporte a Focal Loss,
usadas como campeãs do Tech Challenge.
"""
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Função de Perda Focal (Focal Loss) para Classificação Binária.

    Aborda o desbalanceamento de classes através de ponderação (alpha) e
    reduz dinamicamente o gradiente para exemplos fáceis (gamma).

    Args:
        alpha (float): Fator de ponderação para a classe minoritária (0 a 1). Padrão: 0.75.
        gamma (float): Fator de foco para exemplos difíceis. Padrão: 2.0.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Se os targets não tiverem o mesmo shape dos logits, usamos view(-1) ou formatamos o target no train loop
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce_loss)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()

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
