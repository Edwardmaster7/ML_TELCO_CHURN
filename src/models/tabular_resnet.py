"""
Módulo contendo a arquitetura AdvancedChurnMLP baseada em ResNet Blocks
e Entity Embeddings para dados tabulares.
"""
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Função de Perda Focal para Classificação Binária com foco em instâncias difíceis."""
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce_loss)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()

class GaussianNoise(nn.Module):
    """Injeta ruído gaussiano (N(0, sigma)) em tensores contínuos contra overfitting."""
    def __init__(self, sigma: float = 0.05):
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.sigma > 0:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x

class ResNetBlock(nn.Module):
    """Bloco Residual constante para dados tabulares usando LayerNorm e GELU."""
    def __init__(self, dim: int, dropout_rate: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.lin2 = nn.Linear(dim, dim)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x)
        out = self.lin1(out)
        out = self.act(out)
        out = self.drop1(out)
        out = self.lin2(out)
        out = self.drop2(out)
        return x + out

class AdvancedChurnMLP(nn.Module):
    """Rede Neural Tabular Avançada usando Entity Embeddings e blocos ResNet."""
    def __init__(
        self,
        num_dim: int,
        cat_cardinalities: List[int],
        hidden_dim: int,
        num_blocks: int,
        dropout_rate: float,
        noise_sigma: float = 0.05
    ):
        super().__init__()
        self.noise = GaussianNoise(sigma=noise_sigma)

        # O índice 0 é reservado para 'unknown' gerado pelo OrdinalEncoder (-1 + 1)
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, min(50, (card // 2) + 1), padding_idx=0) for card in cat_cardinalities
        ])

        total_emb_dim = sum(min(50, (card // 2) + 1) for card in cat_cardinalities)
        self.total_input_dim = num_dim + total_emb_dim

        self.initial_projection = nn.Sequential(
            nn.Linear(self.total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.blocks = nn.Sequential(*[
            ResNetBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x_num = self.noise(x_num)

        emb_outputs = []
        for i, emb_layer in enumerate(self.embeddings):
            emb_outputs.append(emb_layer(x_cat[:, i]))

        if emb_outputs:
            x_cat_emb = torch.cat(emb_outputs, dim=1)
            x = torch.cat([x_num, x_cat_emb], dim=1)
        else:
            x = x_num

        x = self.initial_projection(x)
        x = self.blocks(x)
        return self.head(x)
