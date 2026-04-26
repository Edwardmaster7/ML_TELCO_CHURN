"""Módulo contendo implementações customizadas de Datasets do PyTorch."""
import numpy as np
import torch
from torch.utils.data import Dataset

class ChurnEmbeddingDataset(Dataset):
    """
    Dataset que fatia um array pré-processado em tensores numéricos e categóricos
    para alimentação em arquiteturas com Entity Embeddings.
    """
    def __init__(self, X_proc: np.ndarray, y: np.ndarray, num_features_cnt: int):
        self.X_num = torch.tensor(X_proc[:, :num_features_cnt], dtype=torch.float32)
        # Cast para long é obrigatório para camadas nn.Embedding
        self.X_cat = torch.tensor(X_proc[:, num_features_cnt:], dtype=torch.long) + 1
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple:
        return self.X_num[idx], self.X_cat[idx], self.y[idx]
