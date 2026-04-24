# Phase 1: Advanced Loss & Schedulers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quebrar o teto de PR-AUC (0.649) implementando Focal Loss e OneCycleLR com AdamW no PyTorch MLP, documentando as melhorias em um novo notebook para preservar a evolução do projeto e aderindo ao PEP8 e Clean Code.

**Architecture:** Moveremos os componentes modulares de Loss (`FocalLoss`) e a reestruturação do loop de treinamento para um novo notebook isolado (`06_mlp_advanced_loss.ipynb`). Utilizaremos os mesmos dados processados (`churn_processed_advanced.csv`) e rastrearemos no MLflow (`telco_churn_advanced_loss`).

**Tech Stack:** PyTorch, Optuna, MLflow, Scikit-Learn, Pandas.

---

### Task 1: Criar Módulo da Focal Loss (Clean Code & PEP8)

**Files:**

- Create: `notebooks/06_mlp_advanced_loss.ipynb` (Iniciando o notebook)

- [ ] **Step 1: Inicializar o notebook e configurar imports**

Crie a primeira célula do notebook importando as configurações globais de `src` e estabelecendo a definição correta do *device* (`mps`/`cuda`).

```python
 
"""
Notebook 06: Implementação de Estratégias Avançadas para MLP
Foco na Fase 1 do ADR-006: Focal Loss e Otimização via OneCycleLR com AdamW.
"""
import os
import sys

# Adiciona o src/ ao PYTHONPATH para import do config
sys.path.append(os.path.abspath(os.path.join('..')))

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

# Integração de constantes do projeto
from src.ml_telco_churn.config import CONFIG

# Constantes locais do experimento
RANDOM_STATE = CONFIG.random_state
TEST_SIZE = 0.2
VAL_SIZE = 0.15
BATCH_SIZE = 256
N_EPOCHS = 300
PATIENCE = 20
N_TRIALS_OPTUNA = 20
PATH_DATA = '../notebooks/data/processed/churn_processed_advanced.csv'
EXPERIMENT_NAME = "04_PyTorch_Advanced_Loss"

# Configurações de Reproducibilidade e Device
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Usando device: {device}")
```

- [ ] **Step 2: Implementar a classe FocalLoss**

Implemente a classe matemática da Focal Loss com docstrings padronizados de mercado (Sphinx/Google style).

```python
class FocalLoss(nn.Module):
    """
    Função de Perda Focal (Focal Loss) para Classificação Binária.
  
    Aborda o desbalanceamento de classes através de ponderação (alpha) e 
    reduz dinamicamente o gradiente para exemplos fáceis (gamma).
  
    Args:
        alpha (float): Fator de ponderação para a classe minoritária (0 a 1).
            Padrão: 0.75.
        gamma (float): Fator de foco para exemplos difíceis. 
            Valores maiores reduzem a perda para predições com alta confiança. 
            Padrão: 2.0.
    """
  
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
      
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcula a Focal Loss.
      
        Args:
            logits (torch.Tensor): Previsões cruas do modelo (antes da sigmoid).
            targets (torch.Tensor): Rótulos verdadeiros.
          
        Returns:
            torch.Tensor: Perda média calculada para o batch.
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
      
        # P_t é a probabilidade estimada do modelo para a classe alvo real
        p_t = torch.exp(-bce_loss) 
      
        # Fator modulador: diminui para exemplos bem classificados (P_t -> 1)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
      
        loss = focal_weight * bce_loss
        return loss.mean()
```

### Task 2: Carregamento de Dados e Classe ChurnMLP

**Files:**

- Modify: `notebooks/06_mlp_advanced_loss.ipynb`

- [ ] **Step 1: Carregar os dados avançados da Iteração 3**

```python
# Garantir que estamos puxando as features avançadas
df = pd.read_csv(PATH_DATA)

target_col = "Churn"
X = df.drop(columns=[target_col])
y = df[target_col]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train
)

INPUT_DIM = X_train.shape[1]
```

- [ ] **Step 2: Definir a classe base do PyTorch MLP**

Importamos a arquitetura do Experimento 05 mantendo a estrutura linear original, mas aplicando boas práticas de nomenclatura.

```python
class ChurnMLP(nn.Module):
    """
    Rede Neural Multi-Layer Perceptron (MLP) padrão para classificação tabular.
    """
    def __init__(self, input_dim: int, hidden_dims: list, dropout_rate: float = 0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
      
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
          
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
      
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processa as features numéricas através da rede densa."""
        return self.network(x)
```

### Task 3: Refatorar o Loop de Treinamento (AdamW + OneCycleLR)

**Files:**

- Modify: `notebooks/06_mlp_advanced_loss.ipynb`

- [ ] **Step 1: Escrever função de treino com suporte a OneCycleLR e Focal Loss**

Reescrevemos `train_mlp` para suportar `loss_type` paramétrico e desacoplamento do AdamW.

```python
def train_mlp_advanced(
    model: nn.Module, 
    X_tr_np: np.ndarray, 
    y_tr_np: np.ndarray, 
    X_val_np: np.ndarray, 
    y_val_np: np.ndarray, 
    loss_type: str = "bce",
    pos_weight: float = 1.0,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
    n_epochs: int = 150, 
    batch_size: int = 64, 
    max_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20
) -> tuple:
    """
    Realiza o treinamento avançado da rede neural com Early Stopping, AdamW e OneCycleLR.
    """
    # 1. Preparação dos Datasets
    X_tr_t = torch.tensor(X_tr_np, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_np, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val_np, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1)
  
    dataset_tr = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
  
    # 2. Definição da Loss e Otimizador
    if loss_type == "focal":
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma).to(device)
    else:
        pw = torch.tensor([pos_weight], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
      
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
  
    # OneCycleLR (max_lr é atingido a 30% do treino, depois decai)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(loader),
        epochs=n_epochs,
        pct_start=0.3
    )
  
    best_pr_auc = 0.0
    patience_cnt = 0
    best_state = None
    history = []
  
    # 3. Loop de Treinamento
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
      
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
          
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
          
            # Step do OneCycleLR é feito A CADA BATCH
            scheduler.step()
          
            train_losses.append(loss.item())
          
        # 4. Avaliação e Early Stopping
        model.eval()
        with torch.no_grad():
            X_val_t, y_val_t = X_val_t.to(device), y_val_t.to(device)
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
          
            val_pr_auc = average_precision_score(y_val_t.cpu().numpy(), val_probs)
            val_roc_auc = roc_auc_score(y_val_t.cpu().numpy(), val_probs)
          
        history.append({
            "epoch": epoch,
            "train_loss": np.mean(train_losses),
            "val_loss": val_loss,
            "val_pr_auc": val_pr_auc,
            "val_roc_auc": val_roc_auc
        })
      
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            patience_cnt = 0
            best_state = model.state_dict()
        else:
            patience_cnt += 1
          
        if patience_cnt >= patience:
            print(f"Early stopping na época {epoch}")
            break
          
    if best_state is not None:
        model.load_state_dict(best_state)
      
    return model, history
```

### Task 4: Optuna Tuning e MLflow Tracking

**Files:**

- Modify: `notebooks/06_mlp_advanced_loss.ipynb`

- [ ] **Step 1: Configurar a Tracking URI do MLflow e iniciar a busca do Optuna**

```python
# Configuração do MLflow
mlflow.set_tracking_uri(MLFLOW_DB)
mlflow.set_experiment(EXPERIMENT_NAME)

def objective(trial):
    """Função objetivo para otimização Bayesiana da rede com Focal Loss."""
  
    # Espaço de Busca da Arquitetura
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    hidden_size_1 = trial.suggest_categorical("hidden_size_1", [32, 64, 128])
    hidden_size_2 = trial.suggest_categorical("hidden_size_2", [16, 32, 64])
  
    # Espaço de Busca da Topologia de Loss
    focal_gamma = trial.suggest_float("focal_gamma", 0.0, 5.0)
    focal_alpha = trial.suggest_float("focal_alpha", 0.1, 0.9)
    max_lr = trial.suggest_float("max_lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
  
    hidden_dims = [hidden_size_1, hidden_size_2]
    model = ChurnMLP(INPUT_DIM, hidden_dims, dropout_rate).to(device)
  
    # Treinamento
    model, history = train_mlp_advanced(
        model=model,
        X_tr_np=X_tr.values,
        y_tr_np=y_tr.values,
        X_val_np=X_val.values,
        y_val_np=y_val.values,
        loss_type="focal",
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        max_lr=max_lr,
        weight_decay=weight_decay
    )
  
    hist_df = pd.DataFrame(history)
    return hist_df['val_pr_auc'].max()

# Instanciar e rodar o estudo (limitado a 20 para prototipagem rápida, ajuste conforme necessário)
study = optuna.create_study(direction="maximize", study_name="focal_loss_tuning")
study.optimize(objective, n_trials=N_TRIALS_OPTUNA)

print(f"Melhor PR-AUC: {study.best_value}")
print(f"Melhores parâmetros: {study.best_params}")
```

- [ ] **Step 2: Treinar e Registrar o Melhor Modelo no MLflow**

```python
best_params = study.best_params

# Recriar e treinar o modelo com os melhores hiperparâmetros
best_hidden_dims = [best_params["hidden_size_1"], best_params["hidden_size_2"]]
final_model = ChurnMLP(INPUT_DIM, best_hidden_dims, best_params["dropout_rate"]).to(device)

final_model, history = train_mlp_advanced(
    model=final_model,
    X_tr_np=X_tr.values,
    y_tr_np=y_tr.values,
    X_val_np=X_val.values,
    y_val_np=y_val.values,
    loss_type="focal",
    focal_gamma=best_params["focal_gamma"],
    focal_alpha=best_params["focal_alpha"],
    max_lr=best_params["max_lr"],
    weight_decay=best_params["weight_decay"]
)

# Avaliação final no Test Set
final_model.eval()
with torch.no_grad():
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)
  
    test_logits = final_model(X_test_t)
    test_probs = torch.sigmoid(test_logits).cpu().numpy()
  
    # Limiar padrão 0.5 (você pode rodar a otimização de threshold depois se necessário)
    test_preds = (test_probs >= 0.5).astype(int)
    test_pr_auc = average_precision_score(y_test, test_probs)

# Registrar artefato e hiperparâmetros no MLflow
with mlflow.start_run(run_name="MLP_Focal_OneCycleLR"):
    mlflow.log_params(best_params)
    mlflow.log_metric("test_pr_auc", test_pr_auc)
  
    # Signature input_example (Clean Code para evitar warnings)
    input_example = X_test.head(1).values.astype(np.float32)
  
    mlflow.pytorch.log_model(
        final_model, 
        artifact_path="model",
        registered_model_name="MLP_Focal_OneCycleLR",
        input_example=input_example
    )
  
    print(f"Test PR-AUC final: {test_pr_auc:.4f}")
```
