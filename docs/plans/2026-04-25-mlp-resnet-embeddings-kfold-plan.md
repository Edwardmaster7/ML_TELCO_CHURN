# ResNet Embeddings e MLOps Anti-Overfitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar arquitetura de ResNet Tabular com Entity Embeddings blindada contra overfitting (Gaussian Noise + Stratified K-Fold), aplicando rigorosos princípios de Clean Architecture transferindo toda a lógica para os módulos `src/`.

**Architecture:** A lógica do projeto será distribuída seguindo S.O.L.I.D:
1. `src/features/build_features.py`: Isola a criação de variáveis de negócio.
2. `src/features/preprocessing.py`: Fábricas de ColumnTransformers (OHE e Ordinal).
3. `src/data/datasets.py`: Classes `torch.utils.data.Dataset`.
4. `src/models/tabular_resnet.py`: Classes `nn.Module` (FocalLoss, ResNetBlock, AdvancedChurnMLP, GaussianNoise).
5. `src/models/train_model.py`: Função pura do loop de treinamento.
O `notebooks/07_mlp_resnet_embeddings.ipynb` será apenas um orquestrador enxuto para o Optuna e MLflow.

**Tech Stack:** PyTorch, Scikit-Learn, Optuna, MLflow.

---

### Task 1: Clean Architecture - Camada de Features e Dados

**Files:**
- Create: `src/features/build_features.py`
- Create: `src/features/preprocessing.py`
- Create: `src/data/datasets.py`

- [ ] **Step 1: Isolar a Engenharia de Features**
Criar `src/features/build_features.py` para isolar as regras de negócio.

```python
"""Módulo de engenharia de features de negócio para detecção de Churn."""
import numpy as np
import pandas as pd

def engineer_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica transformações e cria variáveis focadas em engajamento e finanças."""
    df = df.copy()
    
    df['is_monthly_contract'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['is_new_customer'] = (df['Tenure'] <= 6).astype(int)

    df['charges_per_tenure'] = np.where(
        df['Tenure'] > 0, df['TotalCharges'] / df['Tenure'], df['MonthlyCharges']
    )
    high_spender_threshold = df['MonthlyCharges'].quantile(0.75)
    df['is_high_spender'] = (df['MonthlyCharges'] >= high_spender_threshold).astype(int)

    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['total_services_count'] = df.apply(
        lambda row: sum(1 for col in service_cols if row[col] not in ['No', 'No internet service', 'No phone service']), 
        axis=1
    )
    
    protection_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    df['has_protection_services'] = df.apply(
        lambda row: int(any(row[col] == 'Yes' for col in protection_cols)), 
        axis=1
    )
    
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
        
    return df
```

- [ ] **Step 2: Isolar os Pipelines do Scikit-Learn (Fábricas de Preprocessors)**
Criar `src/features/preprocessing.py`.

```python
"""Módulo de fábricas para pipelines de pré-processamento do Scikit-Learn."""
from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

def build_ordinal_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Constrói pipeline que transforma categóricas em inteiros para camadas nn.Embedding."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    return ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

def build_ohe_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Constrói pipeline padrão com One-Hot Encoding para modelos lineares."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
```

- [ ] **Step 3: Isolar a Camada de Dataset PyTorch**
Criar `src/data/datasets.py`.

```python
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
        self.X_cat = torch.tensor(X_proc[:, num_features_cnt:], dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple:
        return self.X_num[idx], self.X_cat[idx], self.y[idx]
```

### Task 2: Clean Architecture - Camada de Modelagem (PyTorch)

**Files:**
- Create: `src/models/tabular_resnet.py`
- Create: `src/models/train_model.py`

- [ ] **Step 1: Extrair a Macro-Arquitetura da Rede**
Criar `src/models/tabular_resnet.py` contendo toda a matemática da rede neural.

```python
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
```

- [ ] **Step 2: Extrair a Lógica de Treinamento**
Criar `src/models/train_model.py` como função pura e isolada.

```python
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
```

### Task 3: O Orquestrador (Jupyter Notebook Refatorado)

**Files:**
- Modify: `notebooks/07_mlp_resnet_embeddings.ipynb` (Sobrescreva todas as células existentes)

- [ ] **Step 1: O Notebook Limpo - Imports, Configs e Datasets**

```python
"""
Notebook 07: Entity Embeddings, ResNet Tabular e K-Fold Anti-Overfitting
Clean Architecture: Notebook atuando como camada de orquestração e apresentação (MLflow/Optuna).
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import optuna
import mlflow
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import average_precision_score, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join('..')))

# Importando do pacote `src` (Clean Code)
from src.ml_telco_churn.config import CONFIG
from src.features.build_features import engineer_advanced_features
from src.features.preprocessing import build_ordinal_preprocessor, build_ohe_preprocessor
from src.data.datasets import ChurnEmbeddingDataset
from src.models.tabular_resnet import AdvancedChurnMLP, FocalLoss
from src.models.train_model import train_advanced_model

# Constantes locais
RANDOM_STATE = CONFIG.random_state
TEST_SIZE = 0.2
BATCH_SIZE = 256
N_EPOCHS = 150
PATIENCE = 15
N_TRIALS_OPTUNA = 15
N_SPLITS = 3
EXPERIMENT_NAME = "05_PyTorch_ResNet_Embeddings"

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# 1. Carregamento e Feature Engineering
df_cust = pd.read_csv('../notebooks/data/raw/churn_customers.csv')
df_serv = pd.read_csv('../notebooks/data/raw/churn_services.csv')
df_cont = pd.read_csv('../notebooks/data/raw/churn_contracts.csv')

df_raw = df_cust.merge(df_serv, on='customerID').merge(df_cont, on='customerID')
df_raw = df_raw.rename(columns={'customerID': 'CustomerID', 'tenure': 'Tenure', 'gender': 'Gender'})
df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
df_raw['Churn'] = df_raw['Churn'].map({'Yes': 1, 'No': 0})

df_feat = engineer_advanced_features(df_raw)

target_col = 'Churn'
X = df_feat.drop(columns=[target_col])
y = df_feat[target_col]

num_cols = X.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Holdout Cego 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

preprocessor_base = build_ordinal_preprocessor(num_cols, cat_cols)
num_features_idx = len(num_cols)
```

- [ ] **Step 2: Optuna K-Fold Objective no Notebook**

```python
mlflow.set_tracking_uri("sqlite:///../mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

def objective_kfold(trial):
    # Pruning no Espaço de Busca (Anti-Overfitting)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64])
    num_blocks = trial.suggest_int("num_blocks", 1, 2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    noise_sigma = trial.suggest_float("noise_sigma", 0.01, 0.1)
    
    focal_gamma = trial.suggest_float("focal_gamma", 0.0, 5.0)
    focal_alpha = trial.suggest_float("focal_alpha", 0.1, 0.9)
    max_lr = trial.suggest_float("max_lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 5e-3, log=True)
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []
    
    X_train_np = X_train.values
    y_train_np = y_train.values.astype(np.float32)
    
    for train_idx, val_idx in skf.split(X_train_np, y_train_np):
        X_fold_tr_raw = pd.DataFrame(X_train_np[train_idx], columns=X.columns)
        X_fold_val_raw = pd.DataFrame(X_train_np[val_idx], columns=X.columns)
        y_fold_tr = y_train_np[train_idx]
        y_fold_val = y_train_np[val_idx]
        
        prep_fold = clone(preprocessor_base)
        X_tr_proc = prep_fold.fit_transform(X_fold_tr_raw)
        X_val_proc = prep_fold.transform(X_fold_val_raw)
        
        # Mapeando unknowns (-1) para 0 (padding_idx), e classes conhecidas para 1..N
        X_tr_proc[:, num_features_idx:] += 1
        X_val_proc[:, num_features_idx:] += 1
        cardinalities = [len(prep_fold.transformers_[1][1].named_steps['cat'].named_steps['ordinal'].categories_[i]) + 2 for i in range(len(cat_cols))]
        
        ds_tr = ChurnEmbeddingDataset(X_tr_proc, y_fold_tr, num_features_idx)
        ds_val = ChurnEmbeddingDataset(X_val_proc, y_fold_val, num_features_idx)
        ld_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
        
        model = AdvancedChurnMLP(num_features_idx, cardinalities, hidden_dim, num_blocks, dropout_rate, noise_sigma).to(device)
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma).to(device)
        
        _, fold_auc = train_advanced_model(
            model, criterion, ld_tr, ds_val, device, 
            focal_gamma, focal_alpha, max_lr, weight_decay, N_EPOCHS, PATIENCE
        )
        fold_scores.append(fold_auc)
        
    return np.mean(fold_scores)

study = optuna.create_study(direction="maximize", study_name="resnet_kfold_tuning")
study.optimize(objective_kfold, n_trials=N_TRIALS_OPTUNA)
print(f"Melhor PR-AUC Médio no K-Fold: {study.best_value:.4f}")
```

- [ ] **Step 3: Avaliação Final, Logging MLOps e Gráficos de Comparação**

```python
best = study.best_params

# Retreino em 100% da base de treino
final_preprocessor = clone(preprocessor_base)
X_tr_final_proc = final_preprocessor.fit_transform(X_train)
X_test_final_proc = final_preprocessor.transform(X_test)

X_tr_final_proc[:, num_features_idx:] += 1
X_test_final_proc[:, num_features_idx:] += 1
final_cards = [len(final_preprocessor.transformers_[1][1].named_steps['cat'].named_steps['ordinal'].categories_[i]) + 2 for i in range(len(cat_cols))]

ds_tr_final = ChurnEmbeddingDataset(X_tr_final_proc, y_train.values, num_features_idx)
ld_tr_final = DataLoader(ds_tr_final, batch_size=BATCH_SIZE, shuffle=True)
ds_test_final = ChurnEmbeddingDataset(X_test_final_proc, y_test.values, num_features_idx)

final_model = AdvancedChurnMLP(num_features_idx, final_cards, best["hidden_dim"], best["num_blocks"], best["dropout_rate"], best["noise_sigma"]).to(device)
final_criterion = FocalLoss(alpha=best["focal_alpha"], gamma=best["focal_gamma"]).to(device)

final_model, _ = train_advanced_model(
    final_model, final_criterion, ld_tr_final, ds_test_final, device, 
    best["focal_gamma"], best["focal_alpha"], best["max_lr"], best["weight_decay"], N_EPOCHS, PATIENCE
)

final_model.eval()
with torch.no_grad():
    X_num_test, X_cat_test = ds_test_final.X_num.to(device), ds_test_final.X_cat.to(device)
    test_probs = torch.sigmoid(final_model(X_num_test, X_cat_test)).cpu().numpy()
    test_pr_auc = average_precision_score(ds_test_final.y.numpy(), test_probs)

print(f"Test PR-AUC Vencedor ResNet K-Fold: {test_pr_auc:.4f}")

# MLflow Logging com Pipeline Salvo!
with mlflow.start_run(run_name="MLP_ResNet_KFold"):
    mlflow.log_params(best)
    mlflow.log_metric("test_pr_auc", test_pr_auc)
    
    # 1. Logar o preprocessor exigido pela Governança para o FastAPI
    mlflow.sklearn.log_model(final_preprocessor, "preprocessor")
    
    final_model.cpu()
    sample_num = ds_test_final.X_num[:1].numpy().astype(np.float32)
    sample_cat = ds_test_final.X_cat[:1].numpy().astype(np.int64)
    out_sig = final_model(torch.tensor(sample_num), torch.tensor(sample_cat)).detach().numpy()
    
    from mlflow.models.signature import infer_signature
    sig = infer_signature({"x_num": sample_num, "x_cat": sample_cat}, out_sig)
    
    mlflow.pytorch.log_model(
        final_model, 
        name="model",
        registered_model_name="MLP_ResNet_Embeddings",
        signature=sig
    )

# --- Comparação Baseline e Gráficos ---
lr_preprocessor = build_ohe_preprocessor(num_cols, cat_cols)
X_tr_lr = lr_preprocessor.fit_transform(X_train)
X_test_lr = lr_preprocessor.transform(X_test)

lr_model = LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE, max_iter=1000)
lr_model.fit(X_tr_lr, y_train)
lr_probs = lr_model.predict_proba(X_test_lr)[:, 1]
lr_pr_auc = average_precision_score(y_test, lr_probs)

# Histórico das MLPs Anteriores
mlp_adv_loss_pr = 0.6512
mlp_vanilla_pr = 0.6392

print(f"LogReg Baseline Test PR-AUC: {lr_pr_auc:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

RocCurveDisplay.from_predictions(y_test, test_probs, name="ResNet Embeddings K-Fold", ax=ax[0])
RocCurveDisplay.from_predictions(y_test, lr_probs, name="Logistic Regression Baseline", ax=ax[0], linestyle="--")
ax[0].set_title("Comparação ROC Curve no Test Set")

PrecisionRecallDisplay.from_predictions(y_test, test_probs, name=f"ResNet (PR-AUC={test_pr_auc:.3f})", ax=ax[1])
PrecisionRecallDisplay.from_predictions(y_test, lr_probs, name=f"LogReg (PR-AUC={lr_pr_auc:.3f})", ax=ax[1], linestyle="--")
ax[1].axhline(y=mlp_adv_loss_pr, color='r', linestyle=':', label=f"MLP FocalLoss Teto ({mlp_adv_loss_pr:.3f})")
ax[1].axhline(y=mlp_vanilla_pr, color='g', linestyle=':', label=f"MLP Vanilla Teto ({mlp_vanilla_pr:.3f})")
ax[1].legend()
ax[1].set_title("Comparação Precision-Recall Curve no Test Set")

plt.tight_layout()
plt.show()

with mlflow.start_run(run_name="MLP_ResNet_KFold", nested=True) as run:
    fig.savefig("resnet_comparison.png")
    mlflow.log_artifact("resnet_comparison.png")
```