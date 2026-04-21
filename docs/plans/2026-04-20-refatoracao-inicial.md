# Refatoração Inicial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modularizar os notebooks de Data Science em scripts Python puros para o treinamento e salvar os artefatos de modelo via MLflow.

**Architecture:** A estrutura `src/` será composta de módulos independentes (`config.py`, `data.py`, `features.py` e `train.py`). O `config.py` centralizará todas as constantes seguindo princípios SOLID (separação de responsabilidades e injeção de dependências onde possível). Os scripts encapsularão os fluxos do pandas, scikit-learn pipelines e a rede MLP do PyTorch de forma estruturada.

**Tech Stack:** Python 3.13, Pandas, Scikit-learn, PyTorch, MLflow.

---

### Task 1: Setup da Estrutura de Pastas e Constantes (Clean Code)

**Files:**
- Create: `src/ml_telco_churn/config.py`
- Create: `src/ml_telco_churn/__init__.py`

- [ ] **Step 1: Criar o arquivo de configurações centralizadas**
```python
# src/ml_telco_churn/config.py
"""Configurações e constantes globais do projeto."""

from dataclasses import dataclass, field
from typing import List

@dataclass
class ProjectConfig:
    # Seeds
    random_state: int = 42

    # Arquitetura MLP
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    mlp_dropout_rate: float = 0.3

    # Treinamento
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3

    # Features e Colunas Target (minúsculo para consistência de case treatment)
    target_col: str = "churn"
    id_col: str = "customerid"

    # Features Lists
    num_features: List[str] = field(default_factory=lambda: [
        'tenure', 'MonthlyCharges', 'TotalCharges'
    ])
    cat_features: List[str] = field(default_factory=lambda: [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ])

    # Tracking
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment_name: str = "telco_churn_models_src"

# Instância global para uso nos scripts (pode ser injetada em funções p/ SOLID)
CONFIG = ProjectConfig()
```

- [ ] **Step 2: Commit do Módulo de Configuração**
```bash
git add src/ml_telco_churn/
git commit -m "feat: cria estrutura e configurações base centralizadas"
```

### Task 1.1: Módulo de Dados

**Files:**
- Create: `src/ml_telco_churn/data.py`

- [ ] **Step 1: Criar a estrutura base e escrever a função de carregamento**
```python
# src/ml_telco_churn/__init__.py
"""ML Telco Churn Package."""
```

```python
# src/ml_telco_churn/data.py
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_and_merge_data(
    path_customers: str, 
    path_services: str, 
    path_contracts: str
) -> pd.DataFrame:
    """Carrega os dados particionados em 3 tabelas e faz o merge por CustomerID."""
    try:
        df_customers = pd.read_csv(path_customers)
        df_services = pd.read_csv(path_services)
        df_contracts = pd.read_csv(path_contracts)
        
        df = df_customers.merge(df_services, on="customerID", how="inner")
        df = df.merge(df_contracts, on="customerID", how="inner")
        
        logger.info(f"Shape final do merge: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Erro ao carregar os dados: {e}")
        raise
```

- [ ] **Step 2: Commit da estrutura de dados**
```bash
git add src/ml_telco_churn/data.py
git commit -m "feat: cria módulo de carregamento de dados"
```

### Task 2: Implementar Feature Engineering e Pipeline do Scikit-Learn

**Files:**
- Create: `src/ml_telco_churn/features.py`

- [ ] **Step 1: Escrever a função de construção de Pipeline do sklearn**
```python
# src/ml_telco_churn/features.py
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpezas básicas (cast de tipos, dropna) e padroniza nomes."""
    df_clean = df.copy()

    # Tratamento de case: padroniza nomes das colunas (lowercase, sem espaços)
    df_clean.columns = df_clean.columns.str.strip().str.lower()

    if 'totalcharges' in df_clean.columns:
        df_clean['totalcharges'] = pd.to_numeric(df_clean['totalcharges'], errors='coerce')
        df_clean.dropna(subset=['totalcharges'], inplace=True)

    return df_clean

def get_preprocessor(cat_features: list[str], num_features: list[str], available_columns: list[str]) -> ColumnTransformer:
    """Retorna o ColumnTransformer com validação rigorosa das features injetadas."""
    if not cat_features or not num_features:
        raise ValueError("As listas de features categóricas e numéricas não podem ser vazias ou nulas.")

    # Tratamento de case nas features injetadas
    cat_features_clean = [f.strip().lower() for f in cat_features]
    num_features_clean = [f.strip().lower() for f in num_features]
    available_clean = [f.strip().lower() for f in available_columns]

    # Validação: Verifica se as features injetadas existem no dataset
    all_injected = set(cat_features_clean + num_features_clean)
    missing_cols = all_injected - set(available_clean)

    if missing_cols:
        raise ValueError(f"As seguintes features injetadas não foram encontradas no dataset: {missing_cols}")

    # Numérico
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # Categórico
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_features_clean),
            ("cat", cat_pipe, cat_features_clean),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    return preprocessor

def prepare_target(y_series: pd.Series) -> np.ndarray:
    """Converte 'Yes'/'No' para 1/0."""
    return (y_series.astype(str).str.strip().str.lower() == 'yes').astype(int).values
```

- [ ] **Step 2: Commit do Módulo Features**
```bash
git add src/ml_telco_churn/features.py
git commit -m "feat: implementa pipelines de transformações do sklearn"
```

### Task 3: Implementar a Arquitetura da Rede Neural (PyTorch)

**Files:**
- Create: `src/ml_telco_churn/model_nn.py`

- [ ] **Step 1: Criar o módulo da Rede Neural**
```python
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
```

- [ ] **Step 2: Commit do modelo PyTorch**
```bash
git add src/ml_telco_churn/model_nn.py
git commit -m "feat: cria classe da rede neural ChurnMLP"
```

### Task 4: Criar o Script Principal de Treinamento (Orquestração e MLflow)

**Files:**
- Create: `src/ml_telco_churn/train.py`

- [ ] **Step 1: Implementar o fluxo principal orquestrando dados, features, treino e log**
```python
# src/ml_telco_churn/train.py
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from sklearn.model_selection import train_test_split

# Imports locais (depende de como o pacote foi construído, mas assumindo rodar via sys path)
from src.ml_telco_churn.config import CONFIG
from src.ml_telco_churn.data import load_and_merge_data
from src.ml_telco_churn.features import clean_data, get_preprocessor, prepare_target
from src.ml_telco_churn.model_nn import ChurnMLP

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Treina o modelo de Churn.")
    parser.add_argument("--customers", default="data/raw/churn_customers.csv")
    parser.add_argument("--services", default="data/raw/churn_services.csv")
    parser.add_argument("--contracts", default="data/raw/churn_contracts.csv")
    parser.add_argument("--epochs", type=int, default=10) # 10 p/ demo rapida, o nb usava 300
    args = parser.parse_args()

    # 1. Carregamento
    df_raw = load_and_merge_data(args.customers, args.services, args.contracts)
    df_clean = clean_data(df_raw)

    X = df_clean.drop(columns=[CONFIG.target_col, CONFIG.id_col], errors='ignore')
    y = prepare_target(df_clean[CONFIG.target_col])

    # 2. Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=CONFIG.random_state, stratify=y
    )

    # 3. Preprocessamento (Fit no Treino, Transform no Teste)
    preprocessor = get_preprocessor(
        cat_features=CONFIG.cat_features,
        num_features=CONFIG.num_features,
        available_columns=X_train_raw.columns.tolist()
    )
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)
    
    # Converte p/ Tensores
    X_train_t = torch.tensor(X_train_proc, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    
    # 4. Treinamento da Rede
    input_dim = X_train_t.shape[1]
    model = ChurnMLP(
        input_dim=input_dim, 
        hidden_dims=CONFIG.mlp_hidden_dims, 
        dropout_rate=CONFIG.mlp_dropout_rate
    )
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG.learning_rate)
    
    logger.info("Iniciando treinamento PyTorch...")
    model.train()
    
    epochs_to_run = args.epochs if args.epochs != 10 else CONFIG.epochs
    for epoch in range(epochs_to_run):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    logger.info("Treinamento finalizado.")
    
    # 5. MLflow Tracking
    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)
    
    with mlflow.start_run(run_name="mlp_pytorch_refactored"):
        # Log artifacts (A decisão principal da arquitetura)
        mlflow.sklearn.log_model(preprocessor, "preprocessor")
        mlflow.pytorch.log_model(model, "pytorch_model")
        
        mlflow.log_param("epochs", args.epochs)
        logger.info("Modelos registrados no MLflow.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Testar se o script executa (usará os dados reais do diretório)**
Run: `PYTHONPATH=. python src/ml_telco_churn/train.py --epochs 2`
Expected: INFO logs informando que carregou os dados, treinou por 2 épocas e registrou no MLflow.

- [ ] **Step 3: Commit do script de orquestração**
```bash
git add src/ml_telco_churn/train.py
git commit -m "feat: orquestra pipeline de treino e log de artefatos com mlflow"
```
