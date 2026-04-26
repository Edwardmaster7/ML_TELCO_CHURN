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
    mlflow_experiment_name: str = "03_Refactor_Src"

    # Hiperparâmetros Campeões (Focal Loss + K-Fold) extraídos do notebook 06
    best_params: dict = field(default_factory=lambda: {
        'dropout_rate': 0.364,
        'hidden_size_1': 64,
        'hidden_size_2': 32,
        'focal_gamma': 0.082,
        'focal_alpha': 0.778,
        'max_lr': 0.0039,
        'weight_decay': 0.0049
    })

# Instância global para uso nos scripts (pode ser injetada em funções p/ SOLID)
CONFIG = ProjectConfig()
