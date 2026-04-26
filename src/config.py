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
        'tenure', 'monthlycharges', 'totalcharges',
        'charges_per_tenure', 'total_services_count',
        'is_monthly_contract', 'has_protection_services',
        'is_high_spender', 'is_new_customer'
    ])
    cat_features: List[str] = field(default_factory=lambda: [
        'gender', 'partner', 'dependents', 'phoneservice', 'multiplelines',
        'internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection',
        'techsupport', 'streamingtv', 'streamingmovies', 'contract',
        'paperlessbilling', 'paymentmethod'
    ])

    # Tracking
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment_name: str = "03_Refactor_Src"

    # Hiperparâmetros Campeões (Focal Loss + K-Fold) extraídos do notebook 06 (Trial 5)
    best_params: dict = field(default_factory=lambda: {
        'dropout_rate': 0.385,
        'hidden_size_1': 32,
        'hidden_size_2': 16,
        'focal_gamma': 3.150,
        'focal_alpha': 0.727,
        'max_lr': 0.0046,
        'weight_decay': 0.00025
    })

# Instância global para uso nos scripts (pode ser injetada em funções p/ SOLID)
CONFIG = ProjectConfig()
