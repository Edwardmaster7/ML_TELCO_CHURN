import pandas as pd
import numpy as np
import logging
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)
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
        logger.error("Tentativa de inicializar preprocessor com lista de features nula ou vazia.")
        raise ValueError("As listas de features categóricas e numéricas não podem ser vazias ou nulas.")

    # Tratamento de case nas features injetadas
    cat_features_clean = [f.strip().lower() for f in cat_features]
    num_features_clean = [f.strip().lower() for f in num_features]
    available_clean = [f.strip().lower() for f in available_columns]

    # Validação: Verifica se as features injetadas existem no dataset
    all_injected = set(cat_features_clean + num_features_clean)
    missing_cols = all_injected - set(available_clean)

    if missing_cols:
        logger.error(f"Features injetadas não encontradas no dataset (Possível Data Leakage ou Desalinhamento): {missing_cols}")
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