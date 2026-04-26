import pandas as pd
import numpy as np
import logging
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpezas básicas e feature engineering avançada (Data-Centric)."""
    df_clean = df.copy()

    # Tratamento de case: padroniza nomes das colunas (lowercase, sem espaços)
    df_clean.columns = df_clean.columns.str.strip().str.lower()

    if 'totalcharges' in df_clean.columns:
        df_clean['totalcharges'] = pd.to_numeric(df_clean['totalcharges'], errors='coerce')
        # Preencher NaNs em vez de dropar, seguindo o notebook
        median_tc = df_clean['totalcharges'].median()
        df_clean['totalcharges'] = df_clean['totalcharges'].fillna(median_tc)

    # Transformação Binária de Colunas Yes/No (excluindo target que é tratado depois)
    for col in df_clean.select_dtypes("object").columns:
        if df_clean[col].dropna().isin(['yes', 'no', 'Yes', 'No']).all() and col != 'churn':
            df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})

    # Advanced Feature Engineering (do Notebook 04)
    if 'contract' in df_clean.columns:
        df_clean['is_monthly_contract'] = (df_clean['contract'].str.lower() == 'month-to-month').astype(int)

    if 'tenure' in df_clean.columns:
        df_clean['is_new_customer'] = (df_clean['tenure'] <= 6).astype(int)

    if 'totalcharges' in df_clean.columns and 'tenure' in df_clean.columns:
        df_clean['charges_per_tenure'] = df_clean['totalcharges'] / (df_clean['tenure'] + 1)

    if 'monthlycharges' in df_clean.columns:
        q75_monthly = df_clean['monthlycharges'].quantile(0.75)
        df_clean['is_high_spender'] = (df_clean['monthlycharges'] > q75_monthly).astype(int)

    # Features de Serviços
    service_cols = ['onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies']
    present_services = [c for c in service_cols if c in df_clean.columns]
    if present_services:
        cond = (df_clean[present_services] == 'yes') | (df_clean[present_services] == 'Yes') | (df_clean[present_services] == 1)
        df_clean['total_services_count'] = cond.sum(axis=1)

    protection_cols = ['onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport']
    present_prot = [c for c in protection_cols if c in df_clean.columns]
    if present_prot:
        cond_prot = (df_clean[present_prot] == 'yes') | (df_clean[present_prot] == 'Yes') | (df_clean[present_prot] == 1)
        df_clean['has_protection_services'] = cond_prot.any(axis=1).astype(int)

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