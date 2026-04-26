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
    invalid_values = ['No', 'No internet service', 'No phone service']
    df['total_services_count'] = (~df[service_cols].isin(invalid_values)).sum(axis=1)

    protection_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    df['has_protection_services'] = df[protection_cols].isin(['Yes']).any(axis=1).astype(int)

    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])

    return df
