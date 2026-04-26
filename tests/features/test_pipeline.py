import pandas as pd
import pytest
from src.features.pipeline import clean_data

def test_clean_data_advanced_features():
    # Cria um DataFrame mock similar aos raw tables mesclados
    data = {
        'customerid': ['A', 'B'],
        'churn': ['Yes', 'No'],
        'totalcharges': ['100.0', ' '],
        'tenure': [5, 10],
        'monthlycharges': [50.0, 100.0],
        'contract': ['Month-to-month', 'Two year'],
        'onlinesecurity': ['Yes', 'No'],
        'onlinebackup': ['Yes', 'No'],
        'deviceprotection': ['No', 'No'],
        'techsupport': ['No', 'No'],
        'streamingtv': ['No', 'No'],
        'streamingmovies': ['No', 'No']
    }
    df_raw = pd.DataFrame(data)

    # Processa os dados
    df_clean = clean_data(df_raw)

    # Verifica a conversão binária
    assert df_clean.loc[0, 'onlinesecurity'] == 1
    assert df_clean.loc[1, 'onlinesecurity'] == 0

    # Verifica novas features (derived)
    assert 'is_monthly_contract' in df_clean.columns
    assert df_clean.loc[0, 'is_monthly_contract'] == 1

    assert 'is_new_customer' in df_clean.columns
    assert df_clean.loc[0, 'is_new_customer'] == 1

    assert 'charges_per_tenure' in df_clean.columns

    assert 'total_services_count' in df_clean.columns
    assert df_clean.loc[0, 'total_services_count'] == 2
