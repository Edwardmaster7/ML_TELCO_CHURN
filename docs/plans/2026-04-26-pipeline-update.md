# Pipeline Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adaptar a pipeline de produção em `src/features/pipeline.py` e `src/config.py` para espelhar a feature engineering Data-Centric do Notebook 04 e os hiperparâmetros ganhadores (K-Fold + Focal Loss) do Notebook 06.

**Architecture:** Modificaremos a função `clean_data` para injetar as features avançadas temporal-financeiras no dataset on-the-fly, e adaptaremos o `get_preprocessor` e o script de treinamento se necessário para consumir os hiperparâmetros campeões corretos. Tudo com Test-Driven Development para garantir resiliência.

**Tech Stack:** Python, Pandas, Scikit-Learn, PyTorch, Pytest.

---

### Task 1: Atualizar Testes Unitários de Feature Engineering

**Files:**
- Modify: `tests/features/test_pipeline.py:1-50` (ou crie se não existir)

- [ ] **Step 1: Write the failing test para features avançadas**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/features/test_pipeline.py -v`
Expected: FAIL (As novas features não existem ainda).

- [ ] **Step 3: Write minimal implementation in `src/features/pipeline.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/features/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/features/test_pipeline.py src/features/pipeline.py
git commit -m "feat: implementa features data-centric avancadas na pipeline"
```

---

### Task 2: Atualizar Testes de Configuração (Hiperparâmetros)

**Files:**
- Modify: `tests/test_config.py:1-20` (ou crie se não existir)

- [ ] **Step 1: Write the failing test**

```python
import pytest
from src.config import CONFIG

def test_config_best_params_updated():
    # Verifica se os hiperparâmetros refletem o Trial 5 do KFold do nb 06
    assert CONFIG.best_params['dropout_rate'] == 0.385
    assert CONFIG.best_params['hidden_size_1'] == 32
    assert CONFIG.best_params['hidden_size_2'] == 16
    assert CONFIG.best_params['focal_gamma'] == 3.150
    assert CONFIG.best_params['focal_alpha'] == 0.727
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL (Os valores em src/config.py estão desatualizados).

- [ ] **Step 3: Write minimal implementation em `src/config.py`**

```python
    # Atualize apenas o bloco best_params em src/config.py
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
```
*Note: Apenas modifique a linha do dicionário.*

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_config.py src/config.py
git commit -m "fix: atualiza hiperparametros p/ bater c/ notebook 06"
```

---

### Task 3: Atualizar Listas de Features Injetadas (`src/config.py`)

**Files:**
- Modify: `src/config.py:24-34`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
def test_config_feature_lists():
    # Deve conter as novas colunas derivadas
    assert 'charges_per_tenure' in CONFIG.num_features
    assert 'total_services_count' in CONFIG.num_features
    assert 'is_monthly_contract' in CONFIG.num_features
    
    assert 'has_protection_services' in CONFIG.num_features
    assert 'is_high_spender' in CONFIG.num_features
    assert 'is_new_customer' in CONFIG.num_features
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation em `src/config.py`**

```python
    # Adicione as colunas na lista `num_features`
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: registra novas colunas na config global de features"
```

---

### Task 4: Treinar Modelo e Validar Métricas Finais

**Files:**
- Execution only: `src/models/train.py`

- [ ] **Step 1: Validar execução do pipeline e métricas no console**

Run: `uv run python src/models/train.py --epochs 30`
Expected: A execução deve compilar sem erro de features ausentes (garantia do Validator no preprocessor). O PR-AUC Médio de Validação (KFold) deve subir de volta para a casa de `0.65+` a `0.66+`.

- [ ] **Step 2: Commit**

```bash
git commit --allow-empty -m "test: valida execucao da arquitetura ajustada"
```