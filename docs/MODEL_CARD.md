# Model Card — MLP_Focal_KFold_Script

> **Versão deste documento:** 1.0.0 — 2026-05-02
> **Baseado em:** evidências diretas do código e configurações do repositório.


---

## A. Identificação

| Campo                                     | Valor                                                                                              |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **Nome do Modelo**                  | `MLP_Focal_KFold_Script`                                                                         |
| **Tipo**                            | Classificação binária — MLP PyTorch                                                            |
| **Versão de Produção**           | Alias `@production` no MLflow Model Registry                                                     |
| **Data da Última Run Documentada** | Abril / 2026 (datas nos ADRs: ADR-008, 25/04/2026)                                                 |
| **Proprietário / Equipe**          | Grupo 21 — FIAP Pós-Tech, Produtização de Modelos                                              |
| **Repositório**                    | `ML_TELCO_CHURN/`                                                                                |
| **Artefato do Modelo**              | `mlruns/` + `mlflow.db` (SQLite, local) — alias `models:/MLP_Focal_KFold_Script@production` |
| **Artefato do Preprocessor**        | `runs:/{run_id}/preprocessor` (no mesmo `mlflow.db`)                                           |
| **Código de Treinamento**          | `src/models/train.py`                                                                            |
| **Configuração**                  | `src/core/config.py` — instância global `CONFIG`                                             |
| **ADR de Decisão**                 | `docs/specs/adrs/ADR-008-decisao-modelo-campeao-api.md`                                          |
| **Relatório de Métricas**         | `docs/reports/2026-04-25-relatorio-final-modelagem.md`                                           |

---

## B. Problema e Objetivo

### O que o Modelo Prevê

O modelo prevê a **probabilidade de churn** de um cliente de telecomunicações, ou seja, a probabilidade de o cliente cancelar o contrato de serviço. A saída é um escalar contínuo `churn_probability` ∈ [0.0, 1.0], acompanhado de `churn_prediction` ∈ {0, 1} obtido por limiar fixo de 0.5.

### Interpretação da Saída

| Campo                 | Tipo      | Intervalo  | Significado                                                                 |
| --------------------- | --------- | ---------- | --------------------------------------------------------------------------- |
| `churn_probability` | `float` | [0.0, 1.0] | Probabilidade sigmoid de churn; valores próximos de 1.0 indicam alto risco |
| `churn_prediction`  | `int`   | {0, 1}     | 1 = propenso ao churn (prob ≥ 0.5); 0 = cliente provável a permanecer     |

**Atenção:** O limiar de 0.5 é fixo no código (`src/core/ml_service.py`, linha `prediction = 1 if probability >= 0.5 else 0`) e não é calibrado por custo de negócio. Recomenda-se revisão para ambiente de produção real (ver Seção L).

### Uso no Negócio (evidência: `README.md`, `docs/tech_challenge_decisions.md`)

O resultado é usado para **priorização de campanhas de retenção** pela equipe de CS/Marketing. Clientes com `churn_prediction=1` são candidatos a receber ofertas proativas. A racionalidade de custo documentada é:

- Falso Negativo (perder cliente): custo ~$10
- Falso Positivo (oferecer desconto desnecessário): custo ~$1

Esse assimétrico motivou a escolha de Focal Loss para penalizar mais os falsos negativos.

---

## C. Dados

### Fontes

O dataset é o **IBM Telco Customer Churn**, particionado em 3 arquivos CSV estáticos (evidência: `notebooks/data/raw/`):

| Arquivo                                    | Colunas                                                                                                                                                                                                                | Linhas (aprox.) |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `notebooks/data/raw/churn_customers.csv` | `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`                                                                                                                                             | ~7.043          |
| `notebooks/data/raw/churn_services.csv`  | `customerID` + 9 features de serviços (`PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) | ~7.043          |
| `notebooks/data/raw/churn_contracts.csv` | `customerID`, `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, **`Churn`** (TARGET)                                                                  | ~7.043          |

**Chave de entidade:** `customerID` (string, ex: `"7590-VHVEG"`). Merge por `inner join` em `src/data/loader.py → load_and_merge_data()`.

### Schema Completo do Dataset Mesclado (21 colunas + target)

| Coluna               | Tipo Raw        | Domínio / Exemplo                                   |
| -------------------- | --------------- | ---------------------------------------------------- |
| `customerID`       | string          | `"7590-VHVEG"`                                     |
| `gender`           | string          | `"Male"`, `"Female"`                             |
| `SeniorCitizen`    | int             | `0`, `1`                                         |
| `Partner`          | string          | `"Yes"`, `"No"`                                  |
| `Dependents`       | string          | `"Yes"`, `"No"`                                  |
| `tenure`           | int             | 0–72 (meses)                                        |
| `PhoneService`     | string          | `"Yes"`, `"No"`                                  |
| `MultipleLines`    | string          | `"Yes"`, `"No"`, `"No phone service"`          |
| `InternetService`  | string          | `"DSL"`, `"Fiber optic"`, `"No"`               |
| `OnlineSecurity`   | string          | `"Yes"`, `"No"`, `"No internet service"`       |
| `OnlineBackup`     | string          | `"Yes"`, `"No"`, `"No internet service"`       |
| `DeviceProtection` | string          | `"Yes"`, `"No"`, `"No internet service"`       |
| `TechSupport`      | string          | `"Yes"`, `"No"`, `"No internet service"`       |
| `StreamingTV`      | string          | `"Yes"`, `"No"`, `"No internet service"`       |
| `StreamingMovies`  | string          | `"Yes"`, `"No"`, `"No internet service"`       |
| `Contract`         | string          | `"Month-to-month"`, `"One year"`, `"Two year"` |
| `PaperlessBilling` | string          | `"Yes"`, `"No"`                                  |
| `PaymentMethod`    | string          | 4 valores literais                                   |
| `MonthlyCharges`   | float           | ≥ 0.0 (USD/mês)                                    |
| `TotalCharges`     | float ou string | ≥ 0.0 (pode ser string vazia → 0.0)                |
| **`Churn`**  | string          | **`"Yes"`** → 1, **`"No"`** → 0    |

### Janela Temporal e Snapshot

O dataset é um **snapshot estático** sem coluna de timestamp explícita. Não há série temporal estruturada. Isso implica:

- Não há risco de leakage temporal por ordenação de tempo entre treino e teste.
- Há risco latente de **concept drift** caso o dataset de inferência em produção represente uma distribuição de clientes diferente da snapshot histórica.

### Estratégia de Split (evidência: `src/models/train.py`)

```python
# Holdout cego — 80/20 estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# + StratifiedKFold(n_splits=3) no X_train para tuning robusto
```

- Proporção de churn preservada em todos os splits (estratificado).
- `random_state=42` garante reprodutibilidade.
- Não há split temporal porque o dataset não possui timestamp.

### Desbalanceamento de Classes

| Classe                    | Proporção Estimada |
| ------------------------- | -------------------- |
| Churn = 0 (não cancelou) | ~74%                 |
| Churn = 1 (cancelou)      | ~26%                 |

### Versionamento de Dados

**Desconhecido / Não Implementado.** Não foi encontrado DVC, hashing de arquivos, ou qualquer mecanismo de versionamento nos paths `notebooks/data/`, `.dvc/`, ou arquivos `*.dvc`. Os CSVs são estáticos no repositório. Procurado em: `notebooks/data/`, raiz do projeto, `.gitignore`.

### Privacidade e PII

**Parcialmente tratado.** O campo `customerID` é um identificador sintético opaco (ex: `"7590-VHVEG"`) sem dados pessoais identificáveis diretamente. Não há campos de nome, CPF, e-mail ou endereço. No entanto:

- `gender` e `SeniorCitizen` são atributos sensíveis que podem introduzir viés.
- `PaymentMethod` pode ser proxy para nível socioeconômico.
- Não foi encontrado processo de anonimização formal (`src/data/`, `notebooks/`). Dataset original é público (IBM).

---

## D. Features e Pré-processamento

### Transformações Raw → Features (evidência: `src/features/pipeline.py → clean_data()`)

| Etapa                                   | Detalhe                                                                                   |
| --------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Normalização de colunas**     | `.str.strip().str.lower()` — todos os nomes de colunas em minúsculas                  |
| **Coerção de `TotalCharges`** | `pd.to_numeric(..., errors='coerce')` → NaN preenchido pela **mediana** do batch |
| **Binarização Yes/No**          | Colunas com domínio `{"Yes", "No"}` → `{1, 0}` (exceto o target `churn`)          |
| **`is_monthly_contract`**       | 1 se `contract == "month-to-month"`                                                     |
| **`is_new_customer`**           | 1 se `tenure ≤ 6`                                                                      |
| **`charges_per_tenure`**        | `totalcharges / (tenure + 1)` — razão de gasto médio por mês                        |
| **`is_high_spender`**           | 1 se `monthlycharges > quantile(0.75)` **do batch atual** ⚠️                    |
| **`total_services_count`**      | Contagem de 6 serviços ativados (0–6)                                                   |
| **`has_protection_services`**   | 1 se pelo menos 1 dos 4 serviços de proteção estiver ativo                             |

> ⚠️ **Bug conhecido:** `is_high_spender` usa o quantil do batch de inferência, não do treino. Em predição unitária, o q75 calculado sobre 1 linha resulta sempre em 0. Ver Seção H e L.

### Preprocessor Scikit-Learn (evidência: `src/features/pipeline.py → get_preprocessor()`)

```
ColumnTransformer(remainder="passthrough"):
  ┌─ num_pipe → [9 features numéricas]
  │   SimpleImputer(strategy="median")
  │   StandardScaler()
  └─ cat_pipe → [15 features categóricas]
      SimpleImputer(strategy="most_frequent")
      OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
```

### Consistência Treino vs Inferência

O preprocessor é:

1. **Fitado exclusivamente em `X_train`** (`src/models/train.py`)
2. **Serializado via `mlflow.sklearn.log_model()`** (não joblib)
3. **Carregado na API via `mlflow.sklearn.load_model(f"runs:/{run_id}/preprocessor")`** (`src/core/ml_service.py`)

Isso garante que os parâmetros aprendidos (médias, desvios, categorias OHE) do treino são aplicados identicamente na inferência. **Exceção:** `is_high_spender` é recalculado dinamicamente em `clean_data()`, quebrando essa garantia.

---

## E. Modelo

### Algoritmo

**ChurnMLP** — Rede Neural Multi-Layer Perceptron (MLP) implementada em PyTorch.
Definição: `src/models/architectures.py → class ChurnMLP(nn.Module)`

### Arquitetura

```
Entrada: tensor float32 de shape [batch, input_dim]
    └─ input_dim ≈ 30–35 (após OneHotEncoder)

Camada 1: Linear(input_dim → 32) → BatchNorm1d(32) → ReLU → Dropout(0.385)
Camada 2: Linear(32 → 16)        → BatchNorm1d(16) → ReLU → Dropout(0.385)
Saída:    Linear(16 → 1)                                     ← logit escalar

Ativação final (inferência): torch.sigmoid(logit) → probabilidade [0,1]
```

### Função de Perda

**FocalLoss** (`src/models/architectures.py → class FocalLoss(nn.Module)`):

$$
\mathcal{L} = \alpha \cdot (1 - p_t)^{\gamma} \cdot \text{BCE}(\text{logits}, \text{targets})
$$

- `alpha = 0.727` — pondera a classe positiva (churn)
- `gamma = 3.150` — amplifica o foco em exemplos difíceis

### Hiperparâmetros Campeões (evidência: `src/core/config.py → CONFIG.best_params`)

```python
{
    'hidden_size_1': 32,
    'hidden_size_2': 16,
    'dropout_rate': 0.385,
    'focal_gamma': 3.150,
    'focal_alpha': 0.727,
    'max_lr': 0.0046,
    'weight_decay': 0.00025
}
```

Origem: Trial 5 do Optuna com K-Fold no notebook `notebooks/06_mlp_advanced_loss.ipynb`.

### Otimizador e Scheduler

| Componente     | Configuração                               |
| -------------- | -------------------------------------------- |
| Otimizador     | `AdamW(lr=0.0046, weight_decay=0.00025)`   |
| Scheduler      | `OneCycleLR(max_lr=0.0046, pct_start=0.3)` |
| Batch Size     | 64                                           |
| Epochs máx.   | 150 (via `--epochs` CLI)                   |
| Early Stopping | `patience=20` em `val_pr_auc`            |

### Export e Carregamento

| Artefato               | Método de Salvar                                                                 | Método de Carregar                                                        |
| ---------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Preprocessor (sklearn) | `mlflow.sklearn.log_model(..., serialization_format="skops")`                   | `mlflow.sklearn.load_model(f"runs:/{run_id}/preprocessor")`              |
| Modelo PyTorch         | `mlflow.pytorch.log_model(..., registered_model_name="MLP_Focal_KFold_Script")` | `mlflow.pytorch.load_model("models:/MLP_Focal_KFold_Script@production")` |

**Restrição arquitetural (ADR-003):** Proibido carregar artefatos via `joblib.load()` em produção.

---

## F. Treinamento

### Procedimento End-to-End (evidência: `src/models/train.py`)

1. `load_and_merge_data()` — carrega e faz merge dos 3 CSVs
2. `clean_data()` — limpeza + 6 features derivadas
3. `prepare_target()` — converte "Yes"/"No" → 1/0
4. `train_test_split(test_size=0.2, stratify=y, seed=42)` — holdout cego
5. `get_preprocessor().fit_transform(X_train)` — fit do pipeline Scikit-Learn
6. `StratifiedKFold(n_splits=3)` — 3 folds sobre X_train para métricas de validação robustas
7. `train_focal_model()` × 3 — treino com AdamW + OneCycleLR + EarlyStopping
8. Retreino do modelo final em 80% de X_train (20% interno para EarlyStopping)
9. Avaliação no `X_test` cego (nunca visto)
10. `mlflow.start_run()` — log de params, metrics e artefatos

### Reprodutibilidade

| Elemento              | Valor                                                      |
| --------------------- | ---------------------------------------------------------- |
| Seed global           | `random_state = 42` (definido em `src/core/config.py`) |
| Seed do KFold         | `random_state = 42`                                      |
| Python                | ≥ 3.13 (fixado em `pyproject.toml`)                     |
| Dependências         | Fixadas por `uv.lock` (arquivo de lock do `uv`)        |
| Comando reproduzível | `make train ARGS="--epochs 150"`                         |

### Comando de Treinamento Completo

**bash (Linux/macOS / Git Bash):**

```bash
uv sync
make db-upgrade
MLFLOW_TRACKING_URI=sqlite:///mlflow.db PYTHONPATH=. \
  uv run python src/models/train.py \
    --customers notebooks/data/raw/churn_customers.csv \
    --services  notebooks/data/raw/churn_services.csv \
    --contracts notebooks/data/raw/churn_contracts.csv \
    --epochs 150
```

**PowerShell (Windows — sem `make`):**

```powershell
uv sync
uv run mlflow db upgrade sqlite:///mlflow.db
$env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"; $env:PYTHONPATH="."; uv run python src/models/train.py `
  --customers notebooks/data/raw/churn_customers.csv `
  --services  notebooks/data/raw/churn_services.csv `
  --contracts notebooks/data/raw/churn_contracts.csv `
  --epochs 150
```

### Hardware / Tempo

**Desconhecido** — Não foram encontrados logs de tempo ou especificação de hardware nos arquivos do repositório. O código suporta CUDA, MPS (Apple Silicon) e CPU (`src/models/train.py`: `torch.device('cuda' if ... else 'mps' if ... else 'cpu')`). Em CPU, estimativa empírica: 3–10 min para 150 épocas com 7k registros.

---

## G. Avaliação

### Métricas Usadas (evidência: `src/models/train.py`, `docs/specs/adrs/ADR-001-metrica-primaria-pr-auc.md`)

| Métrica          | Tipo                | Justificativa                                                       |
| ----------------- | ------------------- | ------------------------------------------------------------------- |
| **PR-AUC**  | **Primária** | Robusta ao desbalanceamento (~26% positivo); não é inflada por TN |
| ROC-AUC           | Secundária         | Complementar; superestima o desempenho com classes desbalanceadas   |
| F1-Score          | Secundária         | Equilíbrio Precision/Recall no limiar de 0.5                       |
| Precision         | Monitoramento       | Taxa de acerto nas predições positivas                            |
| Recall            | Monitoramento       | Cobertura dos churners reais                                        |
| mean_kfold_pr_auc | Interna             | Média dos 3 folds — sinal de generalização no X_train           |

**Métricas de regressão (MAE, RMSE, R², MAPE) não são calculadas, pois o problema é classificação binária.**

### Resultados Finais no Conjunto de Teste (evidência: `docs/reports/2026-04-25-relatorio-final-modelagem.md`)

| Modelo                                              | PR-AUC           | ROC-AUC          | F1     | Precision | Recall |
| --------------------------------------------------- | ---------------- | ---------------- | ------ | --------- | ------ |
| LogisticRegression_Advanced*(não em produção)* | **0.6624** | **0.8480** | 0.5952 | 0.6711    | 0.5348 |
| **MLP_Focal_KFold ← PRODUÇÃO**             | **0.6539** | 0.8456           | 0.5877 | 0.6300    | 0.5508 |
| MLP_Focal_OneCycleLR                                | 0.6534           | 0.8460           | 0.5812 | 0.6566    | 0.5214 |
| MLP_Vanilla_KFold                                   | 0.6495           | 0.8441           | 0.6220 | 0.4968    | 0.8316 |
| MLP_ResNet_Embeddings*(overfit)*                  | 0.6450           | 0.8388           | 0.5853 | 0.6503    | 0.5321 |

### Baselines de Referência

| Baseline                                 | PR-AUC                                 |
| ---------------------------------------- | -------------------------------------- |
| Prevendo sempre classe 0                 | ~0.26 (proporção da classe positiva) |
| LogisticRegression (features originais)  | ~0.652                                 |
| LogisticRegression (features avançadas) | ~0.662                                 |

### Análises de Erro Recomendadas (Não Implementadas no Repo)

As análises abaixo **não foram encontradas** em notebooks ou scripts. São recomendações baseadas em boas práticas:

- [ ] Matriz de confusão por segmento (`gender`, `SeniorCitizen`, `Contract`)
- [ ] Curva Precision-Recall com indicação do limiar ótimo por custo de negócio
- [ ] Análise por decil de probabilidade (distribuição de erros por faixa)
- [ ] Comparação de distribuição de `churn_probability` entre churners e não-churners
- [ ] Erro por `tenure` (clientes novos vs antigos têm comportamento diferente)

### Robustez e Drift Temporal

**Não avaliado.** O dataset é um snapshot estático único; não há splits por período nem avaliação de degradação ao longo do tempo. Esta é uma lacuna crítica para uso em produção real.

---

## H. Limitações e Riscos

### 1. Bug: `is_high_spender` em Inferência Unitária

**Descrição:** A feature `is_high_spender` é calculada em `src/features/pipeline.py → clean_data()` usando `df['monthlycharges'].quantile(0.75)` sobre o DataFrame de entrada. Em inferência com 1 cliente, o q75 é idêntico ao único valor, resultando em `is_high_spender = 0` sempre.

**Impacto:** A feature treinada com a distribuição do dataset (~25% dos clientes = 1) é silenciosamente enviada como 0 para todos os clientes em produção. Training-serving skew real.

**Mitigação sugerida:** Persistir `q75_monthly` do X_train como constante no `CONFIG` ou como artefato MLflow.

### 2. Dataset Estático (Snapshot)

O modelo foi treinado em uma única fotografia histórica dos clientes. Em produção real, novos produtos, campanhas de retenção ou mudanças de política alteram a distribuição de churn ao longo do tempo (concept drift). Não há mecanismo de retraining automático implementado.

### 3. Ausência de Calibração de Probabilidades

O modelo retorna probabilidades via sigmoid não calibradas. A sigmoid tende a produzir predições excessivamente confiantes. Para uso como score de priorização (top-K), a calibração pelo Brier Score ou Platt Scaling pode ser relevante. ADR-004 documenta a tentativa anterior de calibração via `IsotonicRegression`, descartada por comprimir o Recall sem ganho de PR-AUC.

### 4. Limiar Fixo de 0.5

O limiar de decisão não é otimizado pela assimetria de custos (FN=$10 vs FP=$1). Para maximizar o valor de negócio, o limiar ótimo deveria ser buscado via `precision_recall_curve` com a função de custo. Código atual: `src/core/ml_service.py`.

### 5. Viés por Atributos Sensíveis

Os campos `gender` e `SeniorCitizen` são usados como features (evidência: `CONFIG.cat_features` e `CONFIG.num_features` indiretamente via limpeza). Não foram realizados testes de equidade ou análise de disparidade de performance por grupo demográfico. Risco de discriminação sistemática.

### 6. Modelo Subótimo por Restrição Acadêmica

O modelo deployado (`MLP_Focal_KFold`, PR-AUC 0.6539) é inferior à Regressão Logística (`LogisticRegression_Advanced`, PR-AUC 0.6624). A escolha é explicitamente por conformidade ao Tech Challenge (ADR-008), não por performance. Para produção real, recomenda-se deploy da `LogisticRegression_Advanced`.

### 7. OOD — Clientes Fora da Distribuição

- Clientes com `tenure = 0` (novos): `charges_per_tenure = 0` e `is_new_customer = 1` — comportamento esperado, mas com incerteza alta.
- Categorias desconhecidas em OneHotEncoder: tratadas por `handle_unknown="ignore"` (codificadas como vetor zero), o que pode distorcer a predição silenciosamente.

---

## I. Uso Pretendido e Não Pretendido

### Usos Pretendidos

- Ranqueamento de clientes por propensão ao churn para campanhas proativas de retenção.
- Triagem de clientes para agentes de atendimento ao cliente (CS).
- Priorização de ofertas comerciais (descontos, upgrades de plano).

### Usos Não Pretendidos

- **Não usar como decisão automática sem revisão humana** — o modelo tem PR-AUC 0.65, o que implica ~35% de erro no espaço Precision-Recall. Decisões automatizadas de bloqueio ou cobrança são inadequadas.
- **Não usar para clientes de outros segmentos** — o modelo foi treinado exclusivamente no perfil IBM Telco. Empresas de outros setores (energia, saúde, SaaS) precisam retreinar.
- **Não usar como score de crédito ou elegibilidade** — o modelo não foi validado para esse uso e pode introduzir discriminação.
- **Não usar sem monitoramento de drift** — o snapshot de treinamento pode se tornar obsoleto em 3–6 meses.

---

## J. Requisitos Operacionais

| Requisito                         | Valor / Observação                                                                         |
| --------------------------------- | -------------------------------------------------------------------------------------------- |
| **Latência alvo**          | Não especificada formalmente. Middleware mede latência em ms (`src/core/middlewares.py`) |
| **Throughput**              | Inferência unitária síncrona. Sem endpoint batch implementado                             |
| **Memória mínima**        | Desconhecido. PyTorch MLP pequeno (~50K parâmetros); preprocessor sklearn leve              |
| **GPU**                     | Opcional. Suporta CUDA, MPS e CPU (`src/core/ml_service.py`)                               |
| **Dependências críticas** | `mlflow.db` + `mlruns/` devem estar íntegros e acessíveis no startup                   |
| **MLflow**                  | Servidor/URI acessível em `MLFLOW_TRACKING_URI` antes do startup da API                   |
| **Python**                  | ≥ 3.13                                                                                      |
| **Framework de Serving**    | FastAPI + Uvicorn (`src/main.py`)                                                          |
| **Porta**                   | 8000 (configurável via `uvicorn ... --port`)                                              |
| **Banco de Dados**          | SQLite via `aiosqlite` — tabela `prediction_logs` criada automaticamente no startup       |
| **Prometheus**              | Métricas expostas em `GET /metrics` via `prometheus-fastapi-instrumentator`               |
| **Grafana**                 | Dashboards em `http://localhost:3000` — datasource Prometheus                              |
| **Logging**                 | JSON estruturado com `correlation_id` e `model_version` em cada requisição                |

### Monitoramento em Produção

| Componente | Localização | Descrição |
|---|---|---|
| Logging JSON estruturado | `src/core/logging_config.py` | `correlation_id`, `model_version`, `latency_ms` |
| Métricas Prometheus | `src/monitoring/metrics.py` | Predições, probabilidades, falhas de validação, drift |
| Drift Detection (PSI/KS/JSD) | `src/monitoring/drift_detector.py` | Detecta mudança na distribuição de features |
| Performance Monitor | `src/monitoring/performance_monitor.py` | Avalia PR-AUC/F1 com ground truth em janela deslizante |
| Alert Check CLI | `src/monitoring/alert_check.py` | Verificação consolidada de alertas |
| Feedback Loop | `POST /api/v1/feedback/{customer_id}` | Registra ground truth de churn em `prediction_logs` |
| Baseline de Treino | `training_baseline.json` (artefato MLflow) | Distribuição de referência para drift detection |

**Comandos de operação:**
```bash
make drift          # gera relatório PSI/KS/JSD
make perf-monitor   # avalia performance com ground truth
make alert-check    # verifica alertas de drift e degradação
```

---

## K. Considerações de Segurança

### Input Validation

O schema `ChurnPredictionRequest` (evidência: `src/core/schemas.py`) usa **Pydantic com Literals estritos** para campos categóricos, prevenindo injeção de valores arbitrários. Exemplo:

```python
gender: Literal["Male", "Female"]
Contract: Literal["Month-to-month", "One year", "Two year"]
```

Campos fora do literal retornam HTTP 422 antes de atingir o modelo.

### TotalCharges Coerção Segura

A coerção de string vazia (`TotalCharges = " "`) para `0.0` está implementada com `try/except` explícito (`src/core/schemas.py → coerce_total_charges`), evitando ValueError silencioso.

### CORS

Configurado em `src/main.py` via `CORSMiddleware`. Os valores padrão são permissivos (`allow_origins=["*"]`), configuráveis por variável de ambiente `CORS_ORIGINS`. **Para produção, restringir `CORS_ORIGINS` ao domínio real.**

### Secrets e Credenciais

- `MLFLOW_TRACKING_URI`, `DATABASE_URL`, `MODEL_NAME`, `MODEL_STAGE` são lidos de variáveis de ambiente (`os.getenv`) em `src/core/config.py`. Sem hardcoding de senhas no código-fonte.
- Não foi encontrado arquivo `.env` ou gestor de secrets (Vault, AWS Secrets Manager) no repositório.

### Proteção do Model Registry

O `mlflow.db` (SQLite) é um arquivo local sem autenticação. Em produção, deve ser substituído por um servidor MLflow com AuthN/AuthZ ou um objeto store remoto (S3 + RDS).

### Logging

O `LoggingMiddleware` (`src/core/middlewares.py`) loga `method`, `path`, `status` e `latency` em cada requisição. **Nenhum dado do payload é logado**, prevenindo vazamento de dados do cliente nos logs.

---

## L. Checklist de Release

```markdown
## Pré-Treino
- [ ] Verificar integridade dos 3 CSVs em notebooks/data/raw/ (contagens, schema)
- [ ] Confirmar que mlflow.db está migrado: `make db-upgrade`
- [ ] Revisar CONFIG.best_params em src/core/config.py
- [ ] Garantir uv.lock atualizado: `uv sync`

## Treino
- [ ] Executar: `make train ARGS="--epochs 150"`
- [ ] Verificar que a run foi registrada: `make mlflowui`
- [ ] Confirmar métricas: test_pr_auc ≥ 0.65, test_roc_auc ≥ 0.84
- [ ] Confirmar artefato `training_baseline.json` registrado no MLflow run

## Aprovação de Modelo
- [ ] Comparar com run anterior no MLflow (métricas não regrediram)
- [ ] Revisar distribuição de churn_probability no conjunto de teste
- [ ] Avaliar desempenho por segmento: tenure, Contract, SeniorCitizen
- [ ] Aprovar com responsável técnico do time

## Publicação
- [ ] Atribuir alias 'production' à nova versão no MLflow Registry
- [ ] Atualizar MODEL_NAME e versão no docker-compose.yml se necessário
- [ ] Executar suíte de testes: `make test`

## Deploy
- [ ] Build da imagem: `docker compose build`
- [ ] Subir serviços: `docker compose up`
- [ ] Smoke test: `curl http://localhost:8000/health`
- [ ] Verificar que /health retorna model_loaded: true, model_version preenchido
- [ ] Teste de inferência: `curl -X POST .../predict -d @mock_request.json`
- [ ] Verificar que `GET /metrics` retorna métricas Prometheus
- [ ] Verificar logs estruturados JSON com correlation_id e model_version
- [ ] Confirmar que Prometheus scraping está ativo: `http://localhost:9090/targets`

## Pós-Deploy
- [ ] Monitorar latência nas primeiras 24h
- [ ] Confirmar que /health retorna model_loaded: true
- [ ] Documentar run_id da versão em produção
- [ ] Executar primeira rodada de drift detection: `make drift`
- [ ] Verificar alertas: `make alert-check`
```

---

## Evidências no Repositório

| Arquivo                                                   | Evidência                                                      |
| --------------------------------------------------------- | --------------------------------------------------------------- |
| `src/models/architectures.py`                           | Definição de `ChurnMLP` e `FocalLoss`                     |
| `src/models/train.py`                                   | Pipeline completo de treinamento, split, K-Fold, MLflow logging |
| `src/models/trainer.py`                                 | Loop AdamW + OneCycleLR + EarlyStopping                         |
| `src/core/config.py`                                    | Hiperparâmetros campeões, seed, listas de features            |
| `src/core/schemas.py`                                   | Contrato de I/O com Literals Pydantic + coerção TotalCharges  |
| `src/core/ml_service.py`                                | Carregamento de artefatos MLflow + pipeline de inferência      |
| `src/features/pipeline.py`                              | `clean_data()`, `get_preprocessor()`, `prepare_target()`  |
| `src/data/loader.py`                                    | `load_and_merge_data()` — merge dos 3 CSVs                   |
| `notebooks/data/raw/`                                   | Arquivos CSV de dados brutos                                    |
| `docs/reports/2026-04-25-relatorio-final-modelagem.md`  | Tabela de métricas finais de todos os modelos                  |
| `docs/tech_challenge_decisions.md`                      | Histórico de decisões de modelagem iterativa                  |
| `docs/specs/adrs/ADR-001-metrica-primaria-pr-auc.md`    | Justificativa da PR-AUC como métrica primária                 |
| `docs/specs/adrs/ADR-007-kfold-cross-validation.md`     | Justificativa e implementação do K-Fold                       |
| `docs/specs/adrs/ADR-008-decisao-modelo-campeao-api.md` | Decisão do modelo final para produção                        |
| `docs/specs/adrs/ADR-009-arquitetura-api.md`            | Decisão de arquitetura da API (Controller-Service)             |
| `mock_request.json`                                     | Payload de exemplo — cliente propenso ao churn                 |
| `mock_request_loyal.json`                               | Payload de exemplo — cliente leal (tenure=72, contrato 2 anos) |
| `pyproject.toml`                                        | Dependências e versão do Python                               |
| `Makefile`                                              | Comandos reproduzíveis de treino, teste e deploy               |
