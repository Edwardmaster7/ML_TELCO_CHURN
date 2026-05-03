# Monitoring Plan — ML Telco Churn (Classificação Binária de Probabilidade)

> **Versão:** 2.0 | **Data:** 2026-05-02 | **Modelo:** MLP_Focal_KFold_Script | **Problema:** Classificação binária com saída de probabilidade contínua (churn_probability ∈ [0, 1])

---

## Convenções deste documento

| Símbolo           | Significado                                                |
| ------------------ | ---------------------------------------------------------- |
| ✅ IMPLEMENTADO    | Existe evidência concreta no repositório (path indicado) |
| 🟡 RECOMENDADO     | Não existe no repo; plano de implementação incluído    |
| 🔴 NÃO APLICÁVEL | Genuinamente fora de escopo para este sistema              |

> **Nota sobre terminologia:** O modelo é um **classificador binário** treinado com Focal Loss que produz uma probabilidade contínua de churn (`churn_probability ∈ [0, 1]`). As métricas de avaliação primárias são PR-AUC, ROC-AUC, F1, Precision e Recall — métricas de classificação, não de regressão (MAE/RMSE). Referências a "regressão" neste contexto aludem ao output de probabilidade contínua.

---

## 1. Contexto e Objetivos

### 1.1 O que é monitorado e por quê

| Dimensão                          | Descrição                                                                                 | Impacto se degradar                                                 |
| ---------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **Qualidade de dados**       | Features de entrada fora de distribuição, nulos inesperados, categorias desconhecidas     | Predições enviesadas; falsos negativos em clientes reais de churn |
| **Drift de distribuição**  | Mudança nas features de input e na distribuição de `churn_probability` em produção   | Modelo obsoleto sem gatilho de retreino                             |
| **Performance pós-label**   | Comparação entre `churn_probability` predita e `actual_churn` registrado via feedback | Degradação silenciosa não detectada                              |
| **Operação da API**        | Latência, taxa de erro, disponibilidade do modelo carregado                                | SLA quebrado; indisponibilidade de inferência                      |
| **Saúde da infraestrutura** | Modelo carregado em memória, Prometheus ativo, banco acessível                            | Downtime não alertado                                              |

### 1.2 Definições operacionais

| Termo                          | Definição neste sistema                                                                                                                                                                                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Produção**           | API FastAPI rodando em `http://localhost:8000` (ou `api:8000` no Docker) com modelo alias `production` no MLflow Registry                                                                                                                                             |
| **Baseline**             | Estatísticas das features e distribuição de `churn_probability` computadas sobre o conjunto de treino durante `make train`, salvas como artefato `training_baseline.json` no MLflow                                                                                |
| **Janela**               | Período retroativo de dados de produção usado para análise (padrão: 7 dias para drift, 30 dias para performance)                                                                                                                                                       |
| **Ground truth / label** | Coluna `actual_churn` (0 ou 1) na tabela `prediction_logs`, preenchida via `POST /api/v1/feedback/{customer_id}`                                                                                                                                                      |
| **Label delay**          | Intervalo entre a predição e o recebimento do `actual_churn`. Neste domínio: tipicamente 30–90 dias (churn confirmado na fatura seguinte ou após período de inatividade). O `perf_monitor` opera sobre amostras com label disponível, independentemente do delay |
| **Drift**                | Mudança estatisticamente significativa entre a distribuição de treino (baseline) e a distribuição de produção recente                                                                                                                                                |

---

## 2. Escopo do Sistema Monitorado

### 2.1 Mapa de componentes

| Componente                              | Função                                                             | Entrypoint principal                                                     | Evidência no repo                             |
| --------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------- |
| **Ingestão de dados**            | Merge de 3 CSVs brutos                                               | `src/data/loader.py` → `load_and_merge_data()`                      | `src/data/loader.py`                         |
| **Pipeline de features**          | Limpeza, feature engineering, ColumnTransformer                      | `src/features/pipeline.py` → `clean_data()`, `get_preprocessor()` | `src/features/pipeline.py`                   |
| **Feature engineering avançada** | Features derivadas (charges_per_tenure, total_services_count etc.)   | `src/features/build_features.py` → `engineer_advanced_features()`   | `src/features/build_features.py`             |
| **Treinamento**                   | MLP PyTorch com Focal Loss + StratifiedKFold(3) + MLflow tracking    | `src/models/train.py`                                                  | `src/models/train.py`                        |
| **Armazenamento de artefatos**    | MLflow SQLite backend + mlruns/                                      | `mlflow.db`, `mlruns/`                                               | `docker-compose.yml`, `src/core/config.py` |
| **Inferência online**            | FastAPI `POST /api/v1/predict`, resposta síncrona                 | `src/main.py`, `src/api/v1/api.py`                                   | `src/main.py`                                |
| **Logging de predições**        | Tabela SQLite `prediction_logs` via background task                | `src/core/database.py`, `src/api/v1/api.py`                          | `src/core/database.py`                       |
| **Feedback loop**                 | `POST /api/v1/feedback/{customer_id}?actual_churn=0                  | 1`                                                                       | `src/api/v1/api.py`                          |
| **Drift detector**                | PSI (numéricas), KS (numéricas), Chi2 (categóricas), JSD (output) | `src/monitoring/drift_detector.py`                                     | `src/monitoring/drift_detector.py`           |
| **Performance monitor**           | PR-AUC, ROC-AUC, F1, precision, recall por janela temporal           | `src/monitoring/performance_monitor.py`                                | `src/monitoring/performance_monitor.py`      |
| **Alert check**                   | Verificação de relatórios JSON + saúde da API                    | `src/monitoring/alert_check.py`                                        | `src/monitoring/alert_check.py`              |
| **Métricas Prometheus**          | Contadores/Gauges/Histogramas expostos em `/metrics`               | `src/monitoring/metrics.py`                                            | `src/monitoring/metrics.py`                  |
| **Prometheus**                    | Coleta de métricas a cada 15s                                       | `infra/prometheus.yml`                                                 | `infra/prometheus.yml`                       |
| **Grafana**                       | Visualização de dashboards                                         | `docker-compose.yml` (porta 3000)                                      | `docker-compose.yml`                         |

### 2.2 Tabela de sinais por componente

| Componente        | Sinais coletados                                                       | Onde coletar                                 | Responsável     |
| ----------------- | ---------------------------------------------------------------------- | -------------------------------------------- | ---------------- |
| API FastAPI       | latência p95, taxa de erro 5xx, taxa de 422, throughput               | `/metrics` (Prometheus) + logs JSON stdout | Infra/MLOps      |
| MLService         | modelo carregado (0/1), model_version, run_id                          | `/health`, gauge `churn_model_loaded`    | MLOps            |
| Predição        | churn_probability (histograma), churn_prediction (rate churn/no_churn) | `/metrics`, tabela `prediction_logs`     | MLOps            |
| Features de input | distribuição das 9 numéricas + 15 categóricas                      | `drift_detector` (json report)             | Data Science     |
| Performance       | PR-AUC, F1, recall (com label)                                         | `performance_monitor` (json report)        | Data Science     |
| Dados brutos      | schema, nulos, cardinalidade                                           | 🟡 RECOMENDADO — ver seção 4.1            | Data Engineering |

---

## 3. Telemetria Mínima (Logging, Métricas e Tracing)

### 3.1 Logging Estruturado

✅ IMPLEMENTADO

- `src/core/logging_config.py` — `JSONFormatter` + `setup_json_logging()`
- `src/core/middlewares.py` — `LoggingMiddleware` emite log por requisição
- `src/main.py` — `setup_json_logging(level="INFO")` chamado no boot

**Campos emitidos por requisição HTTP:**

```json
{
  "timestamp": "2026-05-02T20:00:00.000000+00:00",
  "level": "INFO",
  "module": "api_logger",
  "message": "http_request",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "POST",
  "path": "/api/v1/predict",
  "status": 200,
  "latency_ms": 42.5
}
```

**Campos emitidos no boot (lifespan):**

```json
{"timestamp": "...", "level": "INFO", "module": "src.main", "message": "Iniciando FastAPI: criando tabelas no banco de dados..."}
{"timestamp": "...", "level": "INFO", "module": "src.core.ml_service", "message": "Conectando ao MLflow em sqlite:///mlflow.db"}
```

**Política de PII e redação:**

Os logs **não incluem** o payload completo de request (features do cliente). O campo `input_summary` está disponível na `JSONFormatter` mas deve ser usado apenas com **agregações** (ex: `{"num_features": 20, "missing_count": 0}`), nunca com valores brutos de features demograficamente sensíveis como `gender`, `SeniorCitizen`, `Partner`, `Dependents`.

- `customerID` é propagado internamente via `PredictionLog.customer_id` no banco SQLite (não nos logs de stdout).
- Logs de stdout não devem conter customerID, MonthlyCharges, TotalCharges em claro.

🟡 RECOMENDADO — Persistência e retenção de logs

- **Como implementar:** Configurar driver de log do Docker para enviar stdout para Loki (Grafana stack) ou para um arquivo rotacionado com `logging.handlers.RotatingFileHandler`.
- **Arquivos sugeridos:** `infra/loki.yml`, ajuste em `docker-compose.yml`
- **DoD:** Logs disponíveis por ≥ 30 dias; pesquisáveis por `request_id` e `model_version`

### 3.2 Métricas Operacionais (SLOs/SLA)

✅ IMPLEMENTADO — Métricas Prometheus

- `src/monitoring/metrics.py` — 6 métricas customizadas
- `infra/prometheus.yml` — scrape a cada 15s
- `infra/alert_rules.yml` — 5 regras de alerta
- `docker-compose.yml` — serviços `prometheus` (9090) e `grafana` (3000)

**Métricas expostas em `/metrics`:**

| Nome da métrica                         | Tipo      | Labels                                      | Descrição                                          |
| ---------------------------------------- | --------- | ------------------------------------------- | ---------------------------------------------------- |
| `churn_predictions_total`              | Counter   | `prediction_class` (churn/no_churn/error) | Total de predições por classe                      |
| `churn_model_loaded`                   | Gauge     | —                                          | 1 = modelo em memória; 0 = degradado                |
| `churn_prediction_probability`         | Histogram | —                                          | Distribuição das probabilidades (buckets 0.0–1.0) |
| `churn_data_validation_failures_total` | Counter   | —                                          | Requisições rejeitadas com HTTP 422                |
| `churn_data_drift_psi`                 | Gauge     | `feature`                                 | Último PSI calculado por feature numérica          |
| `churn_data_drift_jsd`                 | Gauge     | —                                          | Último JSD da distribuição de churn_probability   |
| `http_request_duration_seconds`        | Histogram | `handler`, `method`, `status_code`    | Latência HTTP (via instrumentator)                  |
| `http_requests_total`                  | Counter   | `handler`, `method`, `status_code`    | Volume HTTP (via instrumentator)                     |

**SLOs sugeridos (valores iniciais — calibrar após 30 dias de prod):**

| SLO                        | Threshold    | Janela        | Alerta no repo?                 |
| -------------------------- | ------------ | ------------- | ------------------------------- |
| Disponibilidade            | ≥ 99.5%     | 30 dias       | ✅`ApiDown` (1 min)           |
| Latência P95 `/predict` | ≤ 2.0 s     | 5 min rolling | ✅`HighLatencyP95`            |
| Taxa de erro 5xx           | ≤ 5%        | 5 min rolling | ✅`HighErrorRate5xx`          |
| Modelo carregado           | = 1 (sempre) | 2 min         | ✅`ModelNotLoaded`            |
| Taxa de 422                | ≤ 0.1 req/s | 5 min rolling | ✅`HighValidationFailureRate` |

### 3.3 Tracing (Correlation ID)

✅ IMPLEMENTADO

- `src/core/middlewares.py` — `request_id_var: ContextVar[str]`; gera UUID v4 ou reutiliza `X-Request-ID` do cliente
- Header `X-Request-ID` devolvido em todas as respostas
- `PredictionLog.request_id` persiste o correlation_id junto à predição no banco

🟡 RECOMENDADO — Tracing distribuído (OpenTelemetry)

- **Como implementar:** Adicionar `opentelemetry-instrumentation-fastapi` + exportador para Jaeger ou OTLP
- **Arquivos sugeridos:** `src/core/tracing.py`, `infra/otel-collector.yml`
- **DoD:** Spans com `trace_id` visível no Grafana Tempo ou Jaeger; correlação com logs via `trace_id`

---

## 4. Monitoramento de Dados (Input)

### 4.1 Qualidade de Dados e Validação de Schema

✅ IMPLEMENTADO — Validação de schema no endpoint de inferência

- `src/core/schemas.py` — `ChurnPredictionRequest` (Pydantic v2) com `Literal` para todas as categóricas e `Field(ge=0)` para `tenure` e `MonthlyCharges`
- `TotalCharges` aceita `float | str` e passa por `@field_validator` que faz coerção para float (valores vazios → 0.0)
- HTTP 422 emitido automaticamente para qualquer violação de schema; contado em `churn_data_validation_failures_total`

**Cobertura atual da validação:**

| Feature             | Validação                            | Descrição                         |
| ------------------- | -------------------------------------- | ----------------------------------- |
| `gender`          | Literal["Male","Female"]               | Rejeita valores desconhecidos       |
| `SeniorCitizen`   | Literal[0,1]                           | Somente binário                    |
| `Contract`        | Literal[3 valores]                     | Rejeita contratos fora do catálogo |
| `tenure`          | `int`, `ge=0`                      | Rejeita negativos                   |
| `MonthlyCharges`  | `float`, `ge=0.0`                  | Rejeita negativos                   |
| `TotalCharges`    | `float\|str` → coerção             | Trata strings vazias como 0.0       |
| Demais categóricas | Literal com todas as opções válidas | Validação completa                |

🟡 RECOMENDADO — Validação de qualidade no pipeline de treino (Pandera/GE)

- **Como implementar:** Adicionar `pandera.DataFrameSchema` em `src/data/loader.py` para validar o DataFrame antes de `clean_data()`. Checar: taxa de nulos < 1% por coluna, `tenure` ∈ [0, 72], `MonthlyCharges` ∈ [18, 120], cardinalidade de categóricas.
- **Arquivos sugeridos:** `src/data/validation.py`, `configs/data_schema.yaml`
- **DoD:** `make train` falha com mensagem clara se o DataFrame violar o schema; relatório de qualidade salvo em `monitoring/reports/data_quality/`

🟡 RECOMENDADO — Missing rate monitoring em produção

- **Como implementar:** No `drift_detector`, para cada campo do payload, calcular `missing_rate = n_nulls / n_total` e comparar com thresholds (`missing_rate_warning: 0.005`, `missing_rate_critical: 0.01` já em `configs/monitoring.yaml`).
- **Arquivos sugeridos:** Adicionar função `check_missing_rates(production_df, thresholds)` em `src/monitoring/drift_detector.py`
- **DoD:** Alertas de missing rate aparecem no relatório JSON de drift; campo adicionado ao `configs/monitoring.yaml`

### 4.2 Drift e Detecção de Mudança de Distribuição

✅ IMPLEMENTADO

- `src/monitoring/drift_detector.py` — `compute_psi()`, `compute_ks()`, `compute_chi2()`, `compute_jsd_from_histograms()`
- `src/monitoring/drift_detector.py` — `load_baseline()` (tenta MLflow → fallback local)
- `src/monitoring/drift_detector.py` — `generate_drift_report()` (salva JSON em `monitoring/reports/drift/`)
- `src/models/train.py` — `_compute_baseline_stats()` (gera `training_baseline.json` como artefato MLflow)
- `configs/monitoring.yaml` — seção `drift:` com todos os thresholds
- `Makefile` — target `drift`
- `tests/monitoring/test_drift_detector.py` — TestComputePsi (4), TestComputeKs (3), TestComputeJsd (3)

**Métodos por tipo de feature:**

| Tipo                   | Features                                                                                                                                                       | Método                         | Threshold warning        | Threshold crítico       |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- | ------------------------ | ------------------------ |
| Numéricas             | tenure, monthlycharges, totalcharges, charges_per_tenure, total_services_count, is_monthly_contract, has_protection_services, is_high_spender, is_new_customer | PSI + KS-test                   | PSI ≥ 0.10 / KS ≥ 0.05 | PSI ≥ 0.20 / KS ≥ 0.10 |
| Categóricas           | gender, partner, contract, paymentmethod (e demais)                                                                                                            | Chi-Square (p-value)            | p < 0.10                 | p < 0.05                 |
| Output (probabilidade) | churn_probability                                                                                                                                              | Jensen-Shannon Divergence (JSD) | JSD ≥ 0.10              | JSD ≥ 0.15              |

**Baseline:** estatísticas de treino em `training_baseline.json` (artefato MLflow run de produção). Fallback: `configs/monitoring_baseline.json` (ainda não criado — ver recomendação abaixo).

**Interpretação dos thresholds:**

| PSI          | Interpretação                                                        |
| ------------ | ---------------------------------------------------------------------- |
| < 0.10       | Distribuição estável — nenhuma ação necessária                  |
| 0.10 – 0.20 | Mudança moderada — investigar segmentos; monitore por mais 2 semanas |
| > 0.20       | Drift severo — retreinar com dados recentes                           |

**Calibração:** Após 90 dias de produção, revisar thresholds com percentis observados de PSI/KS/JSD para reduzir falsos positivos.

**Formato do relatório JSON** (`monitoring/reports/drift/drift_report_YYYY-MM-DD.json`):

```json
{
  "generated_at": "2026-05-02T20:00:00+00:00",
  "n_production_samples": 500,
  "baseline_run_id": "abc123...",
  "numerical_drift": {
    "tenure": {"psi": 0.04, "ks_statistic": 0.03, "ks_pvalue": 0.80, "baseline_mean": 32.4, "current_mean": 33.1}
  },
  "categorical_drift": {
    "contract": {"chi2_statistic": 1.2, "p_value": 0.55, "dof": 2}
  },
  "prediction_drift": {
    "jsd": 0.05, "baseline_mean": 0.27, "current_mean": 0.28
  },
  "alerts": [],
  "summary": {"status": "ok", "critical_count": 0, "warning_count": 0}
}
```

🟡 RECOMENDADO — Arquivo de fallback do baseline

- **Como implementar:** Após `make train`, copiar o artefato baixado para `configs/monitoring_baseline.json` e comitar no repo (sem dados de clientes — apenas estatísticas agregadas). Isso garante que o drift detector funcione sem acesso ao MLflow.
- **Arquivos sugeridos:** `configs/monitoring_baseline.json`, ajuste no `Makefile` com target `baseline-export`
- **DoD:** `python -m src.monitoring.drift_detector` funciona sem MLflow disponível; CI verde mesmo sem servidor MLflow

### 4.3 Detecção de Anomalias de Dados

🟡 RECOMENDADO — Volume e anomalias súbitas

- **Como implementar:** Adicionar checks em `drift_detector.py` ou novo módulo `src/monitoring/data_anomaly.py`:
  - Volume diário de predições: se cair > 50% em relação à média dos últimos 7 dias → alerta
  - Features constantes (zero variance) por janela: PSI artificialmente zero pode mascarar colapso de feature
  - Spike de `SeniorCitizen = 1` acima de 60% (valor de treino: ~16%)
- **Arquivos sugeridos:** `src/monitoring/data_anomaly.py`, alertas adicionados a `infra/alert_rules.yml`
- **DoD:** Novo check de volume no relatório de drift; alerta Prometheus `LowPredictionVolume`

---

## 5. Monitoramento do Modelo (Output)

✅ IMPLEMENTADO — Distribuição de probabilidades via Prometheus

- `src/monitoring/metrics.py` — `PREDICTION_PROBABILITY` (Histogram, buckets 0.0–1.0)
- `src/monitoring/metrics.py` — `churn_data_drift_jsd` (Gauge)
- `src/api/v1/api.py` — `PREDICTION_PROBABILITY.observe(churn_prob)` em cada predição

**Checks de sanity já disponíveis via Pydantic:**

- `churn_probability` retornado é derivado de `torch.sigmoid(logit)` — garantido ∈ (0, 1)
- `churn_prediction` ∈ {0, 1} (limiar 0.5)

**O que monitorar no output:**

| Sinal                                          | Método                      | Alerta sugerido                    |
| ---------------------------------------------- | ---------------------------- | ---------------------------------- |
| Média de `churn_probability`                | JSD vs baseline              | JSD > 0.15 (já implementado)      |
| Churn rate predito (`churn_prediction == 1`) | Ratio em `prediction_logs` | > 40% ou < 10% por 3 dias seguidos |
| Predições fora de [0.05, 0.95] em excesso    | Histograma Prometheus        | Spike de predições extremas      |

🟡 RECOMENDADO — Churn rate diário no Grafana

- **Como implementar:** Criar painel Grafana consultando `churn_predictions_total{prediction_class="churn"}` / `churn_predictions_total` para exibir taxa diária. Adicionar linha de referência da taxa de treino (~26%).
- **Arquivos sugeridos:** `infra/grafana/dashboards/churn_overview.json`
- **DoD:** Dashboard provisionado automaticamente no `docker-compose.yml` (volume `grafana/provisioning/`)

🟡 RECOMENDADO — Monitoramento de explicabilidade (SHAP)

- **Como implementar:** Calcular SHAP values sobre amostra semanal de produção; comparar top-5 features mais impactantes com baseline de treino. Drift de importância relativa de features pode indicar mudança de padrão antes de degradação de PR-AUC.
- **Arquivos sugeridos:** `src/monitoring/explainability.py`
- **DoD:** Relatório JSON de SHAP feature importance em `monitoring/reports/shap/`; alerta se ordem das top-3 features mudar

---

## 6. Monitoramento de Performance (Quando o Ground Truth Chega)

### 6.1 Estratégia de Aquisição de Label e Label Delay

✅ IMPLEMENTADO — Feedback loop via API

- `src/api/v1/api.py` — `POST /api/v1/feedback/{customer_id}?actual_churn=0|1`
- `src/core/database.py` — `PredictionLog.actual_churn` + `PredictionLog.feedback_at`
- `src/monitoring/performance_monitor.py` — `load_labeled_predictions_from_db()` filtra por `actual_churn IS NOT NULL`

**Pareamento predição ↔ ground truth:**

O sistema usa `customer_id` como chave de pareamento. O endpoint de feedback atualiza o registro **mais recente** de `prediction_logs` para aquele `customer_id`. Para clientes com múltiplas predições, apenas a última é atualizada — limitação a corrigir via `prediction_id` explícito (ver recomendação abaixo).

**Label delay esperado:** 30–90 dias (churn confirmado após ciclo de faturamento). A janela padrão do `performance_monitor` é 30 dias (`window_days: 30` em `configs/monitoring.yaml`), mas pode ser ajustada:

```bash
# Avaliação com janela de 60 dias
python -m src.monitoring.performance_monitor --window-days 60
```

🟡 RECOMENDADO — Feedback via `prediction_id`

- **Como implementar:** Expor o `id` do `PredictionLog` na resposta do `/predict`. Atualizar o endpoint de feedback para aceitar `prediction_id` como parâmetro opcional, evitando ambiguidade em múltiplas predições do mesmo cliente.
- **Arquivos sugeridos:** `src/core/schemas.py` (adicionar `prediction_id` em `ChurnPredictionResponse`), `src/api/v1/api.py`
- **DoD:** `POST /api/v1/feedback/{customer_id}?prediction_id=123&actual_churn=1` funciona; testes cobrem o caso de múltiplas predições

### 6.2 Métricas de Classificação (Pós-Label)

✅ IMPLEMENTADO

- `src/monitoring/performance_monitor.py` — `compute_performance_metrics()`: PR-AUC, ROC-AUC, F1, Precision, Recall, churn_rate_actual, churn_rate_predicted, n_samples
- `src/monitoring/performance_monitor.py` — `evaluate_by_time_window()`: janelas deslizantes de N dias
- `src/monitoring/performance_monitor.py` — `generate_performance_report()`: salva JSON em `monitoring/reports/performance/`
- `configs/monitoring.yaml` — thresholds de performance
- `tests/monitoring/test_performance_monitor.py` — TestComputePerformanceMetrics (5), TestEvaluateByTimeWindow (4)
- `Makefile` — target `perf-monitor`

**Métricas calculadas:**

| Métrica                        | Baseline de treino | Threshold warning | Threshold crítico | Relevância                                                 |
| ------------------------------- | ------------------ | ----------------- | ------------------ | ----------------------------------------------------------- |
| **PR-AUC**                | ~0.693             | < 0.65            | < 0.60             | Métrica primária; robusta ao desbalanceamento (~74%/~26%) |
| **ROC-AUC**               | ~0.84              | —                | —                 | Referência secundária                                     |
| **F1-Score** (limiar 0.5) | —                 | < 0.55            | < 0.50             | Balanceia precision e recall                                |
| **Recall**                | —                 | < 0.60            | < 0.55             | Crítico: não perder clientes de churn                     |
| **Precision**             | —                 | < 0.55            | —                 | Evitar excesso de falsos positivos                          |
| **Churn rate predito**    | ~26%               | > 40% ou < 10%    | —                 | Sanity check de distribuição                              |

**Formato do relatório JSON** (`monitoring/reports/performance/perf_report_YYYY-MM-DD.json`):

```json
{
  "generated_at": "2026-05-02T20:00:00+00:00",
  "window_days": 30,
  "n_samples_with_label": 120,
  "overall": {
    "pr_auc": 0.67, "roc_auc": 0.83, "f1": 0.58, "precision": 0.62, "recall": 0.55,
    "churn_rate_actual": 0.25, "churn_rate_predicted": 0.27, "n_samples": 120
  },
  "by_time_window": [
    {"window_start": "2026-04-02T00:00:00", "window_end": "2026-05-02T00:00:00",
     "pr_auc": 0.67, "n_samples": 120}
  ],
  "alerts": [],
  "summary": {"status": "ok"}
}
```

🟡 RECOMENDADO — Performance por segmento

- **Como implementar:** Em `evaluate_by_time_window()`, adicionar parâmetro `segment_col` para calcular métricas por `Contract` (Month-to-month vs. One year vs. Two year), `SeniorCitizen` (0/1) e faixas de `tenure` (0–6, 7–24, 25–72). Segmentos com PR-AUC < 0.50 indicam viés sistemático.
- **Arquivos sugeridos:** Adicionar função `evaluate_by_segment()` em `src/monitoring/performance_monitor.py`
- **DoD:** Relatório de performance inclui seção `by_segment`; teste unitário cobre segmentação por Contract

### 6.3 Métricas de Negócio

🟡 RECOMENDADO — Lift e custo do erro

- **Contexto:** Falso negativo (churn não detectado) custa ~$10; falso positivo (ação de retenção desnecessária) custa ~$1 (definição do projeto).
- **Como implementar:** Adicionar função `compute_business_metrics(y_true, y_prob, cost_fn=10, cost_fp=1)` em `src/monitoring/performance_monitor.py`. Calcular: custo total evitado pela janela, lift vs. random (% a mais de churns detectados no top-K preditos).
- **Arquivos sugeridos:** `src/monitoring/performance_monitor.py`
- **DoD:** Relatório de performance inclui `business_metrics.total_cost_avoided` e `business_metrics.lift_at_k`

---

## 7. Alertas, Thresholds e Playbooks

### 7.1 Tabela de Alertas

✅ IMPLEMENTADO — Alertas Prometheus em `infra/alert_rules.yml`

| Alerta                        | Condição                       | For   | Severidade | Canal      | Ação imediata                                                         |
| ----------------------------- | -------------------------------- | ----- | ---------- | ---------- | ----------------------------------------------------------------------- |
| `ModelNotLoaded`            | `churn_model_loaded == 0`      | 2 min | critical   | Prometheus | Verificar logs do lifespan; checar MLflow Registry alias `production` |
| `HighErrorRate5xx`          | taxa 5xx > 5% em 5 min           | 5 min | critical   | Prometheus | Verificar `docker logs churn_api`; rollback se recente                |
| `HighLatencyP95`            | P95 `/predict` > 2.0s em 5 min | 5 min | warning    | Prometheus | Verificar carga; checar se GPU/CPU está saturado                       |
| `HighValidationFailureRate` | taxa 422 > 0.1 req/s em 5 min    | 5 min | warning    | Prometheus | Inspecionar payloads recentes; verificar mudança no cliente            |
| `ApiDown`                   | API não responde                | 1 min | critical   | Prometheus | Reiniciar container; escalar réplica                                   |

✅ IMPLEMENTADO — Alert check CLI

- `src/monitoring/alert_check.py` — verifica relatórios JSON de drift + performance + saúde da API
- Exit code: 0 = OK, 1 = CRÍTICO

```bash
# Verificação completa de alertas
python -m src.monitoring.alert_check --health-url http://localhost:8000/health
```

🟡 RECOMENDADO — Alertmanager configurado

- **Evidência:** `infra/prometheus.yml` tem `alertmanagers: static_configs: targets: []` (vazio)
- **Como implementar:** Subir Alertmanager no `docker-compose.yml`; configurar rota para Slack/email em `infra/alertmanager.yml`
- **Arquivos sugeridos:** `infra/alertmanager.yml`, ajuste em `docker-compose.yml` e `infra/prometheus.yml`
- **DoD:** Alertas críticos chegam no canal Slack configurado em < 5 min após disparo

🟡 RECOMENDADO — Alertas de drift no Prometheus

- **Como implementar:** Atualizar `churn_data_drift_psi{feature}` e `churn_data_drift_jsd` via `make drift` e adicionar regras em `infra/alert_rules.yml`:
  ```yaml
  - alert: HighDriftPSI
    expr: churn_data_drift_psi > 0.20
    labels:
      severity: critical
  - alert: HighPredictionDriftJSD
    expr: churn_data_drift_jsd > 0.15
    labels:
      severity: critical
  ```
- **DoD:** `make drift` atualiza Gauges Prometheus; alertas disparam no Grafana

### 7.2 Playbooks

**Playbook: Alerta `ModelNotLoaded`**

1. `docker logs churn_api` — verificar mensagem de erro no lifespan
2. Confirmar que MLflow está UP: `curl http://localhost:5000/health`
3. Verificar que alias `production` existe: `curl http://localhost:5000/api/2.0/mlflow/registered-models/alias?name=MLP_Focal_KFold_Script&alias=production`
4. Se alias não existir: `mlflow models set-alias --model-name MLP_Focal_KFold_Script --alias production --version <N>`
5. Reiniciar container: `docker compose restart api`

**Playbook: Drift Crítico Detectado**

1. `make drift` — obter relatório completo em `monitoring/reports/drift/`
2. Identificar features com PSI > 0.20 — correlacionar com eventos externos (campanhas, sazonalidade)
3. Se drift confirmado em > 3 features numéricas → abrir tarefa de retreino
4. Retreinar com dados dos últimos 90 dias: `make train ARGS="--epochs 100"`
5. Promover nova versão: `mlflow models set-alias ...`
6. Atualizar `configs/monitoring_baseline.json` com novo baseline

**Playbook: Degradação de PR-AUC**

1. `make perf-monitor` — verificar `by_time_window` para localizar quando degradou
2. Correlacionar com mudança de model_version em `prediction_logs`
3. Se degradação coincide com mudança de versão → rollback imediato
4. Verificar se label delay aumentou (ex: `feedback_at - predicted_at` crescente)
5. Se PR-AUC < 0.60 por > 2 semanas → retreinar obrigatório

---

## 8. Retraining e Gestão de Mudanças

### 8.1 Critérios para Retreino

| Gatilho                                                    | Threshold            | Tipo       | Ação                   |
| ---------------------------------------------------------- | -------------------- | ---------- | ------------------------ |
| PR-AUC < 0.60 por ≥ 14 dias                               | Métrica de negócio | Crítico   | Retreino obrigatório    |
| PR-AUC < 0.65 por ≥ 30 dias                               | Degradação gradual | Warning    | Retreino planejado       |
| PSI > 0.20 em ≥ 2 features numéricas                     | Drift severo         | Crítico   | Retreino obrigatório    |
| JSD > 0.15 em `churn_probability`                        | Drift de output      | Crítico   | Investigar + retreino    |
| Mudança de distribuição em `Contract` (chi2 p < 0.05) | Drift categórico    | Warning    | Retreino planejado       |
| 90 dias sem retreino (política de tempo)                  | Tempo                | Preventivo | Retreino de manutenção |

### 8.2 Aprovação e Governança

🟡 RECOMENDADO — Checklist de aprovação formal

- **Como implementar:** Criar ADR de processo em `docs/specs/adrs/ADR-NNN-retraining-governance.md` definindo: quem aprova (Data Scientist lead), quais métricas devem melhorar, período de shadow mode antes de promover alias.
- **Arquivos sugeridos:** `docs/specs/adrs/ADR-004-retraining-governance.md`
- **DoD:** PR de nova versão de modelo requer checklist preenchido; CI verifica métricas no MLflow antes de promover alias

### 8.3 Versionamento

✅ IMPLEMENTADO

- `model_version`: string com número da versão do MLflow Registry, persistida em `PredictionLog.model_version` e campo `model_version` nos logs JSON
- `run_id`: `MLService.run_id` — disponível em `/health`
- `loaded_at`: `MLService.loaded_at` — timestamp UTC do carregamento dos artefatos

🟡 RECOMENDADO — Data version e config version

- **Como implementar:** Em `make train`, calcular hash MD5 dos 3 CSVs de entrada e logar como tag MLflow `data_version`. Logar hash do `configs/monitoring.yaml` como `config_version`.
- **Arquivos sugeridos:** Ajuste em `src/models/train.py`
- **DoD:** `mlflow runs get <RUN_ID>` exibe `data_version` e `config_version` como tags

---

## 9. Segurança, Privacidade e Compliance

### 9.1 O que NUNCA logar

| Campo                                            | Motivo                      | Alternativa permitida                                        |
| ------------------------------------------------ | --------------------------- | ------------------------------------------------------------ |
| `customerID` em stdout/logs                    | Identificador direto (PII)  | Usar apenas em `prediction_logs` (banco restrito)          |
| `gender` em texto dos logs                     | Dado demográfico sensível | Apenas como agregação (ex: distribuição no drift report) |
| `SeniorCitizen` individual                     | Dado de saúde/idade        | Apenas taxa agregada (% de idosos no batch)                  |
| `MonthlyCharges`, `TotalCharges` individuais | Dados financeiros           | Apenas percentis/média em reports agregados                 |
| Tokens, senhas,`DATABASE_URL`                  | Credenciais                 | Usar variáveis de ambiente; nunca em logs                   |

### 9.2 Controle de Acesso

🟡 RECOMENDADO — Autenticação na API

- **Como implementar:** Adicionar `APIKeyMiddleware` ou OAuth2 Bearer Token em `src/core/middlewares.py`. Para ambiente interno: API Key via header `X-API-Key` validada contra variável de ambiente.
- **Arquivos sugeridos:** `src/core/auth.py`, ajuste em `src/main.py`
- **DoD:** `/predict` retorna 401 sem API Key válida; chave rotacionável sem restart

### 9.3 Retenção

| Dado                               | Local                       | Retenção sugerida                           | Implementado?                           |
| ---------------------------------- | --------------------------- | --------------------------------------------- | --------------------------------------- |
| Logs de stdout                     | Docker stdout               | 30 dias (via log rotation)                    | 🟡 RECOMENDADO (configurar no driver)   |
| `prediction_logs` (banco SQLite) | `mlflow.db`               | 365 dias (purge de registros antigos)         | 🟡 RECOMENDADO (adicionar job de purge) |
| Relatórios JSON de drift/perf     | `monitoring/reports/`     | 90 dias                                       | 🟡 RECOMENDADO (cron/Makefile)          |
| Métricas Prometheus (TSDB)        | Docker volume               | 7 dias (`--storage.tsdb.retention.time=7d`) | ✅`docker-compose.yml`                |
| Artefatos MLflow                   | `mlruns/` + `mlflow.db` | Indefinido (por versão de modelo)            | ✅ MLflow nativo                        |

---

## 10. Implementação no Repositório (Mapa de Arquivos)

### 10.1 O que já existe

```
✅ src/monitoring/
   ├── __init__.py
   ├── drift_detector.py       — PSI, KS, chi2, JSD, generate_drift_report()
   ├── metrics.py              — Prometheus metrics (Counter, Gauge, Histogram)
   ├── performance_monitor.py  — compute_performance_metrics(), evaluate_by_time_window()
   └── alert_check.py          — check_drift_report(), check_performance_report(), main()

✅ src/core/
   ├── logging_config.py       — JSONFormatter, setup_json_logging()
   ├── middlewares.py          — LoggingMiddleware, request_id_var (ContextVar)
   ├── database.py             — PredictionLog (SQLAlchemy), init_db()
   └── ml_service.py           — MLService singleton, model_version, run_id, loaded_at

✅ src/api/v1/api.py            — POST /predict (com log async), POST /feedback/{customer_id}
✅ src/main.py                   — /health enriquecido, setup_metrics(app), setup_json_logging()
✅ src/models/train.py           — _compute_baseline_stats(), log de training_baseline.json
✅ src/core/schemas.py           — Pydantic ChurnPredictionRequest com validação estrita

✅ configs/monitoring.yaml       — Todos os thresholds e configurações centralizadas

✅ infra/prometheus.yml          — Scrape job "telco-churn-api" a cada 15s
✅ infra/alert_rules.yml         — 5 alertas (ModelNotLoaded, HighErrorRate5xx, HighLatencyP95, HighValidationFailureRate, ApiDown)

✅ docker-compose.yml            — Serviços: mlflow, api, prometheus (9090), grafana (3000)
✅ Makefile                      — Targets: drift, perf-monitor, alert-check

✅ tests/monitoring/
   ├── __init__.py
   ├── test_drift_detector.py  — 10 testes (PSI, KS, JSD)
   ├── test_performance_monitor.py — 9 testes (métricas, janelas temporais)
   └── test_alert_check.py     — 11 testes (drift report, perf report, alert report)
```

### 10.2 O que deve ser criado

```
🟡 configs/monitoring_baseline.json     — Fallback local do baseline de treino
🟡 src/data/validation.py               — Schema Pandera para validação pré-treino
🟡 src/monitoring/data_anomaly.py       — Detecção de anomalias de volume e variance
🟡 src/monitoring/explainability.py     — SHAP values semanais
🟡 src/core/auth.py                     — API Key middleware
🟡 infra/alertmanager.yml               — Configuração Alertmanager (Slack/email)
🟡 infra/grafana/
   ├── provisioning/datasources/prometheus.yaml
   └── dashboards/churn_overview.json
🟡 .github/workflows/monitoring.yml     — CI job semanal: make drift + make alert-check
```

---

## 11. Execução Local (Windows e Linux)

### 11.1 Pré-requisitos

```bash
# 1. Instalar dependências (uv)
uv sync

# 2. Treinar modelo e gerar baseline (necessário antes de rodar drift)
# Linux/macOS:
PYTHONPATH=. MLFLOW_TRACKING_URI=sqlite:///mlflow.db uv run python src/models/train.py --epochs 100

# Windows PowerShell:
$env:PYTHONPATH="."; $env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"; uv run python src/models/train.py --epochs 100

# 3. Subir a API (para gerar prediction_logs)
make run
# ou: uv run uvicorn src.main:app --host 0.0.0.0 --port 8000

# 4. Fazer algumas predições para popular prediction_logs
# Usar os arquivos mock incluídos no repo:
curl -X POST http://localhost:8000/api/v1/predict -H "Content-Type: application/json" -d @mock_request.json
```

### 11.2 Detecção de Drift

```bash
# ── Via Makefile (recomendado) ──────────────────────────────────────
make drift
make drift DRIFT_ARGS="--window-days 14"

# ── Python direto (cross-platform) ─────────────────────────────────
# Usando prediction_logs do banco (padrão):
python -m src.monitoring.drift_detector --window-days 7

# Usando CSV de produção externo:
python -m src.monitoring.drift_detector --production-csv monitoring/requests.csv --window-days 7

# Com output customizado:
python -m src.monitoring.drift_detector --window-days 7 --output-dir monitoring/reports/drift

# Ajuda:
python -m src.monitoring.drift_detector --help
```

> **Pré-requisito externo:** O drift detector precisa do `training_baseline.json` no MLflow Run de produção, ou do arquivo `configs/monitoring_baseline.json` como fallback. Se nenhum dos dois existir, o detector emite erro e sai com código 1.

### 11.3 Monitoramento de Performance Pós-Label

```bash
# ── Via Makefile ────────────────────────────────────────────────────
make perf-monitor
make perf-monitor PERF_ARGS="--window-days 60"

# ── Python direto ──────────────────────────────────────────────────
# Usando prediction_logs com actual_churn preenchido:
python -m src.monitoring.performance_monitor --window-days 30

# Usando CSVs externos (para teste sem API rodando):
python -m src.monitoring.performance_monitor \
    --predictions-csv monitoring/preds.csv \
    --labels-csv monitoring/labels.csv \
    --window-days 30

# Ajuda:
python -m src.monitoring.performance_monitor --help
```

> **Pré-requisito externo:** Predições com `actual_churn` preenchido via `POST /api/v1/feedback/`. Em ambiente de desenvolvimento sem feedback real, use CSVs sintéticos com colunas `churn_probability`, `churn_prediction`, `predicted_at`, `actual_churn`.

### 11.4 Verificação de Alertas

```bash
# ── Via Makefile ────────────────────────────────────────────────────
make alert-check
make alert-check ALERT_ARGS="--health-url http://localhost:8000/health --fail-on-warning"

# ── Python direto ──────────────────────────────────────────────────
# Verificar relatórios mais recentes + saúde da API:
python -m src.monitoring.alert_check --health-url http://localhost:8000/health

# Apontar para relatórios específicos:
python -m src.monitoring.alert_check \
    --drift-report monitoring/reports/drift/drift_report_2026-05-02.json \
    --perf-report monitoring/reports/performance/perf_report_2026-05-02.json

# Exit codes: 0 = OK | 1 = CRÍTICO | 2 = ERRO
echo "Exit code: $?"  # Linux
echo "Exit code: $LASTEXITCODE"  # PowerShell
```

### 11.5 Stack Completa de Observabilidade (Docker)

```bash
docker compose up --build

# Serviços:
# API:        http://localhost:8000
# MLflow UI:  http://localhost:5000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin/admin)

# Endpoints de monitoramento:
curl http://localhost:8000/health
curl http://localhost:8000/metrics
curl -X POST "http://localhost:8000/api/v1/feedback/7590-VHVEG?actual_churn=1"
```

---

## 12. Checklist Operacional

### 12.1 Checklist Semanal

```markdown
## Checklist de Monitoramento — Semana de YYYY-MM-DD

### Drift de Dados
- [ ] `make drift` executado sem erros
- [ ] Nenhum alerta CRITICAL em `monitoring/reports/drift/drift_report_<data>.json`
- [ ] PSI < 0.20 em todas as features numéricas
- [ ] JSD < 0.15 em `churn_probability`
- [ ] Chi2 p-value > 0.05 em todas as categóricas

### Performance do Modelo (se houver labels)
- [ ] `make perf-monitor` executado sem erros
- [ ] PR-AUC ≥ 0.65 (warning) / ≥ 0.60 (crítico)
- [ ] Recall ≥ 0.60
- [ ] Nenhuma tendência de queda em `by_time_window` por > 2 semanas

### Operação da API
- [ ] `make alert-check --health-url http://localhost:8000/health` → exit code 0
- [ ] `/health` retorna `model_loaded: true`
- [ ] P95 latência < 2.0s (verificar Prometheus/Grafana)
- [ ] Taxa de erro 5xx < 5% (verificar Grafana)

### Infraestrutura
- [ ] Prometheus coletando métricas: `http://localhost:9090/targets` (Estado: UP)
- [ ] Volume de `mlflow.db` estável (sem crescimento anômalo)
```

### 12.2 Checklist Mensal

```markdown
## Checklist Mensal de Monitoramento — Mês YYYY-MM

- [ ] Revisão de thresholds em `configs/monitoring.yaml` (calibrar com dados do mês)
- [ ] Verificar label delay médio: `feedback_at - predicted_at` na tabela prediction_logs
- [ ] Analisar tendência de PR-AUC nos últimos 30 dias
- [ ] Verificar distribuição de churn_rate predito vs. real
- [ ] Revisar alertas disparados no mês — ajustar thresholds se muitos falsos positivos
- [ ] Checar se `training_baseline.json` está desatualizado (> 90 dias)
- [ ] Verificar crescimento de `prediction_logs` e aplicar purge se necessário
- [ ] Atualizar `docs/tech_challenge_decisions.md` com insights do mês
```

### 12.3 O que Revisar em Incidentes

| Cenário                          | Primeiro check                                | Segundo check                             | Escalação                                           |
| --------------------------------- | --------------------------------------------- | ----------------------------------------- | ----------------------------------------------------- |
| API retornando 503                | `docker logs churn_api` → erro no lifespan | MLflow acessível?`curl localhost:5000` | Reiniciar container; verificar alias `production`   |
| PR-AUC < 0.60                     | `make perf-monitor` → `by_time_window`   | Drift correlacionado?`make drift`       | Retreino urgente                                      |
| Alta taxa de 422                  | Inspecionar payloads recentes (logs JSON)     | Mudança no cliente consumidor?           | Comunicar time de produto                             |
| Spike de churn rate predito > 40% | `make drift` → PSI/JSD alto?               | Verificar se dados reais de churn mudaram | Investigar sazonalidade                               |
| Latência P95 > 5s                | Verificar uso de CPU/memória do container    | Modelo em CPU em vez de GPU?              | Escalar instância; verificar device no `MLService` |

### 12.4 Definition of Done — Alterações no Monitoramento

Uma alteração no sistema de monitoramento é considerada pronta quando:

- [ ] Código com testes unitários (`pytest tests/monitoring/` verde)
- [ ] Thresholds novos ou alterados documentados em `configs/monitoring.yaml` com comentários
- [ ] Novos alertas adicionados a `infra/alert_rules.yml` com `annotations.description` explicativa
- [ ] README e/ou DEPLOYMENT_ARCHITECTURE.md atualizados se novos endpoints ou serviços
- [ ] ADR criado em `docs/specs/adrs/` se for decisão arquitetural (ex: troca de método de drift)
- [ ] `make test` verde no CI
- [ ] Checklist de release (`docs/MODEL_CARD.md`, seção L) atualizado se necessário
