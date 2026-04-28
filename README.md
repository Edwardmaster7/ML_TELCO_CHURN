# 📱 Previsão de Churn em Empresa Telco com Machine Learning

Projeto de **Machine Learning End-to-End** para previsão de churn em empresa de telecom utilizando dados da base IBM Telco Customer Churn.

O objetivo é construir um **pipeline profissional de ML**, desde a exploração dos dados até a disponibilização do modelo via **API**, seguindo boas práticas de engenharia de Machine Learning, princípios SOLID e integração com MLOps (MLflow).

---

## 🎯 Problema de Negócio

Empresas de telecomunicação enfrentam o desafio constante de retenção de clientes. O "Churn" (taxa de cancelamento) tem impacto direto na receita e no Customer Lifetime Value (CLV).

Reter um cliente existente geralmente é muito mais barato do que adquirir um novo. Este projeto visa antecipar quais clientes têm alta probabilidade de cancelamento de contrato, permitindo que a equipe de marketing ou CS aja de forma proativa (ex: ofertas direcionadas).

---

## 🧠 Objetivo do Modelo

Prever a variável **CHURN**, classificando os clientes propensos a cancelar o serviço.

- **Classe 0 (No):** Cliente continua ativo
- **Classe 1 (Yes):** Cliente propenso a dar Churn

A otimização foca num trade-off de negócio usando **Cost-Sensitive Learning**: onde um falso negativo (perder um cliente) custa muito mais caro ($ 10) do que um falso positivo (oferecer desconto para um cliente que já ia ficar - $ 1).

---

## 🏗 Arquitetura do Projeto

A evolução do projeto seguiu do formato Notebook procedural para um **código modular isolado** e pronto para produção, focando em MLOps:

```text
Dados → Notebooks EDA → Refatoração em Scripts (src/) → MLflow Registry → API (FastAPI)
```

### 📂 Estrutura do Projeto

```text
ML_TELCO_CHURN/
│
├── data/
├── docs/                    # Artefatos arquiteturais, Plantas e Relatórios
│   ├── ModelCard.md                 # Entregável Etapa 4: Model Card
│   └── Deploy_Monitoramento.md      # Entregável Etapa 4: Deploy e Monitoramento
│
├── notebooks/               # Análise Exploratória e Baselines (Scikit-Learn/PyTorch)
│
├── src/                     # Código de Produção Refatorado
│   ├── api/                 # Endpoints FastAPI e Roteamento
│   ├── core/                # Schemas Pydantic, ML Service, Middlewares e Configs
│   ├── data/                # Ingestão e loaders
│   ├── features/            # Processamento, Pipelines e Scalers Scikit-Learn
│   ├── models/              # Treinamento, Teste e Arquitetura ChurnMLP (PyTorch)
│   └── main.py              # Entrypoint da API REST
│
├── tests/                   # Suíte de testes Pytest (Smoke, Validação, etc.)
│
├── pyproject.toml           # Single Source of Truth das Dependências (uv / Hatch)
└── README.md
```

---

## 🛠 Tecnologias Utilizadas

### Machine Learning
- **Scikit-Learn:** Criação de pipelines (ColumnTransformer, Imputer, StandardScaler, OneHotEncoder).
- **PyTorch:** Construção da Rede Neural MLP (Multi-Layer Perceptron), Early Stopping, Focal Loss.

### Engenharia de ML e Dev Tools
- **MLflow:** Rastreamento de métricas, parâmetros e Model Registry.
- **FastAPI / Pydantic:** Disponibilização da Inference API com validação forte em runtime.
- **uv / pip:** Gerenciamento rápido de dependências Python.
- **Ruff e Pytest:** Linters e testes automatizados.

---

## ⚙️ Como Usar (Guia Rápido)

### 1️⃣ Configuração do Ambiente

O projeto usa **uv** e `pyproject.toml` para gestão unificada.

```bash
# Instalar o uv (se não tiver): curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### 2️⃣ Subir o MLflow (Registry Local)

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Acesse a UI no navegador via: `http://localhost:5000`

### 3️⃣ Subir a API de Predição do Modelo (FastAPI)

Foi providenciado um comando Makefile para carregar a API com os modelos registrados (Certifique-se de que o MLFlow tenha gerado a run do modelo `MLP_Focal_KFold_Script`):

```bash
make run
# ou: uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```
- Validação no Pydantic via Endpoints Locais:
  - Documentação Swagger da API: `http://localhost:8000/docs`
  - URL p/ POST de cliente: `http://localhost:8000/api/v1/predict` (Pode utilizar os mock files da raiz do repositório)

### 4️⃣ Suíte de Testes (Pytest)

```bash
uv run pytest tests/
```

---

## 📄 Documentação Obrigatória (Etapa 4)

Conforme os requisitos da fase final de documentação, arquivos complementares consolidados que documentam performance, cenários falhos e planos de retenção no ambiente produtivo para o Tech Challenge estão anexados:

* [Model Card Oficial do Projeto](./docs/ModelCard.md)
* [Plano de Deploy da Arquitetura e Estratégia de Monitoramento](./docs/Deploy_Monitoramento.md)

---

## 🚀 Roadmap (Entregas do Tech Challenge)

* ✅ **Etapa 1:** EDA (EDA + ML Canvas + Baselines), formulados nos Notebooks e registrados no MLFlow.
* ✅ **Etapa 2:** MLP PyTorch + comparação de modelos + análise de custo logada no tracking do MLFlow.
* ✅ **Etapa 3:** Refatoração de todo o treinamento num pipeline limpo, código modular na Pasta `src/`. API FastAPI robusta + `pytest` unitários de inferência/schemas finalizados.
* ✅ **Etapa 4:** Model Card construído, Arquitetura e Monitoramento desenhados, README.md repaginado. Gravação do Video (STAR) executada.

---

## 👨‍💻 Autores

**Grupo 21 (FIAP)**
- **Eduardo Batista** (eduardoobatista2002@hotmail.com)
- **Braian Montoro** (brnlmontoro@gmail.com)
