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
Dados → Notebooks EDA → Refatoração em Scripts (src/) → MLflow Registry → API (Futuro)
```

### 📂 Estrutura do Projeto

```text
ML_TELCO_CHURN/
│
├── data/
│   ├── raw/                 # Arquivos CSV originais (customers, services, contracts)
│   └── processed/           # Dados intermediários de exploração
│
├── docs/                    # Artefatos arquiteturais
│   ├── plans/               # Planos de implementação executados
│   └── specs/               # Especificações técnicas e Design Docs
│
├── notebooks/               # Análise Exploratória e Baselines (Scikit-Learn/PyTorch)
│
├── src/ml_telco_churn/      # Código de Produção Refatorado
│   ├── config.py            # Variáveis globais do sistema e DataClasses (SOLID)
│   ├── data.py              # Ingestão e merge da base bruta
│   ├── features.py          # Processamento, Pipelines, Scalers e Encoders (Scikit)
│   ├── model_nn.py          # Arquitetura da Rede Neural (PyTorch - ChurnMLP)
│   └── train.py             # Script orquestrador que envia artefatos ao MLFlow
│
├── tests/                   # Suíte de testes Pytest (em desenvolvimento)
│
├── README.md
├── CLAUDE.md                # Diretrizes de desenvolvimento para IAs
├── pyproject.toml           # Dependências modernas via `uv` / Hatch
└── mlflow.db                # Banco local para o registro de experimentos do MLflow
```

---

## 🛠 Tecnologias Utilizadas

### Machine Learning
- **Scikit-Learn:** Criação de pipelines (ColumnTransformer, Imputer, StandardScaler, OneHotEncoder).
- **PyTorch:** Construção da Rede Neural MLP (Multi-Layer Perceptron).
- **Pandas e NumPy:** Manipulação e vetorização de dados.

### Engenharia de ML e Dev Tools
- **MLflow:** Rastreamento de métricas, parâmetros e Model Registry (exportação dos `.joblib` e `.pt` de forma unificada).
- **uv / pip:** Gerenciamento rápido de dependências Python.
- **Ruff e Pytest:** Linters e testes automatizados.

---

## ⚙️ Como Usar (Guia Rápido)

### 1️⃣ Configuração do Ambiente

O projeto usa **uv** e `pyproject.toml` para gestão de dependências. Mas pode ser usado com pip padrão.

#### Opção A: Usando `uv` (Recomendado)
```bash
# Instalar o uv (se não tiver): curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

#### Opção B: Usando `venv` e `pip`
```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate no Windows
pip install -e .
```

---

### 2️⃣ Subir o servidor do MLflow

O projeto salva **o modelo e os pipelines de transformação** diretamente no MLflow usando um banco SQLite local. Abra uma aba no terminal e rode:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Deixe esse terminal rodando. Acesse a UI no navegador via: `http://localhost:5000`

---

### 3️⃣ Executar o Treinamento Refatorado

Para treinar a rede neural PyTorch com a nova estrutura modular orientada a objetos (que lê os dados, faz os tratamentos com Scikit-learn, treina e injeta os artefatos no MLFlow):

Em uma nova aba de terminal, rode:
```bash
python src/ml_telco_churn/train.py --epochs 10 --customers notebooks/data/raw/churn_customers.csv --services notebooks/data/raw/churn_services.csv --contracts notebooks/data/raw/churn_contracts.csv
```

**O que vai acontecer?**
1. Os dados brutos serão lidos, tratados e formatados.
2. O PyTorch vai treinar por 10 épocas.
3. O MLflow salvará uma run contendo o modelo Scikit-Learn (`preprocessor`) e o modelo PyTorch (`pytorch_model`) vinculados.

---

## 🚀 Roadmap de Evolução

* ✅ **Etapa 1:** EDA, baselines, e protótipos de modelos PyTorch em Notebooks.
* ✅ **Etapa 2:** Refatoração de todo o treinamento num pipeline limpo, seguindo SOLID e focado em engenharia (Pasta `src/`). Integração unificada dos artefatos ao MLFlow.
* ⏳ **Etapa 3:** Desenvolvimento da suíte de `Testes Detalhados`.
* ⏳ **Etapa 4:** Criação da API de inferência usando **FastAPI** para servir as predições carregando os artefatos salvos pelo MLFlow.
* ⏳ **Etapa 5:** Documentação Final e Deploy da aplicação em nuvem/Docker.

---

## 📚 Contexto Acadêmico

Este projeto faz parte do **Tech Challenge da Pós-Graduação em Machine Learning Engineering da FIAP (Fase 1 - Grupo 21)**. O desafio mimetiza a evolução real de um produto de ML: iniciar com a exploração de dados simples e progredir até um pipeline sustentável de MLOps de código robusto.

---

## 👨‍💻 Autores

**Grupo 21 (FIAP)**
- **Eduardo Batista** (eduardoobatista2002@hotmail.com)
- **Braian Montoro**