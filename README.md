# 📱 Previsão de churn em empresa telco com Machine Learning

Projeto de **Machine Learning End-to-End** para previsão de churn em empresa de telecom utilizando dados estruturados do mercado automotivo.

O objetivo é construir um **pipeline profissional de ML**, desde a exploração dos dados até a disponibilização do modelo via **API**, seguindo boas práticas de engenharia de Machine Learning.

---

# 🎯 Problema de Negócio

Em Construção

---

# 🧠 Objetivo do Modelo

Prever a variável:

**CHURN**

Onde:

Em Construção

A variável é derivada da coluna original do dataset:

```
churn
```

---

# 🏗 Arquitetura do Projeto

Este projeto segue uma arquitetura de **Machine Learning modular**, separando responsabilidades em diferentes camadas.

Fluxo do pipeline:

```
Dados → EDA → Feature Engineering → Baselines → Rede Neural (MLP) → API → Deploy
```

---

# 📂 Estrutura do Projeto

```
ev-adoption-ml/
│
├── data
│   ├── raw              # Dados originais
│   └── processed        # Dados tratados
│
├── notebooks            # Análises exploratórias e experimentos
│
├── src
│   ├── api              # API de inferência (FastAPI)
│   ├── data             # Carregamento e preprocessamento
│   ├── features         # Engenharia de atributos
│   ├── models           # Treinamento e inferência
│   └── pipelines        # Pipelines de treinamento
│
├── tests                # Testes automatizados
│
├── docs                 # Documentação do projeto
│
├── models               # Artefatos de modelos treinados
├── mlruns               # Experimentos MLflow
│
├── README.md
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

---

# 🛠 Tecnologias Utilizadas

Principais ferramentas utilizadas no projeto:

### Machine Learning

* Scikit-Learn
* PyTorch

### Engenharia de ML

* MLflow
* Pytest

### API

* FastAPI
* Uvicorn
* Flask

### Análise de Dados

* Pandas
* NumPy
* Matplotlib
* Seaborn

---

# ⚙️ Configuração do Ambiente

### 1️⃣ Criar ambiente virtual

```
python -m venv .venv
```

### 2️⃣ Ativar ambiente

Windows:

```
.venv\Scripts\activate
```

Linux / Mac:

```
source .venv/bin/activate
```

---

### 3️⃣ Instalar dependências

```
pip install -r requirements.txt
```

---

# 📊 Executando a Análise Exploratória

Para iniciar a exploração dos dados:

```
jupyter notebook
```

Abrir o notebook:

```
notebooks/01_eda.ipynb
```

---

# 🧪 Experimentos com MLflow

Os experimentos de treinamento são registrados utilizando **MLflow**.

Para iniciar o servidor local:

```
mlflow ui
```

Depois acessar:

```
http://localhost:5000
```

---

# 🚀 Roadmap do Projeto

Etapas planejadas para evolução do projeto:

### Etapa 1

* Estrutura do projeto
* EDA
* Criação do target
* Baselines com Scikit-Learn
* Tracking com MLflow

### Etapa 2

* Construção de rede neural MLP com PyTorch
* Comparação com modelos tradicionais

### Etapa 3

* Refatoração do código em módulos
* Criação de pipeline reprodutível
* Implementação da API de inferência

### Etapa 4

* Model Card
* Documentação
* Deploy em nuvem

---

# 📚 Contexto Acadêmico

Este projeto faz parte do **Tech Challenge da Pós-Graduação em Machine Learning Engineering da FIAP**, cujo objetivo é construir um pipeline completo de Machine Learning aplicando boas práticas de engenharia de software e ciência de dados.

---

# 👨‍💻 Autor

**Braian Montoro**

Projeto desenvolvido como parte dos estudos em **Machine Learning Engineering**.
