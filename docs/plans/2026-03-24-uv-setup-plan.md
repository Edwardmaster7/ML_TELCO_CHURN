# UV Setup & Dependency Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configurar o `uv` com Python 3.13 e isolar as dependências de EDA e Desenvolvimento em grupos separados (PEP 735) no `pyproject.toml`.

**Architecture:** Utilizaremos o `uv` como gerenciador de pacotes único. As dependências de produção (API/ML) serão as principais, enquanto visualização (EDA) e ferramentas de qualidade (DEV) ficarão em grupos de dependências nomeados, garantindo um ambiente de execução enxuto.

**Tech Stack:** Python 3.13, UV, FastAPI, PyTorch, Scikit-Learn, MLflow, Ruff, Pytest.

---

### Task 1: Inicializar pyproject.toml com UV

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Inicializar o projeto com uv**

Run: `uv init --python 3.13`
Expected: `pyproject.toml` criado com metadados básicos e versão do python 3.13.

- [ ] **Step 2: Adicionar dependências de Produção (App)**

Run: `uv add fastapi uvicorn pydantic torch scikit-learn mlflow pandas numpy`
Expected: Seção `[project.dependencies]` preenchida no `pyproject.toml`.

- [ ] **Step 3: Criar Grupo de Dependências: eda**

Run: `uv add --group eda matplotlib seaborn scipy ipykernel jupyterlab`
Expected: Seção `[dependency-groups.eda]` criada no `pyproject.toml`.

- [ ] **Step 4: Criar Grupo de Dependências: dev**

Run: `uv add --group dev ruff pytest pandera`
Expected: Seção `[dependency-groups.dev]` criada no `pyproject.toml`.

- [ ] **Step 5: Sincronizar o ambiente**

Run: `uv sync --all-groups`
Expected: Arquivo `uv.lock` gerado e ambiente `.venv` criado com todas as libs.

---

### Task 2: Configurar Ferramentas (Ruff)

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Adicionar configuração do Ruff no pyproject.toml**

```toml
[tool.ruff]
line-length = 88
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "I", "B"]
ignore = []

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

- [ ] **Step 2: Validar configuração com ruff**

Run: `uv run ruff check .`
Expected: Execução sem erros de configuração (pode encontrar erros de linting nos arquivos existentes).

---

### Task 3: Validação Final do Ambiente

- [ ] **Step 1: Testar import do App (Produção)**

Run: `uv run python -c "import fastapi, torch, sklearn, mlflow; print('App libs OK')"`
Expected: Saída "App libs OK".

- [ ] **Step 2: Testar import do Grupo EDA**

Run: `uv run --group eda python -c "import matplotlib, seaborn, scipy; print('EDA libs OK')"`
Expected: Saída "EDA libs OK".

- [ ] **Step 3: Verificar isolamento (EDA não deve estar no base)**

Run: `uv run python -c "import seaborn" || echo "Isolamento OK"`
Expected: Erro de import ou mensagem "Isolamento OK".
