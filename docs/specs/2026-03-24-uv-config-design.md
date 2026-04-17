# Design Spec: Configuração UV e Isolamento de Dependências

**Data:** 2026-03-24  
**Status:** Aprovado  
**Contexto:** Tech Challenge Fase 01 - ML Telco Churn  

## 1. Objetivo
Configurar o `uv` como gerenciador de dependências central do projeto, garantindo conformidade com os requisitos do Tech Challenge (Single Source of Truth) e isolando as ferramentas de Exploração de Dados (EDA) do ambiente de Produção (API).

## 2. Decisões Técnicas

### 2.1 Versão do Python
- **Versão:** 3.13
- **Justificativa:** Utilizar a versão mais estável e moderna suportada pelo `uv` e pelas bibliotecas de ML (PyTorch/Scikit-Learn).

### 2.2 Gerenciamento de Dependências (PEP 735)
- **Abordagem:** Dependency Groups (Lean & Isolated).
- **Justificativa:** Evitar o "bloat" do ambiente de produção com bibliotecas de visualização (Matplotlib, Seaborn) e ferramentas de notebook (Jupyter), mantendo o container de API leve e seguro.

## 3. Estrutura do `pyproject.toml`

### 3.1 Dependências de Produção (App/API)
Essenciais para a execução do modelo e serviço:
- `fastapi`, `uvicorn`, `pydantic` (API Layer)
- `torch` (Inferência da MLP)
- `scikit-learn` (Pipeline de Pré-processamento e Métricas)
- `mlflow` (Tracking e Registry)
- `pandas`, `numpy` (Manipulação de dados)

### 3.2 Grupos de Dependências
- **`eda`**: Bibliotecas exclusivas para `notebooks/`:
  - `matplotlib`, `seaborn`, `scipy`, `ipykernel`, `jupyterlab`.
- **`dev`**: Ferramentas de qualidade e testes:
  - `ruff` (Linting/Formatting - Obrigatório no PDF)
  - `pytest` (Testes Automatizados - Obrigatório no PDF)
  - `pandera` (Validação de Schema - Obrigatório no PDF)

## 4. Workflow de Desenvolvimento

| Ação | Comando |
|---|---|
| Instalação Inicial | `uv sync` |
| Rodar Notebooks | `uv run --group eda jupyter lab` |
| Rodar Linting | `uv run ruff check .` |
| Rodar Testes | `uv run pytest` |
| Rodar API | `uv run uvicorn src.main:app` |

## 5. Arquitetura de Pastas
- `docs/specs/`: Especificações técnicas.
- `docs/plans/`: Planos de implementação.
- `pyproject.toml`: Configuração central.
