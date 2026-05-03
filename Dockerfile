# Estágio de build com uv
FROM ghcr.io/astral-sh/uv:python3.13-slim AS builder

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Copiar arquivos de dependências
COPY pyproject.toml uv.lock ./

# Instalar dependências sem o projeto (cache layer)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Estágio final de execução
FROM python:3.13-slim

WORKDIR /app

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Copiar ambiente virtual do builder
COPY --from=builder /app/.venv /app/.venv

# Copiar o código fonte e dados necessários
COPY src /app/src
COPY notebooks/data/raw /app/notebooks/data/raw

# Embutir artefatos do MLflow (modelo + registry)
COPY mlruns /app/mlruns
COPY mlflow.db /app/mlflow.db

# Expor porta da API
EXPOSE 8000

# Comando para rodar a aplicação
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
