# Plano de Implementação: Integração de Banco de Dados e Dockerização

> **Para agentes:** REQUISITO: Use superpowers:subagent-driven-development (recomendado) ou superpowers:executing-plans para implementar este plano tarefa por tarefa. Os passos usam a sintaxe de checkbox (`- [ ]`) para acompanhamento.

**Objetivo:** Implementar um sistema centralizado de gestão de banco de dados seguindo as melhores práticas do FastAPI (baseado no gsd-new-api) e dockerizar todo o ecossistema (API + Servidor de Tracking do MLflow) utilizando `uv` e `docker-compose`.

**Arquitetura:** 
- **Camada de Banco de Dados:** Arquivos centralizados `database.py` e `deps.py` para sessões assíncronas do SQLAlchemy.
- **Camada Docker:** Configuração de dois containers. Um para o MLflow (tracking/artefatos) e um para a API FastAPI, comunicando-se via rede interna.
- **Fluxo de Trabalho:** Automatizado via um `Makefile` padronizado.

**Tech Stack:** FastAPI, SQLAlchemy (Async), MLflow, Docker, Docker Compose, UV.

---

### Tarefa 1: Infraestrutura de Banco de Dados

**Arquivos:**
- Criar: `src/core/database.py`
- Criar: `src/core/deps.py`
- Modificar: `src/core/config.py`

- [ ] **Passo 1: Atualizar Configuração**
Adicionar URLs de banco de dados ao `ProjectConfig`.

```python
# src/core/config.py
    database_url: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///mlflow.db")
```

- [ ] **Passo 2: Criar database.py**
Implementar o engine e a factory de sessão.

```python
# src/core/database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from src.core.config import CONFIG

engine = create_async_engine(CONFIG.database_url, echo=False)
SessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)
```

- [ ] **Passo 3: Criar deps.py**
Implementar a dependência `get_db`.

```python
# src/core/deps.py
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import SessionLocal

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session
```

- [ ] **Passo 4: Commit**
```bash
git add src/core/database.py src/core/deps.py src/core/config.py
git commit -m "feat: adicionar infraestrutura de banco de dados e dependencias"
```

---

### Tarefa 2: Dockerização

**Arquivos:**
- Criar: `Dockerfile`
- Criar: `docker-compose.yml`
- Criar: `.dockerignore`

- [ ] **Passo 1: Criar Dockerfile**
Usar build multi-estágio com `uv`.

```dockerfile
FROM ghcr.io/astral-sh/uv:latest AS uv_bin
FROM python:3.13-slim-bullseye
COPY --from=uv_bin /uv /uvx /bin/
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev
COPY . .
RUN uv sync --frozen --no-dev
EXPOSE 8000
CMD ["uv", "run", "fastapi", "run", "src/main.py", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Passo 2: Criar docker-compose.yml**
Definir os serviços `api` e `mlflow`.

```yaml
services:
  mlflow:
    image: ghcr.io/astral-sh/uv:latest
    container_name: mlflow_server
    ports:
      - "5000:5000"
    volumes:
      - ./mlflow.db:/app/mlflow.db
      - ./mlruns:/app/mlruns
    working_dir: /app
    command: uv run mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

  api:
    build: .
    container_name: churn_api
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - DATABASE_URL=sqlite+aiosqlite:///mlflow.db
    depends_on:
      - mlflow
```

- [ ] **Passo 3: Commit**
```bash
git add Dockerfile docker-compose.yml .dockerignore
git commit -m "feat: dockerizacao da api e servidor mlflow"
```

---

### Tarefa 3: Makefile e Integração Final

**Arquivos:**
- Modificar: `Makefile`
- Modificar: `src/main.py`

- [ ] **Passo 1: Atualizar Makefile**
Adicionar comandos para docker, treinamento e execução local.

```makefile
# Makefile
help:
	@echo "Comandos disponíveis:"
	@echo "  train         - Treina o modelo"
	@echo "  run           - Roda a API localmente"
	@echo "  docker-up     - Sobe os containers"
	@echo "  docker-down   - Derruba os containers"

train:
	uv run src/models/train.py

run:
	uv run fastapi dev src/main.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
```

- [ ] **Passo 2: Ajustar MLflow URI no main.py**
Garantir que a API usa a URI do ambiente se disponível.

```python
# src/main.py (no lifespan ou configuração)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", settings.mlflow_tracking_uri)
```

- [ ] **Passo 3: Verificar e Commit**
```bash
git add Makefile src/main.py
git commit -m "chore: atualizacao do makefile e integracao de variaveis de ambiente"
```
