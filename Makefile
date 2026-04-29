.PHONY: help test run train mlflowui db-upgrade

# Variáveis de comando customizáveis
# Ex: make train ARGS="--epochs 10"
ARGS ?= --epochs 5
MLFLOW_DB_URI ?= sqlite:///mlflow.db
UV_RUN ?= uv run

help:
	@echo "======================================================================"
	@echo "                       ML_TELCO_CHURN API                             "
	@echo "======================================================================"
	@echo ""
	@echo "Comandos disponíveis:"
	@echo "  make help       - Mostra esta mensagem de ajuda"
	@echo "  make test       - Executa toda a suíte de testes com Pytest"
	@echo "  make run        - Inicia o servidor FastAPI em modo de desenvolvimento"
	@echo "  make train      - Executa o treinamento do modelo."
	@echo "                    Argumentos suportados via ARGS: --epochs, --customers, --services, --contracts"
	@echo "                    (Padrão: ARGS=\"--epochs 5\"). Ex: make train ARGS=\"--epochs 10\""
	@echo "  make mlflowui  - Inicia a interface do MLflow localmente"
	@echo ""


test:
	$(UV_RUN) pytest tests/ -v

db-upgrade:
	$(UV_RUN) mlflow db upgrade $(MLFLOW_DB_URI)

run:
	uv sync && $(MAKE) db-upgrade && $(UV_RUN) uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

train:
	@echo "Executando treinamento com argumentos: $(ARGS)"
	uv sync && $(MAKE) db-upgrade && MLFLOW_TRACKING_URI=$(MLFLOW_DB_URI) PYTHONPATH=. $(UV_RUN) python src/models/train.py $(ARGS)

mlflowui:
	uv sync && $(MAKE) db-upgrade && $(UV_RUN) mlflow ui --backend-store-uri $(MLFLOW_DB_URI) --port 5001