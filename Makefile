.PHONY: help test run train mlflowui db-upgrade drift perf-monitor alert-check

# Variáveis de comando customizáveis
# Ex: make train ARGS="--epochs 10"
ARGS ?= --epochs 5
MLFLOW_DB_URI ?= sqlite:///mlflow.db
UV_RUN ?= uv run
DRIFT_ARGS ?=
PERF_ARGS ?=
ALERT_ARGS ?=

help:
	@echo "======================================================================"
	@echo "                       ML_TELCO_CHURN API                             "
	@echo "======================================================================"
	@echo ""
	@echo "Comandos disponíveis:"
	@echo "  make help          - Mostra esta mensagem de ajuda"
	@echo "  make test          - Executa toda a suíte de testes com Pytest"
	@echo "  make run           - Inicia o servidor FastAPI em modo de desenvolvimento"
	@echo "  make train         - Executa o treinamento do modelo."
	@echo "                       Argumentos via ARGS: --epochs, --customers, --services, --contracts"
	@echo "                       (Padrão: ARGS=\"--epochs 5\"). Ex: make train ARGS=\"--epochs 10\""
	@echo "  make mlflowui      - Inicia a interface do MLflow localmente"
	@echo "  make drift         - Executa detecção de data drift (PSI/KS/JSD)"
	@echo "                       Ex: make drift DRIFT_ARGS=\"--window-days 14\""
	@echo "  make perf-monitor  - Avalia performance do modelo com ground truth"
	@echo "                       Ex: make perf-monitor PERF_ARGS=\"--window-days 30\""
	@echo "  make alert-check   - Verifica alertas a partir dos relatórios JSON mais recentes"
	@echo "                       Ex: make alert-check ALERT_ARGS=\"--health-url http://localhost:8000/health\""
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

drift:
	PYTHONPATH=. $(UV_RUN) python -m src.monitoring.drift_detector --window-days 7 $(DRIFT_ARGS)

perf-monitor:
	PYTHONPATH=. $(UV_RUN) python -m src.monitoring.performance_monitor --window-days 30 $(PERF_ARGS)

alert-check:
	PYTHONPATH=. $(UV_RUN) python -m src.monitoring.alert_check $(ALERT_ARGS)