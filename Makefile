.PHONY: help test run train docker-up docker-down mlflow-ui

help:
	@echo "======================================================================"
	@echo "                       ML_TELCO_CHURN API                             "
	@echo "======================================================================"
	@echo ""
	@echo "Comandos disponíveis:"
	@echo "  make help       - Mostra esta mensagem de ajuda"
	@echo "  make test       - Executa toda a suíte de testes com Pytest"
	@echo "  make run        - Inicia o servidor FastAPI em modo de desenvolvimento"
	@echo "  make train      - Executa o script de treinamento do modelo"
	@echo "  make mlflow-ui  - Inicia a interface do MLflow localmente"
	@echo "  make docker-up  - Sobe os serviços via Docker Compose (build e background)"
	@echo "  make docker-down- Para e remove os serviços do Docker Compose"
	@echo ""

test:
	uv run pytest tests/ -v

run:
	uv run fastapi dev src/main.py

train:
	MLFLOW_TRACKING_URI=sqlite:///mlflow.db PYTHONPATH=. uv run python src/models/train.py --epochs 5

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
