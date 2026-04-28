.PHONY: help test run

help:
	@echo "======================================================================"
	@echo "                       ML_TELCO_CHURN API                             "
	@echo "======================================================================"
	@echo ""
	@echo "Comandos disponíveis:"
	@echo "  make help       - Mostra esta mensagem de ajuda"
	@echo "  make test       - Executa toda a suíte de testes com Pytest"
	@echo "  make run        - Inicia o servidor FastAPI em modo de desenvolvimento"
	@echo ""

test:
	uv run pytest tests/ -v

run:
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
