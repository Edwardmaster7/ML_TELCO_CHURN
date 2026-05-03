.PHONY: help test run train mlflowui db-upgrade drift perf-monitor alert-check docker-up docker-down docker-logs docker-build docker-restart docker-ps docker-clean docker-train docker-train-full docker-health docker-check-model

# Variáveis de comando customizáveis
EPOCHS ?= 5
MLFLOW_DB_URI ?= sqlite:///mlflow.db
MLFLOW_TRACKING_URI ?= $(MLFLOW_DB_URI)
DOCKER_MLFLOW_URI ?= http://localhost:5001
UV_RUN ?= uv run
DRIFT_ARGS ?=
PERF_ARGS ?=
ALERT_ARGS ?=
ARGS ?=

# Allow shorthand like: make train-sync-docker EPOCHS 400
ifneq (,$(filter train-sync-docker,$(MAKECMDGOALS)))
EXTRA_ARGS := $(MAKECMDGOALS)
ifneq ($(strip $(EXTRA_ARGS)),)
EPOCHS_VALUE := $(filter-out EPOCHS,$(EXTRA_ARGS))
ifneq ($(strip $(EPOCHS_VALUE)),)
EPOCHS := $(firstword $(EPOCHS_VALUE))
endif
$(eval $(EXTRA_ARGS):;@:)
endif
endif

help:
	@echo "======================================================================"
	@echo "                       ML_TELCO_CHURN API                             "
	@echo "======================================================================"
	@echo ""
	@echo "Docker Setup:"
	@echo "  make docker-up                  - Inicia MLflow, API, Prometheus, Grafana"
	@echo "  make docker-down                - Para todos os serviços"
	@echo "  make docker-clean               - Remove containers, volumes e cache"
	@echo ""
	@echo "⭐ RECOMMENDED WORKFLOWS:"
	@echo "  make docker-from-scratch        - 🔥 Limpa e inicia do zero"
	@echo "  make docker-train-full          - 🐳 Treina no Docker (CPU)"
	@echo ""
	@echo "Docker Operations:"
	@echo "  make docker-train               - Treina no container"
	@echo "  make docker-alias               - Configura alias 'production'"
	@echo "  make docker-restart             - Reinicia API"
	@echo "  make docker-health              - Verifica model_loaded?"
	@echo "  make docker-check-model         - Lista versões"
	@echo "  make docker-ps                  - Status dos containers"
	@echo "  make docker-logs                - Logs em tempo real"
	@echo ""
	@echo "Local Training (with GPU):"
	@echo "  make train-local-setup          - Info de setup"
	@echo "  make train                      - Treina localmente (padrão: 5 epochs)"
	@echo "  make train EPOCHS=100           - Treina localmente com 100 epochs"
	@echo "  make mlflowui                   - MLflow UI local"
	@echo ""
	@echo "Local Development:"
	@echo "  make test                       - Executa testes"
	@echo "  make run                        - FastAPI local (dev mode)"
	@echo ""
	@echo "Monitoring:"
	@echo "  make drift                      - Data drift detection"
	@echo "  make perf-monitor               - Performance evaluation"
	@echo "  make alert-check                - Verifica alertas"
	@echo ""

test:
	$(UV_RUN) pytest tests/ -v

db-upgrade:
	$(UV_RUN) mlflow db upgrade $(MLFLOW_DB_URI)

run:
	uv sync && $(MAKE) db-upgrade && $(UV_RUN) uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

train:
	@echo "🎓 Executando treinamento LOCAL ($(EPOCHS) epochs)"
	@echo "💡 Use GPU no seu computador (recomendado para grandes datasets)"
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) PYTHONPATH=. $(UV_RUN) python src/models/train.py --epochs $(EPOCHS) $(ARGS)

mlflowui:
	uv sync && $(UV_RUN) mlflow ui --backend-store-uri $(MLFLOW_DB_URI) --port 5001

drift:
	PYTHONPATH=. $(UV_RUN) python -m src.monitoring.drift_detector --window-days 7 $(DRIFT_ARGS)

perf-monitor:
	PYTHONPATH=. $(UV_RUN) python -m src.monitoring.performance_monitor --window-days 30 $(PERF_ARGS)

alert-check:
	PYTHONPATH=. $(UV_RUN) python -m src.monitoring.alert_check $(ALERT_ARGS)

# ============================================================================
# Docker Compose Commands - Core
# ============================================================================

docker-up:
	@echo "🔍 Detectando GPU..."
	docker compose up -d
	@echo "⏳ Aguardando MLflow inicializar..."
	@sleep 6
	@echo ""
	@echo "✅ Serviços iniciados:"
	@echo "   🌐 API:        http://localhost:8000/docs"
	@echo "   📊 MLflow:     http://localhost:5001"
	@echo "   📈 Prometheus: http://localhost:9090"
	@echo "   🎨 Grafana:    http://localhost:3000"
	@echo ""
	@docker compose ps

docker-down:
	@echo "⏹️  Parando Docker Compose..."
	docker compose down
	@echo "✅ Serviços parados"

docker-clean:
	@echo "🧹 Limpando tudo (containers, volumes, cache)..."
	docker compose down -v
	docker system prune -f
	@echo "✅ Limpeza concluída"

docker-ps:
	@echo "📋 Status dos containers:"
	@docker compose ps

docker-logs:
	@echo "📜 Logs da API (Ctrl+C para sair):"
	docker compose logs -f api

docker-build:
	@echo "🔨 Construindo imagem da API..."
	docker compose build api
	@echo "✅ Build concluído"

docker-restart:
	@echo "🔄 Reiniciando API..."
	docker compose restart api
	@sleep 3
	@echo "✅ API reiniciada"

# ============================================================================
# Docker Training & Model Management
# ============================================================================

docker-train:
	@echo "🎓 Treinando modelo no Docker ($(EPOCHS) epochs)..."
	docker compose exec -e MLFLOW_TRACKING_URI=http://mlflow:5000 -e PYTHONPATH=/app api python src/models/train.py --epochs $(EPOCHS)
	@echo "✅ Treinamento concluído"
	@echo "🏷️  Configurando alias 'production'..."
	docker compose exec api python scripts/set_model_alias.py
	@echo "✅ Alias configurado"

docker-alias:
	@docker compose exec api python scripts/set_model_alias.py

docker-check-model:
	@echo "📦 Versões do modelo no registry:"
	docker compose exec api python scripts/check_model_versions.py

docker-health:
	@echo "🏥 Verificando saúde da API..."
	@curl -s http://localhost:8000/health | python3 -m json.tool || echo "❌ API não respondeu"


# ============================================================================
# Quick start sequences
# ============================================================================

docker-from-scratch:
	@echo "🔥 Limpando e reiniciando do zero..."
	@$(MAKE) docker-clean
	@$(MAKE) docker-up
	@echo ""
	@echo "✅ Stack pronta. Próximo: make docker-train-full"
	@echo ""

# ============================================================================
# ⭐ Main Command: One-Step Training
# ============================================================================

docker-train-full:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════╗"
	@echo "║     🚀 DOCKER FULL WORKFLOW (Train → Alias → Ready)   ║"
	@echo "╚════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Step 1/4: Verificando Docker..."
	@docker compose ps > /dev/null 2>&1 || (echo "❌ Docker não está rodando. Execute: make docker-up" && exit 1)
	@echo "  ✅ Docker está pronto"
	@echo ""
	@echo "Step 2/4: Treinando modelo ($(EPOCHS) epochs)..."
	@$(MAKE) docker-train EPOCHS=$(EPOCHS)
	@echo ""
	@echo "Step 3/4: Reiniciando API..."
	@$(MAKE) docker-restart
	@echo ""
	@echo "Aguardando API carregar modelo..."
	@sleep 3
	@echo ""
	@echo "╔════════════════════════════════════════════════════════╗"
	@echo "║                 ✅ PRONTO PARA PRODUÇÃO!              ║"
	@echo "╚════════════════════════════════════════════════════════╝"
	@echo ""
	@$(MAKE) docker-health
	@echo ""
	@$(MAKE) docker-check-model
	@echo ""
	@echo "Acesse:"
	@echo "  🌐 API:    http://localhost:8000/docs"
	@echo "  📊 MLflow: http://localhost:5001"
	@echo ""

train-local-setup:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════╗"
	@echo "║          📋 LOCAL TRAINING SETUP (Com GPU)            ║"
	@echo "╚════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "✅ Pré-requisitos:"
	@echo "   1. MLflow UI rodando localmente (make mlflowui &)"
	@echo "   2. GPU disponível (torch deve detectar CUDA/MPS)"
	@echo "   3. Docker rodando com MLflow (make docker-up)"
	@echo ""
	@echo "📌 Fluxo:"
	@echo "   1. make mlflowui             # Em outro terminal"
	@echo "   2. make train EPOCHS=100"
	@echo ""
