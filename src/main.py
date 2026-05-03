"""Ponto de entrada (Router) principal do FastAPI.

Responsável pelas rotas HTTP e inicialização assíncrona (Lifespan) do processo.
"""
import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.core.ml_service import MLService
from src.core.middlewares import LoggingMiddleware
from src.core.config import CONFIG as settings
from src.api.v1.api import router as v1_router
from src.core.logging_config import setup_json_logging

# Logging estruturado JSON (substitui basicConfig simples)
setup_json_logging(level="INFO")

logger = logging.getLogger(__name__)
ml_service = MLService()

MODEL_NAME = os.getenv("MODEL_NAME", "MLP_Focal_KFold_Script")
STAGE_OR_ALIAS = os.getenv("MODEL_STAGE", "production")

_app_start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Contexto de inicialização Cold-Start para FastAPI conectando na plataforma MLFlow.

    Garante que os artefatos de treinamento serão hidratados em memória da RAM antes
    que qualquer requisição externa tente usar o endpoint de predição.

    Args:
        app (FastAPI): Instância da aplicação principal.
    """
    from src.core.database import init_db
    from src.monitoring.metrics import MODEL_LOADED

    logger.info("Iniciando FastAPI: criando tabelas no banco de dados...")
    try:
        await init_db()
    except Exception as exc:
        logger.error(f"Erro ao inicializar banco de dados: {exc}")

    logger.info("Carregando artefatos do modelo no MLService...")
    try:
        ml_service.load_model_artifacts(
            model_name=MODEL_NAME,
            stage_or_alias=STAGE_OR_ALIAS,
            tracking_uri=settings.mlflow_tracking_uri
        )
        MODEL_LOADED.set(1)
    except Exception as e:
        logger.error(f"Erro no Lifespan: {e}")
        MODEL_LOADED.set(0)
    yield
    logger.info("Desligando API.")

app = FastAPI(
    title="API de Previsão de Churn",
    description="Tech Challenge - ML Engineering API para inferência do modelo PyTorch (MLP_Focal_KFold)",
    version="1.0.0",
    lifespan=lifespan
)

# Registra métricas Prometheus no /metrics
from src.monitoring.metrics import setup_metrics
setup_metrics(app)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

app.add_middleware(LoggingMiddleware)

# Adiciona rotas da API
app.include_router(v1_router, prefix="/api/v1")

@app.get("/health")
def health_check():
    """Verifica a saúde básica do contêiner e prontidão dos artefatos scikit/torch.

    Returns:
        dict: Dicionário de status da API com informações do modelo e uptime.
    """
    model_loaded = ml_service.model is not None and ml_service.preprocessor is not None
    status = "ok" if model_loaded else "degraded"
    uptime_seconds = round(time.time() - _app_start_time, 1)

    return {
        "status": status,
        "model_loaded": model_loaded,
        "model_name": ml_service.model_name,
        "model_version": ml_service.model_version,
        "loaded_at": ml_service.loaded_at.isoformat() if ml_service.loaded_at else None,
        "uptime_seconds": uptime_seconds,
        "mlflow_uri": settings.mlflow_tracking_uri,
    }