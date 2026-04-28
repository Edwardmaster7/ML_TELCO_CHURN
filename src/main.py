"""Ponto de entrada (Router) principal do FastAPI.

Responsável pelas rotas HTTP e inicialização assíncrona (Lifespan) do processo.
"""
import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.core.ml_service import MLService
from src.core.middlewares import LoggingMiddleware
from src.core.config import CONFIG as settings
from src.api.v1.api import router as v1_router

# Configuração global de logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)

logger = logging.getLogger(__name__)
ml_service = MLService()

MODEL_NAME = os.getenv("MODEL_NAME", "MLP_Focal_KFold_Script")
STAGE_OR_ALIAS = os.getenv("MODEL_STAGE", "production")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Contexto de inicialização Cold-Start para FastAPI conectando na plataforma MLFlow.

    Garante que os artefatos de treinamento serão hidratados em memória da RAM antes
    que qualquer requisição externa tente usar o endpoint de predição.

    Args:
        app (FastAPI): Instância da aplicação principal.
    """
    logger.info("Iniciando FastAPI e carregando modelos no MLService...")
    try:
        ml_service.load_model_artifacts(model_name=MODEL_NAME, stage_or_alias=STAGE_OR_ALIAS)
    except Exception as e:
        logger.error(f"Erro no Lifespan: {e}")
    yield
    logger.info("Desligando API.")

app = FastAPI(
    title="API de Previsão de Churn",
    description="Tech Challenge - ML Engineering API para inferência do modelo PyTorch (MLP_Focal_KFold)",
    version="1.0.0",
    lifespan=lifespan
)

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
        dict: Dicionário de status da API.

    Raises:
        HTTPException: Erro 503 Service Unavailable caso o modelo não tenha sido carregado.
    """

    model_loaded = ml_service.model is not None and ml_service.preprocessor is not None
    status = "ok" if model_loaded else "degraded"

    return {"status": status, "model_loaded": model_loaded}