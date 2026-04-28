"""Ponto de entrada (Router) principal do FastAPI.

Responsável pelas rotas HTTP e inicialização assíncrona (Lifespan) do processo.
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import ChurnPredictionRequest, ChurnPredictionResponse
from src.api.ml_service import MLService
from src.api.middlewares import LoggingMiddleware
from src.config import CONFIG as settings

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

@app.post("/predict", response_model=ChurnPredictionResponse)
def predict(request: ChurnPredictionRequest):
    """Invoca as pipelines de predição do modelo campeão (Pytorch Focal Loss).

    Args:
        request (ChurnPredictionRequest): Payload pydantic contendo atributos em string, inteiro ou float da operadora.

    Returns:
        ChurnPredictionResponse: Resposta serializada garantindo que um número real e a predição chegam no cliente.

    Raises:
        HTTPException: 503 Se ocorrer Runtime de Modelos offline.
        HTTPException: 500 Se ocorrer uma quebra silenciosa e imprevista do pandas.
    """
    try:
        result = ml_service.predict_churn(request.model_dump())
        return result
    except RuntimeError as re:
        logger.error(f"RuntimeError na predição: {re}")
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.error(f"Erro inesperado na predição: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during inference.")