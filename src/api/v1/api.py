from fastapi import APIRouter, HTTPException
from src.core.schemas import ChurnPredictionRequest, ChurnPredictionResponse
from src.core.ml_service import MLService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/predict", response_model=ChurnPredictionResponse)
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
    ml_service = MLService()  # Instância local do serviço de ML (pode ser otimizada para singleton se necessário)

    try:
        result = ml_service.predict_churn(request.model_dump())
        return result
    except RuntimeError as re:
        logger.error(f"RuntimeError na predição: {re}")
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.error(f"Erro inesperado na predição: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during inference.")