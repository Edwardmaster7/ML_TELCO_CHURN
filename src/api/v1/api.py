from datetime import datetime, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import PredictionLog, SessionLocal
from src.core.deps import get_db
from src.core.middlewares import request_id_var
from src.core.ml_service import MLService
from src.core.schemas import ChurnPredictionRequest, ChurnPredictionResponse
from src.monitoring.metrics import PREDICTION_PROBABILITY, PREDICTIONS_TOTAL
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


async def _log_prediction_to_db(
    customer_id: str,
    probability: float,
    prediction: int,
    model_version: str,
    req_id: str,
) -> None:
    """Background task: persiste log da predição no banco SQLite."""
    try:
        async with SessionLocal() as db:
            log_entry = PredictionLog(
                customer_id=customer_id,
                churn_probability=probability,
                churn_prediction=prediction,
                predicted_at=datetime.now(timezone.utc),
                model_version=model_version,
                request_id=req_id,
            )
            db.add(log_entry)
            await db.commit()
    except Exception as exc:
        logger.warning(f"Falha ao persistir PredictionLog: {exc}")


@router.post("/predict", response_model=ChurnPredictionResponse)
def predict(request: ChurnPredictionRequest, background_tasks: BackgroundTasks):
    """Invoca as pipelines de predição do modelo campeão (Pytorch Focal Loss).

    Args:
        request (ChurnPredictionRequest): Payload pydantic contendo atributos em string, inteiro ou float da operadora.
        background_tasks: BackgroundTasks do FastAPI para log assíncrono.

    Returns:
        ChurnPredictionResponse: Resposta serializada garantindo que um número real e a predição chegam no cliente.

    Raises:
        HTTPException: 503 Se ocorrer Runtime de Modelos offline.
        HTTPException: 500 Se ocorrer uma quebra silenciosa e imprevista do pandas.
    """
    ml_service = MLService()

    try:
        result = ml_service.predict_churn(request.model_dump())
    except RuntimeError as re:
        logger.error(f"RuntimeError na predição: {re}")
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.error(f"Erro inesperado na predição: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during inference.")

    # Atualizar métricas Prometheus
    churn_prob = result.get("churn_probability", 0.0)
    churn_pred = result.get("churn_prediction", 0)
    PREDICTION_PROBABILITY.observe(churn_prob)
    PREDICTIONS_TOTAL.labels(prediction_class="churn" if churn_pred == 1 else "no_churn").inc()

    # Persistir log da predição de forma assíncrona
    background_tasks.add_task(
        _log_prediction_to_db,
        customer_id=str(request.customerID),
        probability=float(churn_prob),
        prediction=int(churn_pred),
        model_version=ml_service.model_version or "unknown",
        req_id=request_id_var.get(),
    )

    return result


@router.post("/feedback/{customer_id}", status_code=200)
async def register_feedback(
    customer_id: str,
    actual_churn: Annotated[Literal[0, 1], Query(description="Ground truth: 0 = não churn, 1 = churn")],
    db: AsyncSession = Depends(get_db),
):
    """Registra o ground truth de churn para um cliente já predito.

    Atualiza o campo ``actual_churn`` e ``feedback_at`` no registro mais recente
    de ``prediction_logs`` para o ``customer_id`` fornecido.

    Args:
        customer_id:  Identificador do cliente.
        actual_churn: Rótulo real (0 ou 1).
        db:           Sessão assíncrona injetada por dependência.

    Returns:
        dict: Confirmação com customer_id e actual_churn registrados.

    Raises:
        HTTPException: 404 se nenhuma predição for encontrada para o customer_id.
    """
    stmt = (
        select(PredictionLog)
        .where(PredictionLog.customer_id == customer_id)
        .order_by(PredictionLog.predicted_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    log_entry: PredictionLog | None = result.scalar_one_or_none()

    if log_entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Nenhuma predição encontrada para customer_id '{customer_id}'.",
        )

    log_entry.actual_churn = actual_churn
    log_entry.feedback_at = datetime.now(timezone.utc)
    await db.commit()

    logger.info(
        "feedback_registrado",
        extra={"customer_id": customer_id, "actual_churn": actual_churn},
    )

    return {"customer_id": customer_id, "actual_churn": actual_churn, "status": "ok"}