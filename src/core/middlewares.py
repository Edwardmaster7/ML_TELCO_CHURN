"""Módulo de Middlewares ASGI para observabilidade."""
import time
import uuid
import logging
from contextvars import ContextVar
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.monitoring.metrics import PREDICTIONS_TOTAL, VALIDATION_FAILURES

logger = logging.getLogger("api_logger")

# ContextVar que carrega o correlation_id para todos os logs da requisição corrente.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware FastAPI para mensuração de tráfego HTTP.

    Gera um ``correlation_id`` por requisição (UUID v4, ou reutiliza X-Request-ID
    enviado pelo cliente), propaga via ContextVar para que logs downstream incluam
    o mesmo identificador, e devolve o header X-Request-ID na resposta.
    """

    async def dispatch(self, request: Request, call_next):
        """Processa requisição, emite log JSON estruturado e atualiza métricas Prometheus.

        Args:
            request (Request): O objeto de requisição do FastAPI.
            call_next (Callable): Função delegate para seguir na chain de middlewares/rotas.

        Returns:
            Response: Resposta da requisição original com header X-Request-ID.
        """
        correlation_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request_id_var.set(correlation_id)

        start_time = time.time()
        response = await call_next(request)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            "http_request",
            extra={
                "request_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": latency_ms,
            },
        )

        # Atualiza métricas Prometheus
        if request.url.path == "/api/v1/predict":
            label = "churn" if response.status_code == 200 else "error"
            PREDICTIONS_TOTAL.labels(prediction_class=label).inc()
        if response.status_code == 422:
            VALIDATION_FAILURES.inc()

        response.headers["X-Request-ID"] = correlation_id
        return response
