"""Módulo de Middlewares ASGI para observabilidade."""
import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    logger.addHandler(handler)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware FastAPI para mensuração de tráfego HTTP.

    Interceptor que avalia o tempo de cada requisição e lança um log
    estruturado no output sem uso de commands de print().
    """

    async def dispatch(self, request: Request, call_next):
        """Processa requisição e emite log formatado em pares key=value.

        Args:
            request (Request): O objeto de requisição do FastAPI.
            call_next (Callable): Função delegate para seguir na chain de middlewares/rotas.

        Returns:
            Response: Resposta da requisição original, inalterada.
        """
        start_time = time.time()

        response = await call_next(request)

        process_time = time.time() - start_time
        latency_ms = round(process_time * 1000, 2)

        log_msg = f"method={request.method} path={request.url.path} status={response.status_code} latency={latency_ms}ms"
        logger.info(log_msg)

        return response
