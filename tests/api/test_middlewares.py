import pytest
import logging
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.core.middlewares import LoggingMiddleware

def test_logging_middleware_records_latency(caplog):
    """Testa se o middleware injeta com sucesso um log de monitoramento.

    Verifica se a resposta de uma rota qualquer dispara o interceptor
    que salva logs contendo os termos method, path, status e latency.

    Args:
        caplog (LogCaptureFixture): Objeto nativo do pytest para capturar emissões do logging.
    """
    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

    @app.get("/test")
    def test_route():
        return {"msg": "ok"}

    client = TestClient(app)

    with caplog.at_level(logging.INFO):
        response = client.get("/test")

    assert response.status_code == 200

    logs = [record.message for record in caplog.records]
    log_found = any("method=GET" in log and "path=/test" in log and "status=200" in log and "latency=" in log for log in logs)
    assert log_found, f"Expected structured log not found. Logs: {logs}"
