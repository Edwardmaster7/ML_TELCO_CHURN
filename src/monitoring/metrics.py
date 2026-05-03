"""Configuração e exposição de métricas Prometheus para a API de churn.

Uso em src/main.py:
    from src.monitoring.metrics import setup_metrics, MODEL_LOADED
    setup_metrics(app)
    MODEL_LOADED.set(1)
"""
import logging
from fastapi import FastAPI

logger = logging.getLogger(__name__)

# ── Importação defensiva: prometheus_client pode não estar instalado ──────────
try:
    from prometheus_client import Counter, Gauge, Histogram
    from prometheus_fastapi_instrumentator import Instrumentator

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning(
        "prometheus_client / prometheus_fastapi_instrumentator não instalados. "
        "Métricas Prometheus desativadas. "
        "Adicione 'prometheus-fastapi-instrumentator' às dependências."
    )

# ── Métricas customizadas ─────────────────────────────────────────────────────

if _PROMETHEUS_AVAILABLE:
    PREDICTIONS_TOTAL = Counter(
        "churn_predictions_total",
        "Total de predições realizadas, por classe predita",
        ["prediction_class"],  # labels: "churn" | "no_churn"
    )

    MODEL_LOADED = Gauge(
        "churn_model_loaded",
        "1 se o modelo PyTorch está carregado em memória, 0 caso contrário",
    )

    PREDICTION_PROBABILITY = Histogram(
        "churn_prediction_probability",
        "Distribuição das probabilidades de churn retornadas pela API",
        buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )

    VALIDATION_FAILURES = Counter(
        "churn_data_validation_failures_total",
        "Total de requisições rejeitadas por falha de validação de schema (HTTP 422)",
    )

    DATA_DRIFT_PSI = Gauge(
        "churn_data_drift_psi",
        "Último PSI calculado por feature numérica",
        ["feature"],
    )

    DATA_DRIFT_JSD = Gauge(
        "churn_data_drift_jsd",
        "Último JSD calculado para a distribuição de churn_probability",
    )

else:
    # Stubs silenciosos para manter o restante do código sem guard clauses.
    class _NoOp:
        def labels(self, **_):
            return self

        def inc(self, *_):
            pass

        def observe(self, *_):
            pass

        def set(self, *_):
            pass

    PREDICTIONS_TOTAL = _NoOp()  # type: ignore[assignment]
    MODEL_LOADED = _NoOp()  # type: ignore[assignment]
    PREDICTION_PROBABILITY = _NoOp()  # type: ignore[assignment]
    VALIDATION_FAILURES = _NoOp()  # type: ignore[assignment]
    DATA_DRIFT_PSI = _NoOp()  # type: ignore[assignment]
    DATA_DRIFT_JSD = _NoOp()  # type: ignore[assignment]


# ── Setup da instrumentação automática ───────────────────────────────────────

def setup_metrics(app: FastAPI) -> None:
    """Instrumenta o app FastAPI e expõe o endpoint /metrics.

    Se ``prometheus_fastapi_instrumentator`` não estiver instalado, a função
    retorna silenciosamente sem modificar a aplicação.

    Args:
        app: Instância FastAPI já configurada (antes de ``app.include_router``).
    """
    if not _PROMETHEUS_AVAILABLE:
        logger.warning("setup_metrics: Prometheus não disponível — /metrics não exposto.")
        return

    Instrumentator(
        should_group_status_codes=False,
        excluded_handlers=["/health", "/metrics"],
    ).instrument(app).expose(app, endpoint="/metrics")

    logger.info("Prometheus /metrics exposto com sucesso.")
