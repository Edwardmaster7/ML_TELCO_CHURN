"""Configuração de logging estruturado em JSON para a API e scripts de monitoramento."""
import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Formata entradas de log como objetos JSON com campos padrão de observabilidade.

    Campos fixos: timestamp, level, module, message.
    Campos opcionais (via ``extra={}``): request_id, model_version, latency_ms,
    status, method, path, input_summary, error_type.
    """

    _EXTRA_FIELDS = (
        "request_id",
        "model_version",
        "latency_ms",
        "status",
        "method",
        "path",
        "input_summary",
        "error_type",
    )

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }

        for field in self._EXTRA_FIELDS:
            if hasattr(record, field):
                log_entry[field] = getattr(record, field)

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_json_logging(level: str = "INFO") -> None:
    """Substitui a configuração de logging global por saída JSON estruturada.

    Deve ser chamada **antes** de qualquer import que configure loggers, ou
    logo no início do ``lifespan`` da aplicação.

    Args:
        level: Nível mínimo de log (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, etc.).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)
