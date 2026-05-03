"""Monitor de performance do modelo quando ground truth está disponível.

Lê predições da tabela ``prediction_logs`` (gerada pela API) e calcula métricas
de ML para amostras que já receberam feedback (``actual_churn`` preenchido).

Uso (CLI cross-platform):
    python -m src.monitoring.performance_monitor --window-days 30
    python -m src.monitoring.performance_monitor --predictions-csv preds.csv --labels-csv labels.csv
    python -m src.monitoring.performance_monitor --help

Saída:
    JSON em ``monitoring/reports/performance/perf_report_YYYY-MM-DD.json``
    Log no MLflow (se --log-to-mlflow)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn não disponível. Métricas de performance desativadas.")

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Cálculo de métricas
# ─────────────────────────────────────────────────────────────────────────────

def compute_performance_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calcula métricas de classificação para um conjunto de predições + labels.

    Args:
        y_true:    Rótulos reais (0 ou 1).
        y_prob:    Probabilidades preditas ∈ [0, 1].
        threshold: Limiar de classificação (padrão 0.5).

    Returns:
        Dicionário com ``pr_auc``, ``roc_auc``, ``f1``, ``precision``, ``recall``,
        ``churn_rate_actual``, ``churn_rate_predicted``.
    """
    if not _SKLEARN_AVAILABLE:
        return {}

    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, float] = {}

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        metrics["pr_auc"] = 0.0

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = 0.0

    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["churn_rate_actual"] = float(np.mean(y_true))
    metrics["churn_rate_predicted"] = float(np.mean(y_pred))
    metrics["n_samples"] = int(len(y_true))

    return metrics


def evaluate_by_time_window(
    df: pd.DataFrame,
    window_days: int = 30,
    date_col: str = "predicted_at",
    prob_col: str = "churn_probability",
    label_col: str = "actual_churn",
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Avalia performance por janela deslizante de ``window_days`` dias.

    Args:
        df:           DataFrame com colunas ``date_col``, ``prob_col``, ``label_col``.
        window_days:  Tamanho da janela em dias.
        date_col:     Coluna de data da predição.
        prob_col:     Coluna de probabilidade predita.
        label_col:    Coluna com o ground truth.
        threshold:    Limiar de classificação.

    Returns:
        Lista de dicionários, um por janela, com métricas + período.
    """
    if df.empty or label_col not in df.columns:
        return []

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, prob_col, label_col])

    if df.empty:
        return []

    results = []
    min_date = df[date_col].min()
    max_date = df[date_col].max()

    current = min_date
    while current <= max_date:
        end = current + timedelta(days=window_days)
        window_df = df[(df[date_col] >= current) & (df[date_col] < end)]

        if len(window_df) >= 10:
            metrics = compute_performance_metrics(
                y_true=window_df[label_col].values,
                y_prob=window_df[prob_col].values,
                threshold=threshold,
            )
            metrics["window_start"] = current.isoformat()
            metrics["window_end"] = end.isoformat()
            results.append(metrics)

        current = end

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento de dados
# ─────────────────────────────────────────────────────────────────────────────

def load_labeled_predictions_from_db(db_url: str, window_days: int) -> pd.DataFrame:
    """Carrega predições com ground truth disponível da tabela prediction_logs.

    Args:
        db_url:      URL do banco SQLite.
        window_days: Janela de dias retroativos.

    Returns:
        DataFrame com as colunas da tabela ou vazio se nenhum dado disponível.
    """
    try:
        import sqlite3

        db_path = db_url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
        if not Path(db_path).exists():
            logger.warning(f"Banco SQLite não encontrado: {db_path}")
            return pd.DataFrame()

        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            """
            SELECT customer_id, churn_probability, churn_prediction,
                   predicted_at, actual_churn, feedback_at, model_version
            FROM prediction_logs
            WHERE actual_churn IS NOT NULL
              AND predicted_at >= ?
            """,
            conn,
            params=(cutoff.strftime("%Y-%m-%d %H:%M:%S"),),
        )
        conn.close()
        logger.info(f"Carregadas {len(df)} predições com ground truth.")
        return df
    except Exception as exc:
        logger.warning(f"Erro ao carregar prediction_logs: {exc}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Geração do relatório
# ─────────────────────────────────────────────────────────────────────────────

def generate_performance_report(
    df: pd.DataFrame,
    output_dir: str = "monitoring/reports/performance",
    window_days: int = 30,
    pr_auc_critical: float = 0.60,
    pr_auc_warning: float = 0.65,
    log_to_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Gera relatório de performance com ground truth.

    Args:
        df:                  DataFrame com predições + labels (de ``load_labeled_predictions_from_db``).
        output_dir:          Diretório de saída.
        window_days:         Janela de avaliação em dias.
        pr_auc_critical:     Threshold PR-AUC crítico.
        pr_auc_warning:      Threshold PR-AUC de aviso.
        log_to_mlflow:       Se True, loga no MLflow.
        mlflow_tracking_uri: URI do MLflow.

    Returns:
        Dicionário com o relatório completo.
    """
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(df),
        "window_days": window_days,
        "overall_metrics": {},
        "window_metrics": [],
        "alerts": [],
        "summary": {"status": "ok", "critical_count": 0, "warning_count": 0},
    }

    if df.empty or "actual_churn" not in df.columns:
        logger.warning("Nenhum dado com ground truth disponível.")
        return report

    df_labeled = df.dropna(subset=["actual_churn", "churn_probability"])
    if df_labeled.empty:
        logger.warning("Nenhuma amostra com actual_churn preenchido.")
        return report

    # ── 1. Métricas gerais ────────────────────────────────────────────────────
    overall = compute_performance_metrics(
        y_true=df_labeled["actual_churn"].values,
        y_prob=df_labeled["churn_probability"].values,
    )
    report["overall_metrics"] = overall

    # ── 2. Métricas por janela ────────────────────────────────────────────────
    if "predicted_at" in df_labeled.columns:
        report["window_metrics"] = evaluate_by_time_window(
            df_labeled,
            window_days=window_days,
        )

    # ── 3. Alertas ────────────────────────────────────────────────────────────
    pr_auc = overall.get("pr_auc", 1.0)
    if pr_auc < pr_auc_critical:
        report["alerts"].append(
            {"type": "CRITICAL", "metric": "pr_auc", "value": pr_auc, "threshold": pr_auc_critical}
        )
        report["summary"]["critical_count"] += 1
    elif pr_auc < pr_auc_warning:
        report["alerts"].append(
            {"type": "WARNING", "metric": "pr_auc", "value": pr_auc, "threshold": pr_auc_warning}
        )
        report["summary"]["warning_count"] += 1

    if report["summary"]["critical_count"] > 0:
        report["summary"]["status"] = "critical"
    elif report["summary"]["warning_count"] > 0:
        report["summary"]["status"] = "warning"

    # ── 4. Salvar relatório ───────────────────────────────────────────────────
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / f"perf_report_{date.today()}.json"
    with open(report_file, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    logger.info(f"Relatório de performance salvo em: {report_file}")

    # ── 5. Log no MLflow ──────────────────────────────────────────────────────
    if log_to_mlflow and _MLFLOW_AVAILABLE and mlflow_tracking_uri:
        _log_performance_to_mlflow(overall, mlflow_tracking_uri)

    return report


def _log_performance_to_mlflow(metrics: dict, tracking_uri: str) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("monitoring")
        with mlflow.start_run(run_name=f"monitor_perf_{date.today()}"):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"monitor_{key}", value)
        logger.info("Métricas de performance registradas no MLflow.")
    except Exception as exc:
        logger.warning(f"Erro ao logar performance no MLflow: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.monitoring.performance_monitor",
        description="Avalia performance do modelo com ground truth disponível.",
    )
    p.add_argument(
        "--predictions-csv",
        default=None,
        help="CSV com colunas: customer_id, churn_probability, churn_prediction, predicted_at.",
    )
    p.add_argument(
        "--labels-csv",
        default=None,
        help="CSV com colunas: customer_id, actual_churn. Necessário se --predictions-csv.",
    )
    p.add_argument(
        "--db-url",
        default=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///mlflow.db"),
        help="URL do banco SQLite (usa prediction_logs se não houver CSV).",
    )
    p.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Janela de dias retroativos (padrão: 30).",
    )
    p.add_argument(
        "--output-dir",
        default="monitoring/reports/performance",
        help="Diretório para salvar o relatório JSON.",
    )
    p.add_argument(
        "--log-to-mlflow",
        action="store_true",
        help="Se definido, loga as métricas no MLflow.",
    )
    p.add_argument(
        "--mlflow-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        help="URI do MLflow Tracking Server.",
    )
    p.add_argument(
        "--config",
        default="configs/monitoring.yaml",
        help="Caminho para o arquivo de configuração YAML.",
    )
    return p


def main() -> int:
    """Entrypoint CLI. Retorna 0 se OK, 1 se há alertas críticos."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    args = _build_parser().parse_args()

    cfg: dict = {}
    try:
        import yaml

        with open(args.config, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        pass

    perf_cfg = cfg.get("performance", {})

    # Carregar dados
    if args.predictions_csv:
        preds = pd.read_csv(args.predictions_csv)
        if args.labels_csv:
            labels = pd.read_csv(args.labels_csv)
            df = preds.merge(labels, on="customer_id", how="inner")
        else:
            df = preds
    else:
        df = load_labeled_predictions_from_db(args.db_url, args.window_days)

    # Gerar relatório
    report = generate_performance_report(
        df=df,
        output_dir=args.output_dir,
        window_days=args.window_days,
        pr_auc_critical=perf_cfg.get("pr_auc_critical", 0.60),
        pr_auc_warning=perf_cfg.get("pr_auc_warning", 0.65),
        log_to_mlflow=args.log_to_mlflow,
        mlflow_tracking_uri=args.mlflow_uri,
    )

    # Print resumo
    status = report["summary"]["status"]
    overall = report.get("overall_metrics", {})
    print(f"\n{'='*60}")
    print(f"PERFORMANCE REPORT — {date.today()} — Status: {status.upper()}")
    print(f"Amostras com ground truth: {report['n_samples']}")
    if overall:
        print(f"PR-AUC: {overall.get('pr_auc', 'N/A'):.4f}")
        print(f"F1:     {overall.get('f1', 'N/A'):.4f}")
        print(f"Recall: {overall.get('recall', 'N/A'):.4f}")

    if report["alerts"]:
        print("\nAlertas:")
        for alert in report["alerts"]:
            prefix = "🔴 CRÍTICO" if alert["type"] == "CRITICAL" else "🟡 AVISO"
            print(f"  {prefix} | {alert['metric']} = {alert.get('value', ''):.4f} (threshold={alert.get('threshold', '')})")
    print(f"{'='*60}\n")

    return 1 if report["summary"]["critical_count"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
