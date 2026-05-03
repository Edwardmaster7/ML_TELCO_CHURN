"""Detector de drift de dados para o modelo de churn.

Calcula PSI (numéricas), Teste KS (numéricas), Qui-Quadrado (categóricas) e
JSD (distribuição de probabilidade de churn) entre a distribuição de treino
(baseline) e dados de produção recentes.

Uso (CLI cross-platform):
    python -m src.monitoring.drift_detector \\
        --production-csv monitoring/requests.csv \\
        --window-days 7 \\
        --output-dir monitoring/reports/drift

    python -m src.monitoring.drift_detector --help

Baseline:
    Carregado a partir do artefato MLflow ``training_baseline.json`` (gerado
    por ``src/models/train.py``). Se o MLflow não estiver acessível, usa o
    fallback em ``configs/monitoring_baseline.json``.
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

# ── Importações opcionais ─────────────────────────────────────────────────────
try:
    from scipy.stats import ks_2samp, chi2_contingency
    from scipy.spatial.distance import jensenshannon

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logger.warning("scipy não instalado. KS e JSD desativados.")

try:
    import mlflow
    import mlflow.tracking

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Funções de cálculo de métricas de drift
# ─────────────────────────────────────────────────────────────────────────────

def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calcula o Population Stability Index (PSI) entre duas distribuições numéricas.

    PSI < 0.10 → estável | 0.10–0.20 → mudança moderada | > 0.20 → drift severo.

    Args:
        expected: Array de valores da distribuição de referência (treino).
        actual:   Array de valores da distribuição atual (produção).
        n_bins:   Número de bins do histograma.

    Returns:
        Valor PSI (float ≥ 0).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    _, bin_edges = np.histogram(expected, bins=n_bins)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    eps = 1e-6
    exp_pct = np.maximum(exp_counts / len(expected), eps)
    act_pct = np.maximum(act_counts / len(actual), eps)

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return max(psi, 0.0)


def compute_ks(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Calcula o Teste de Kolmogorov-Smirnov entre duas amostras.

    Args:
        expected: Distribuição de referência.
        actual:   Distribuição atual.

    Returns:
        Dicionário com ``{"ks_statistic": float, "p_value": float}``.
    """
    if not _SCIPY_AVAILABLE:
        return {"ks_statistic": 0.0, "p_value": 1.0}

    stat, pvalue = ks_2samp(expected, actual)
    return {"ks_statistic": float(stat), "p_value": float(pvalue)}


def compute_chi2(
    expected_freqs: dict[str, float],
    actual_counts: dict[str, int],
    n_actual: int,
) -> dict[str, float]:
    """Calcula Qui-Quadrado entre frequências esperadas (treino) e contagens reais.

    Args:
        expected_freqs: Frequências relativas da distribuição de treino {categoria: freq}.
        actual_counts:  Contagens absolutas em produção {categoria: count}.
        n_actual:       Total de amostras em produção.

    Returns:
        Dicionário com ``{"chi2_statistic": float, "p_value": float, "dof": int}``.
    """
    if not _SCIPY_AVAILABLE or n_actual == 0:
        return {"chi2_statistic": 0.0, "p_value": 1.0, "dof": 0}

    categories = sorted(expected_freqs.keys())
    observed = np.array([actual_counts.get(c, 0) for c in categories], dtype=float)
    expected = np.array([expected_freqs.get(c, 0) * n_actual for c in categories], dtype=float)

    # Evitar zeros que causam divisão por zero
    mask = expected > 0
    if mask.sum() < 2:
        return {"chi2_statistic": 0.0, "p_value": 1.0, "dof": 0}

    from scipy.stats import chisquare

    stat, pvalue = chisquare(observed[mask], f_exp=expected[mask])
    return {"chi2_statistic": float(stat), "p_value": float(pvalue), "dof": int(mask.sum() - 1)}


def compute_jsd_from_histograms(
    baseline_hist_counts: list[float],
    current_values: np.ndarray,
    hist_edges: list[float],
) -> float:
    """Calcula Jensen-Shannon Divergence entre histograma baseline e valores atuais.

    Args:
        baseline_hist_counts: Contagens do histograma de treino (sem normalização).
        current_values:       Valores atuais de churn_probability.
        hist_edges:           Arestas do histograma de treino.

    Returns:
        JSD ∈ [0, 1].
    """
    if not _SCIPY_AVAILABLE or len(current_values) == 0:
        return 0.0

    edges = np.array(hist_edges)
    current_counts, _ = np.histogram(current_values, bins=edges)

    eps = 1e-10
    p = np.array(baseline_hist_counts, dtype=float) + eps
    q = np.array(current_counts, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()

    return float(jensenshannon(p, q))


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento do baseline
# ─────────────────────────────────────────────────────────────────────────────

def load_baseline(
    mlflow_tracking_uri: str | None = None,
    model_name: str = "MLP_Focal_KFold_Script",
    alias: str = "production",
    artifact_name: str = "training_baseline.json",
    fallback_path: str = "configs/monitoring_baseline.json",
) -> dict[str, Any] | None:
    """Carrega o baseline de distribuição de treino.

    Tenta primeiro via MLflow (artefato ``training_baseline.json`` do run de produção).
    Se não conseguir, usa o arquivo local de fallback.

    Args:
        mlflow_tracking_uri: URI do tracking server MLflow.
        model_name:          Nome do modelo registrado.
        alias:               Alias de produção (ex: ``"production"``).
        artifact_name:       Nome do arquivo de artefato.
        fallback_path:       Caminho local de fallback.

    Returns:
        Dicionário com estatísticas do baseline, ou ``None`` se nenhuma fonte disponível.
    """
    # Tentativa 1: MLflow
    if _MLFLOW_AVAILABLE and mlflow_tracking_uri:
        try:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_tracking_uri)
            mv = client.get_model_version_by_alias(name=model_name, alias=alias)
            run_id = mv.run_id
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/{artifact_name}"
            )
            with open(local_path, encoding="utf-8") as fh:
                baseline = json.load(fh)
            logger.info(f"Baseline carregado do MLflow (run_id={run_id}).")
            return baseline
        except Exception as exc:
            logger.warning(f"Não foi possível carregar baseline do MLflow: {exc}")

    # Tentativa 2: Arquivo local
    fallback = Path(fallback_path)
    if fallback.exists():
        with open(fallback, encoding="utf-8") as fh:
            baseline = json.load(fh)
        logger.info(f"Baseline carregado do fallback local: {fallback_path}")
        return baseline

    logger.error(
        "Nenhuma fonte de baseline disponível. "
        "Execute 'make train' para gerar o artefato de baseline."
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Geração do relatório de drift
# ─────────────────────────────────────────────────────────────────────────────

def generate_drift_report(
    baseline: dict[str, Any],
    production_df: pd.DataFrame,
    output_dir: str = "monitoring/reports/drift",
    psi_critical: float = 0.20,
    psi_warning: float = 0.10,
    ks_critical: float = 0.10,
    chi2_pvalue_critical: float = 0.05,
    jsd_critical: float = 0.15,
    jsd_warning: float = 0.10,
    log_to_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Gera relatório completo de drift comparando baseline com dados de produção.

    Args:
        baseline:             Dicionário carregado por ``load_baseline()``.
        production_df:        DataFrame com dados de produção. Deve conter pelo menos
                              a coluna ``churn_probability``. Colunas de features são
                              opcionais — usadas para input drift se presentes.
        output_dir:           Diretório onde salvar o relatório JSON.
        psi_critical:         Threshold PSI crítico (padrão 0.20).
        psi_warning:          Threshold PSI de aviso (padrão 0.10).
        ks_critical:          KS statistic crítico (padrão 0.10).
        chi2_pvalue_critical: p-value chi2 crítico (padrão 0.05).
        jsd_critical:         JSD crítico (padrão 0.15).
        jsd_warning:          JSD de aviso (padrão 0.10).
        log_to_mlflow:        Se True, loga métricas no MLflow.
        mlflow_tracking_uri:  URI do MLflow se log_to_mlflow=True.

    Returns:
        Dicionário com o relatório completo de drift.
    """
    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_production_samples": len(production_df),
        "baseline_run_id": baseline.get("run_id", "unknown"),
        "numerical_drift": {},
        "categorical_drift": {},
        "prediction_drift": {},
        "alerts": [],
        "summary": {"status": "ok", "critical_count": 0, "warning_count": 0},
    }

    # ── 1. Prediction Distribution Drift (JSD) ────────────────────────────────
    pred_col = "churn_probability"
    if pred_col in production_df.columns and "prediction_distribution" in baseline:
        baseline_pred = baseline["prediction_distribution"]
        current_probs = production_df[pred_col].dropna().values

        jsd = compute_jsd_from_histograms(
            baseline_hist_counts=baseline_pred["hist_counts"],
            current_values=current_probs,
            hist_edges=baseline_pred["hist_edges"],
        )
        current_mean = float(np.mean(current_probs)) if len(current_probs) else 0.0
        current_churn_rate = float(
            (production_df.get("churn_prediction", pd.Series(dtype=int)) == 1).mean()
        )

        results["prediction_drift"] = {
            "jsd": jsd,
            "baseline_mean": baseline_pred.get("mean", 0.0),
            "current_mean": current_mean,
            "baseline_churn_rate": baseline_pred.get("churn_rate", 0.0),
            "current_churn_rate": current_churn_rate,
        }

        if jsd >= jsd_critical:
            results["alerts"].append(
                {"type": "CRITICAL", "metric": "JSD", "feature": "churn_probability", "value": jsd}
            )
            results["summary"]["critical_count"] += 1
        elif jsd >= jsd_warning:
            results["alerts"].append(
                {"type": "WARNING", "metric": "JSD", "feature": "churn_probability", "value": jsd}
            )
            results["summary"]["warning_count"] += 1

    # ── 2. Numerical Feature Drift (PSI + KS) ────────────────────────────────
    if "numerical_features" in baseline:
        for feature, stats in baseline["numerical_features"].items():
            if feature not in production_df.columns:
                continue

            prod_values = production_df[feature].dropna().values
            if len(prod_values) == 0:
                continue

            baseline_values = _reconstruct_numerical_from_stats(stats)

            psi = compute_psi(baseline_values, prod_values)
            ks_result = compute_ks(baseline_values, prod_values)

            results["numerical_drift"][feature] = {
                "psi": psi,
                "ks_statistic": ks_result["ks_statistic"],
                "ks_pvalue": ks_result["p_value"],
                "baseline_mean": stats.get("mean", 0.0),
                "current_mean": float(np.mean(prod_values)),
                "baseline_std": stats.get("std", 0.0),
                "current_std": float(np.std(prod_values)),
            }

            if psi >= psi_critical or ks_result["ks_statistic"] >= ks_critical:
                results["alerts"].append(
                    {
                        "type": "CRITICAL",
                        "metric": "PSI/KS",
                        "feature": feature,
                        "psi": psi,
                        "ks": ks_result["ks_statistic"],
                    }
                )
                results["summary"]["critical_count"] += 1
            elif psi >= psi_warning:
                results["alerts"].append(
                    {"type": "WARNING", "metric": "PSI", "feature": feature, "value": psi}
                )
                results["summary"]["warning_count"] += 1

    # ── 3. Categorical Feature Drift (Chi-Square) ─────────────────────────────
    if "categorical_features" in baseline:
        for feature, expected_freqs in baseline["categorical_features"].items():
            if feature not in production_df.columns:
                continue

            prod_col = production_df[feature].dropna()
            if len(prod_col) == 0:
                continue

            actual_counts = prod_col.value_counts().to_dict()
            chi2_result = compute_chi2(expected_freqs, actual_counts, len(prod_col))

            results["categorical_drift"][feature] = chi2_result

            if chi2_result["p_value"] < chi2_pvalue_critical:
                results["alerts"].append(
                    {
                        "type": "CRITICAL",
                        "metric": "Chi2",
                        "feature": feature,
                        "p_value": chi2_result["p_value"],
                    }
                )
                results["summary"]["critical_count"] += 1

    # ── 4. Status final ───────────────────────────────────────────────────────
    if results["summary"]["critical_count"] > 0:
        results["summary"]["status"] = "critical"
    elif results["summary"]["warning_count"] > 0:
        results["summary"]["status"] = "warning"

    # ── 5. Salvar relatório ───────────────────────────────────────────────────
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / f"drift_report_{date.today()}.json"
    with open(report_file, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    logger.info(f"Relatório de drift salvo em: {report_file}")

    # ── 6. Log no MLflow ──────────────────────────────────────────────────────
    if log_to_mlflow and _MLFLOW_AVAILABLE and mlflow_tracking_uri:
        _log_drift_to_mlflow(results, mlflow_tracking_uri)

    return results


def _reconstruct_numerical_from_stats(stats: dict) -> np.ndarray:
    """Reconstrói amostras sintéticas a partir das estatísticas do baseline.

    Usa os percentis do baseline para gerar uma distribuição aproximada.
    """
    percentiles = [
        stats.get("q10", stats.get("q25", stats.get("min", 0))),
        stats.get("q25", 0),
        stats.get("q50", stats.get("mean", 0)),
        stats.get("q75", 0),
        stats.get("q90", stats.get("max", 0)),
    ]
    mean = stats.get("mean", np.mean(percentiles))
    std = stats.get("std", np.std(percentiles) + 1e-9)

    rng = np.random.default_rng(42)
    return rng.normal(loc=mean, scale=std, size=1000)


def _log_drift_to_mlflow(results: dict, tracking_uri: str) -> None:
    """Loga métricas de drift no MLflow como run de monitoramento."""
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("monitoring")

        with mlflow.start_run(run_name=f"drift_check_{date.today()}"):
            if "prediction_drift" in results and results["prediction_drift"]:
                mlflow.log_metric("drift_jsd_churn_probability", results["prediction_drift"].get("jsd", 0))

            for feature, data in results.get("numerical_drift", {}).items():
                mlflow.log_metric(f"drift_psi_{feature}", data.get("psi", 0))
                mlflow.log_metric(f"drift_ks_{feature}", data.get("ks_statistic", 0))

            mlflow.log_metric("drift_critical_count", results["summary"]["critical_count"])
            mlflow.log_metric("drift_warning_count", results["summary"]["warning_count"])

        logger.info("Métricas de drift registradas no MLflow.")
    except Exception as exc:
        logger.warning(f"Erro ao logar drift no MLflow: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.monitoring.drift_detector",
        description="Detecta drift de dados comparando baseline de treino com produção.",
    )
    p.add_argument(
        "--production-csv",
        default=None,
        help=(
            "CSV com dados de produção. Colunas esperadas: churn_probability (obrigatória), "
            "mais quaisquer features para input drift. "
            "Se omitido, tenta ler da tabela prediction_logs no SQLite."
        ),
    )
    p.add_argument(
        "--db-url",
        default=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///mlflow.db"),
        help="URL do banco SQLite para ler PredictionLog (padrão: DATABASE_URL env).",
    )
    p.add_argument(
        "--window-days",
        type=int,
        default=7,
        help="Janela de dias de produção para análise (padrão: 7).",
    )
    p.add_argument(
        "--mlflow-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        help="URI do MLflow Tracking Server.",
    )
    p.add_argument(
        "--output-dir",
        default="monitoring/reports/drift",
        help="Diretório para salvar o relatório JSON.",
    )
    p.add_argument(
        "--log-to-mlflow",
        action="store_true",
        help="Se definido, loga as métricas de drift no MLflow.",
    )
    p.add_argument(
        "--config",
        default="configs/monitoring.yaml",
        help="Caminho para o arquivo de configuração YAML.",
    )
    return p


def _load_config(config_path: str) -> dict:
    try:
        import yaml

        with open(config_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _load_production_from_db(db_url: str, window_days: int) -> pd.DataFrame:
    """Carrega predições recentes da tabela prediction_logs via SQLite (síncrono)."""
    try:
        import sqlite3
        from urllib.parse import urlparse

        # Extrai o caminho do arquivo do SQLite URL
        db_path = db_url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
        if not Path(db_path).exists():
            logger.warning(f"Banco SQLite não encontrado: {db_path}")
            return pd.DataFrame()

        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            "SELECT * FROM prediction_logs WHERE predicted_at >= ?",
            conn,
            params=(cutoff.strftime("%Y-%m-%d %H:%M:%S"),),
        )
        conn.close()
        logger.info(f"Carregadas {len(df)} predições dos últimos {window_days} dias.")
        return df
    except Exception as exc:
        logger.warning(f"Erro ao carregar prediction_logs: {exc}")
        return pd.DataFrame()


def main() -> int:
    """Entrypoint CLI. Retorna 0 se OK, 1 se há alertas críticos."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    args = _build_parser().parse_args()
    cfg = _load_config(args.config)

    # Parâmetros de thresholds do YAML ou defaults
    drift_cfg = cfg.get("drift", {})
    baseline_cfg = cfg.get("baseline", {})
    mlflow_cfg = cfg.get("mlflow", {})

    # Carregar baseline
    baseline = load_baseline(
        mlflow_tracking_uri=args.mlflow_uri or mlflow_cfg.get("tracking_uri"),
        model_name=baseline_cfg.get("mlflow_model_name", "MLP_Focal_KFold_Script"),
        alias=baseline_cfg.get("mlflow_alias", "production"),
        artifact_name=baseline_cfg.get("artifact_name", "training_baseline.json"),
        fallback_path=baseline_cfg.get("fallback_path", "configs/monitoring_baseline.json"),
    )

    if baseline is None:
        logger.error("Baseline não encontrado. Impossível calcular drift.")
        return 2

    # Carregar dados de produção
    if args.production_csv:
        production_df = pd.read_csv(args.production_csv)
        logger.info(f"Dados de produção carregados de: {args.production_csv} ({len(production_df)} linhas)")
    else:
        production_df = _load_production_from_db(args.db_url, args.window_days)

    if len(production_df) == 0:
        logger.warning("Sem dados de produção para análise. Tente --production-csv.")
        return 0

    min_samples = drift_cfg.get("min_samples", 100)
    if len(production_df) < min_samples:
        logger.warning(
            f"Apenas {len(production_df)} amostras (mínimo: {min_samples}). "
            "Resultado pode não ser estatisticamente significativo."
        )

    # Gerar relatório
    report = generate_drift_report(
        baseline=baseline,
        production_df=production_df,
        output_dir=args.output_dir,
        psi_critical=drift_cfg.get("psi_critical", 0.20),
        psi_warning=drift_cfg.get("psi_warning", 0.10),
        ks_critical=drift_cfg.get("ks_critical", 0.10),
        chi2_pvalue_critical=drift_cfg.get("chi2_pvalue_critical", 0.05),
        jsd_critical=drift_cfg.get("jsd_critical", 0.15),
        jsd_warning=drift_cfg.get("jsd_warning", 0.10),
        log_to_mlflow=args.log_to_mlflow,
        mlflow_tracking_uri=args.mlflow_uri,
    )

    # Print resumo
    status = report["summary"]["status"]
    n_critical = report["summary"]["critical_count"]
    n_warning = report["summary"]["warning_count"]
    print(f"\n{'='*60}")
    print(f"DRIFT REPORT — {date.today()} — Status: {status.upper()}")
    print(f"Amostras de produção: {report['n_production_samples']}")
    print(f"Alertas críticos: {n_critical} | Avisos: {n_warning}")

    if report["alerts"]:
        print("\nAlertas:")
        for alert in report["alerts"]:
            prefix = "🔴 CRÍTICO" if alert["type"] == "CRITICAL" else "🟡 AVISO"
            feature = alert.get("feature", "")
            value = alert.get("value") or alert.get("psi") or alert.get("p_value", "")
            print(f"  {prefix} | {alert['metric']} | {feature} | valor={value:.4f}")

    print(f"{'='*60}\n")

    return 1 if n_critical > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
