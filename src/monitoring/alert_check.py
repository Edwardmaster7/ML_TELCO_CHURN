"""Verificador de alertas de monitoramento — avalia relatórios de drift e performance.

Lê relatórios JSON gerados pelos módulos ``drift_detector`` e ``performance_monitor``
e verifica thresholds. Sai com código 0 (OK), 1 (CRÍTICO) ou 2 (ERRO).

Uso (CLI cross-platform):
    python -m src.monitoring.alert_check
    python -m src.monitoring.alert_check --drift-report monitoring/reports/drift/drift_report_2026-05-02.json
    python -m src.monitoring.alert_check --perf-report monitoring/reports/performance/perf_report_2026-05-02.json
    python -m src.monitoring.alert_check --health-url http://localhost:8000/health

Use em CI/CD (pipeline falha se alertas críticos):
    python -m src.monitoring.alert_check || echo "ALERTA: monitoramento requer atenção"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de carregamento
# ─────────────────────────────────────────────────────────────────────────────

def load_latest_report(reports_dir: str) -> dict[str, Any] | None:
    """Carrega o relatório JSON mais recente de um diretório."""
    path = Path(reports_dir)
    if not path.exists():
        return None

    json_files = sorted(path.glob("*.json"), reverse=True)
    if not json_files:
        return None

    try:
        with open(json_files[0], encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning(f"Erro ao carregar {json_files[0]}: {exc}")
        return None


def load_config(config_path: str) -> dict:
    try:
        import yaml

        with open(config_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Verificações individuais
# ─────────────────────────────────────────────────────────────────────────────

def check_drift_report(
    report: dict[str, Any],
    psi_critical: float = 0.20,
    jsd_critical: float = 0.15,
) -> list[dict[str, Any]]:
    """Verifica thresholds no relatório de drift.

    Args:
        report:       Relatório JSON gerado por ``drift_detector``.
        psi_critical: Threshold PSI crítico.
        jsd_critical: Threshold JSD crítico.

    Returns:
        Lista de alertas gerados (dicionários com type, metric, feature, value).
    """
    alerts = []

    # Alertas já calculados no relatório
    for alert in report.get("alerts", []):
        alerts.append(alert)

    # Verificações diretas nos dados numéricos (override / confirmação)
    for feature, data in report.get("numerical_drift", {}).items():
        psi = data.get("psi", 0.0)
        if psi >= psi_critical and not any(
            a.get("feature") == feature and a.get("type") == "CRITICAL" for a in alerts
        ):
            alerts.append(
                {"type": "CRITICAL", "metric": "PSI", "feature": feature, "value": psi}
            )

    pred_drift = report.get("prediction_drift", {})
    jsd = pred_drift.get("jsd", 0.0)
    if jsd >= jsd_critical and not any(
        a.get("feature") == "churn_probability" and a.get("type") == "CRITICAL"
        for a in alerts
    ):
        alerts.append(
            {"type": "CRITICAL", "metric": "JSD", "feature": "churn_probability", "value": jsd}
        )

    return alerts


def check_performance_report(
    report: dict[str, Any],
    pr_auc_critical: float = 0.60,
) -> list[dict[str, Any]]:
    """Verifica thresholds no relatório de performance.

    Args:
        report:          Relatório JSON gerado por ``performance_monitor``.
        pr_auc_critical: Threshold PR-AUC crítico.

    Returns:
        Lista de alertas.
    """
    alerts = list(report.get("alerts", []))

    pr_auc = report.get("overall_metrics", {}).get("pr_auc", 1.0)
    if pr_auc > 0 and pr_auc < pr_auc_critical and not any(
        a.get("metric") == "pr_auc" and a.get("type") == "CRITICAL" for a in alerts
    ):
        alerts.append(
            {"type": "CRITICAL", "metric": "pr_auc", "value": pr_auc, "threshold": pr_auc_critical}
        )

    return alerts


def check_api_health(health_url: str) -> list[dict[str, Any]]:
    """Verifica o endpoint /health da API.

    Args:
        health_url: URL completa do endpoint (ex: ``http://localhost:8000/health``).

    Returns:
        Lista de alertas (vazia se saudável).
    """
    alerts = []
    try:
        import urllib.request

        with urllib.request.urlopen(health_url, timeout=5) as resp:
            data = json.loads(resp.read())
            if not data.get("model_loaded", False):
                alerts.append(
                    {
                        "type": "CRITICAL",
                        "metric": "model_loaded",
                        "feature": "api_health",
                        "value": 0,
                    }
                )
    except Exception as exc:
        alerts.append(
            {
                "type": "CRITICAL",
                "metric": "api_unreachable",
                "feature": "health_endpoint",
                "value": str(exc),
            }
        )
    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# Geração do relatório de alertas
# ─────────────────────────────────────────────────────────────────────────────

def generate_alert_report(
    all_alerts: list[dict[str, Any]],
    output_dir: str = "monitoring/reports/alerts",
) -> dict[str, Any]:
    """Consolida alertas em um relatório JSON e salva em disco.

    Args:
        all_alerts: Lista consolidada de alertas.
        output_dir: Diretório de saída.

    Returns:
        Dicionário do relatório de alertas.
    """
    critical = [a for a in all_alerts if a.get("type") == "CRITICAL"]
    warnings = [a for a in all_alerts if a.get("type") == "WARNING"]

    report = {
        "generated_at": str(date.today()),
        "status": "critical" if critical else ("warning" if warnings else "ok"),
        "critical_count": len(critical),
        "warning_count": len(warnings),
        "alerts": all_alerts,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / f"alert_report_{date.today()}.json"
    with open(report_file, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    logger.info(f"Relatório de alertas salvo em: {report_file}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.monitoring.alert_check",
        description="Verifica alertas de monitoramento a partir de relatórios JSON.",
    )
    p.add_argument(
        "--drift-report",
        default=None,
        help="Caminho para o JSON de drift. Se omitido, usa o mais recente em monitoring/reports/drift/.",
    )
    p.add_argument(
        "--perf-report",
        default=None,
        help="Caminho para o JSON de performance. Se omitido, usa o mais recente em monitoring/reports/performance/.",
    )
    p.add_argument(
        "--health-url",
        default=None,
        help="URL do endpoint /health da API (ex: http://localhost:8000/health).",
    )
    p.add_argument(
        "--output-dir",
        default="monitoring/reports/alerts",
        help="Diretório para salvar o relatório de alertas.",
    )
    p.add_argument(
        "--config",
        default="configs/monitoring.yaml",
        help="Caminho para o arquivo de configuração YAML.",
    )
    p.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Se definido, exit code 1 em qualquer alerta (crítico ou aviso).",
    )
    return p


def main() -> int:
    """Entrypoint CLI.

    Returns:
        0 se nenhum alerta crítico, 1 se há alertas críticos.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    args = _build_parser().parse_args()
    cfg = load_config(args.config)

    drift_cfg = cfg.get("drift", {})
    perf_cfg = cfg.get("performance", {})
    output_cfg = cfg.get("output", {})

    all_alerts: list[dict[str, Any]] = []

    # ── 1. Verificar relatório de drift ───────────────────────────────────────
    drift_report = None
    if args.drift_report:
        try:
            with open(args.drift_report, encoding="utf-8") as fh:
                drift_report = json.load(fh)
        except Exception as exc:
            logger.warning(f"Não foi possível carregar drift report: {exc}")
    else:
        drift_dir = output_cfg.get("drift_reports_dir", "monitoring/reports/drift")
        drift_report = load_latest_report(drift_dir)

    if drift_report:
        drift_alerts = check_drift_report(
            drift_report,
            psi_critical=drift_cfg.get("psi_critical", 0.20),
            jsd_critical=drift_cfg.get("jsd_critical", 0.15),
        )
        all_alerts.extend(drift_alerts)
    else:
        logger.info("Nenhum relatório de drift encontrado. Ignorando verificação de drift.")

    # ── 2. Verificar relatório de performance ─────────────────────────────────
    perf_report = None
    if args.perf_report:
        try:
            with open(args.perf_report, encoding="utf-8") as fh:
                perf_report = json.load(fh)
        except Exception as exc:
            logger.warning(f"Não foi possível carregar perf report: {exc}")
    else:
        perf_dir = output_cfg.get("performance_reports_dir", "monitoring/reports/performance")
        perf_report = load_latest_report(perf_dir)

    if perf_report:
        perf_alerts = check_performance_report(
            perf_report,
            pr_auc_critical=perf_cfg.get("pr_auc_critical", 0.60),
        )
        all_alerts.extend(perf_alerts)
    else:
        logger.info("Nenhum relatório de performance encontrado.")

    # ── 3. Verificar saúde da API ─────────────────────────────────────────────
    if args.health_url:
        health_alerts = check_api_health(args.health_url)
        all_alerts.extend(health_alerts)

    # ── 4. Consolidar e salvar relatório ──────────────────────────────────────
    alert_dir = output_cfg.get("alert_reports_dir", args.output_dir)
    report = generate_alert_report(all_alerts, output_dir=alert_dir)

    # ── 5. Print final ────────────────────────────────────────────────────────
    status = report["status"]
    n_critical = report["critical_count"]
    n_warning = report["warning_count"]

    print(f"\n{'='*60}")
    print(f"ALERT CHECK — {date.today()} — Status: {status.upper()}")
    print(f"Alertas críticos: {n_critical} | Avisos: {n_warning}")

    if all_alerts:
        print("\nDetalhes:")
        for alert in all_alerts:
            prefix = "🔴 CRÍTICO" if alert.get("type") == "CRITICAL" else "🟡 AVISO"
            metric = alert.get("metric", "?")
            feature = alert.get("feature", "")
            value = alert.get("value", "")
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            print(f"  {prefix} | {metric} | {feature} | {value_str}")
    else:
        print("  ✅ Nenhum alerta ativo.")

    print(f"{'='*60}\n")

    if args.fail_on_warning:
        return 1 if (n_critical + n_warning) > 0 else 0
    return 1 if n_critical > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
