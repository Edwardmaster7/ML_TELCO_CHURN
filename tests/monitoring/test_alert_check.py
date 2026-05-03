"""Testes unitários para src/monitoring/alert_check.py."""
import json
import tempfile
from pathlib import Path

import pytest

from src.monitoring.alert_check import (
    check_drift_report,
    check_performance_report,
    generate_alert_report,
)


class TestCheckDriftReport:
    """Testes para check_drift_report."""

    def _make_drift_report(self, psi=0.10, jsd=0.05, alerts=None):
        return {
            "prediction_drift": {"jsd": jsd},
            "numerical_drift": {
                "tenure": {"psi": psi},
            },
            "alerts": alerts or [],
        }

    def test_sem_drift_sem_alertas(self):
        report = self._make_drift_report(psi=0.05, jsd=0.03)
        alerts = check_drift_report(report, psi_critical=0.20, jsd_critical=0.15)
        critical = [a for a in alerts if a["type"] == "CRITICAL"]
        assert len(critical) == 0

    def test_psi_critico_gera_alerta(self):
        report = self._make_drift_report(psi=0.25, jsd=0.03)
        alerts = check_drift_report(report, psi_critical=0.20, jsd_critical=0.15)
        critical = [a for a in alerts if a["type"] == "CRITICAL"]
        assert len(critical) >= 1
        features = [a.get("feature") for a in critical]
        assert "tenure" in features

    def test_jsd_critico_gera_alerta(self):
        report = self._make_drift_report(psi=0.05, jsd=0.20)
        alerts = check_drift_report(report, psi_critical=0.20, jsd_critical=0.15)
        critical = [a for a in alerts if a["type"] == "CRITICAL"]
        assert len(critical) >= 1

    def test_alertas_preexistentes_sao_incluidos(self):
        existing = [{"type": "WARNING", "metric": "chi2", "feature": "contract"}]
        report = self._make_drift_report(psi=0.05, jsd=0.05, alerts=existing)
        alerts = check_drift_report(report)
        assert any(a.get("metric") == "chi2" for a in alerts)


class TestCheckPerformanceReport:
    """Testes para check_performance_report."""

    def test_sem_degradacao_sem_alertas(self):
        report = {"overall_metrics": {"pr_auc": 0.75}, "alerts": []}
        alerts = check_performance_report(report, pr_auc_critical=0.60)
        assert alerts == []

    def test_pr_auc_critico_gera_alerta(self):
        report = {"overall_metrics": {"pr_auc": 0.55}, "alerts": []}
        alerts = check_performance_report(report, pr_auc_critical=0.60)
        critical = [a for a in alerts if a["type"] == "CRITICAL"]
        assert len(critical) >= 1
        assert critical[0]["metric"] == "pr_auc"

    def test_relatorio_sem_metricas_nao_explode(self):
        report = {"overall_metrics": {}, "alerts": []}
        alerts = check_performance_report(report, pr_auc_critical=0.60)
        assert isinstance(alerts, list)


class TestGenerateAlertReport:
    """Testes para generate_alert_report."""

    def test_sem_alertas_status_ok(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = generate_alert_report([], output_dir=tmpdir)
            assert report["status"] == "ok"
            assert report["critical_count"] == 0
            assert report["warning_count"] == 0

    def test_alertas_criticos_status_critical(self):
        alerts = [{"type": "CRITICAL", "metric": "PSI", "feature": "tenure", "value": 0.30}]
        with tempfile.TemporaryDirectory() as tmpdir:
            report = generate_alert_report(alerts, output_dir=tmpdir)
            assert report["status"] == "critical"
            assert report["critical_count"] == 1

    def test_apenas_warnings_status_warning(self):
        alerts = [{"type": "WARNING", "metric": "pr_auc", "value": 0.63}]
        with tempfile.TemporaryDirectory() as tmpdir:
            report = generate_alert_report(alerts, output_dir=tmpdir)
            assert report["status"] == "warning"

    def test_arquivo_json_criado(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_alert_report([], output_dir=tmpdir)
            json_files = list(Path(tmpdir).glob("*.json"))
            assert len(json_files) == 1
            # verifica que o JSON é válido
            with open(json_files[0]) as fh:
                data = json.load(fh)
            assert "generated_at" in data
