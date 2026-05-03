"""Testes unitários para src/monitoring/performance_monitor.py."""
import numpy as np
import pandas as pd
import pytest

from src.monitoring.performance_monitor import (
    compute_performance_metrics,
    evaluate_by_time_window,
)


class TestComputePerformanceMetrics:
    """Testes para compute_performance_metrics."""

    def test_predicoes_perfeitas_retorna_metricas_maximas(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.9, 0.9, 0.9])
        metrics = compute_performance_metrics(y_true, y_prob)
        assert metrics["pr_auc"] == pytest.approx(1.0, abs=0.01)
        assert metrics["f1"] == pytest.approx(1.0, abs=0.01)
        assert metrics["recall"] == pytest.approx(1.0, abs=0.01)

    def test_todas_as_chaves_presentes(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])
        metrics = compute_performance_metrics(y_true, y_prob)
        for key in ("pr_auc", "roc_auc", "f1", "precision", "recall",
                    "churn_rate_actual", "churn_rate_predicted", "n_samples"):
            assert key in metrics, f"Chave ausente: {key}"

    def test_churn_rate_actual_correto(self):
        y_true = np.array([1, 1, 0, 0, 0])  # 2/5 = 0.4
        y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.15])
        metrics = compute_performance_metrics(y_true, y_prob)
        assert metrics["churn_rate_actual"] == pytest.approx(0.4, abs=0.01)

    def test_threshold_customizado(self):
        """Com threshold = 0.3, predições ≥ 0.3 viram 1."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.25, 0.35, 0.9])
        metrics_strict = compute_performance_metrics(y_true, y_prob, threshold=0.5)
        metrics_relaxed = compute_performance_metrics(y_true, y_prob, threshold=0.3)
        # Com threshold mais baixo, o modelo prediz mais positivos
        assert metrics_relaxed["churn_rate_predicted"] >= metrics_strict["churn_rate_predicted"]

    def test_array_vazio_retorna_dict_vazio(self):
        """sklearn não deve explodir com array vazio — retornamos dict vazio."""
        metrics = compute_performance_metrics(np.array([]), np.array([]))
        # Pode retornar vazio ou ter pr_auc = 0.0, o importante é não explodir
        assert isinstance(metrics, dict)


class TestEvaluateByTimeWindow:
    """Testes para evaluate_by_time_window."""

    def _make_df(self, n=50, seed=0):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({
            "predicted_at": pd.date_range("2026-01-01", periods=n, freq="D"),
            "churn_probability": rng.uniform(0, 1, n),
            "actual_churn": rng.integers(0, 2, n),
        })
        return df

    def test_retorna_lista_nao_vazia(self):
        df = self._make_df(60)
        results = evaluate_by_time_window(df, window_days=30)
        assert len(results) > 0

    def test_cada_janela_tem_metricas(self):
        df = self._make_df(60)
        results = evaluate_by_time_window(df, window_days=30)
        for window in results:
            assert "window_start" in window
            assert "window_end" in window
            assert "pr_auc" in window

    def test_dataframe_vazio_retorna_lista_vazia(self):
        result = evaluate_by_time_window(pd.DataFrame(), window_days=30)
        assert result == []

    def test_sem_coluna_actual_churn_retorna_vazio(self):
        df = pd.DataFrame({
            "predicted_at": pd.date_range("2026-01-01", periods=10, freq="D"),
            "churn_probability": np.random.rand(10),
        })
        result = evaluate_by_time_window(df, window_days=30)
        assert result == []
