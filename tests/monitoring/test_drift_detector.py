"""Testes unitários para src/monitoring/drift_detector.py."""
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.monitoring.drift_detector import (
    compute_jsd_from_histograms,
    compute_ks,
    compute_psi,
)


class TestComputePsi:
    """Testes para a função compute_psi."""

    def test_distribuicoes_identicas_retorna_zero(self):
        """PSI deve ser ~0 quando as distribuições são idênticas."""
        data = np.random.default_rng(42).normal(0, 1, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.01, f"PSI esperado ≈ 0, obtido {psi}"

    def test_distribuicoes_muito_diferentes_psi_alto(self):
        """PSI deve ser alto quando as distribuições são muito diferentes."""
        rng = np.random.default_rng(0)
        expected = rng.normal(loc=0, scale=1, size=1000)
        actual = rng.normal(loc=5, scale=1, size=1000)
        psi = compute_psi(expected, actual)
        assert psi > 0.20, f"PSI esperado > 0.20 (crítico), obtido {psi}"

    def test_retorna_float(self):
        data = np.linspace(0, 1, 100)
        psi = compute_psi(data, data[:50])
        assert isinstance(psi, float)

    def test_array_vazio_retorna_zero(self):
        """compute_psi deve retornar 0.0 para arrays vazios sem explodir."""
        psi = compute_psi(np.array([]), np.array([]))
        assert psi == 0.0


class TestComputeKs:
    """Testes para a função compute_ks."""

    def test_mesma_distribuicao_pvalue_alto(self):
        rng = np.random.default_rng(7)
        data = rng.normal(0, 1, 500)
        result = compute_ks(data, data)
        assert result["p_value"] == 1.0 or result["p_value"] > 0.05

    def test_distribuicoes_diferentes_pvalue_baixo(self):
        rng = np.random.default_rng(7)
        d1 = rng.normal(0, 1, 500)
        d2 = rng.normal(10, 1, 500)
        result = compute_ks(d1, d2)
        assert result["p_value"] < 0.05
        assert "ks_statistic" in result

    def test_drift_detected_quando_pvalue_baixo(self):
        rng = np.random.default_rng(99)
        d1 = rng.uniform(0, 1, 300)
        d2 = rng.uniform(0.5, 1.5, 300)
        result = compute_ks(d1, d2)
        assert result["p_value"] < 0.05


class TestComputeJsd:
    """Testes para compute_jsd_from_histograms."""

    def test_histogramas_identicos_retorna_zero(self):
        baseline_counts = [10, 20, 30, 25, 15]
        edges = np.linspace(0, 1, 6)
        values = np.concatenate([
            np.full(10, 0.05),
            np.full(20, 0.25),
            np.full(30, 0.45),
            np.full(25, 0.65),
            np.full(15, 0.85),
        ])
        jsd = compute_jsd_from_histograms(baseline_counts, values, edges.tolist())
        assert jsd < 0.05, f"JSD esperado ≈ 0, obtido {jsd}"

    def test_histogramas_diferentes_jsd_alto(self):
        baseline_counts = [50, 50, 0, 0, 0]
        edges = np.linspace(0, 1, 6)
        values = np.full(100, 0.85)  # todos no último bin
        jsd = compute_jsd_from_histograms(baseline_counts, values, edges.tolist())
        assert jsd > 0.3, f"JSD esperado > 0.3, obtido {jsd}"

    def test_retorna_float_nao_nan(self):
        baseline_counts = [20, 30, 50]
        edges = [0.0, 0.33, 0.66, 1.0]
        values = np.linspace(0, 1, 100)
        jsd = compute_jsd_from_histograms(baseline_counts, values, edges)
        assert isinstance(jsd, float)
        assert not math.isnan(jsd)
