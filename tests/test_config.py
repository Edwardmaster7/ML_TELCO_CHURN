import pytest
from src.config import CONFIG

def test_config_best_params_updated():
    # Verifica se os hiperparâmetros refletem o Trial 5 do KFold do nb 06
    assert CONFIG.best_params['dropout_rate'] == 0.385
    assert CONFIG.best_params['hidden_size_1'] == 32
    assert CONFIG.best_params['hidden_size_2'] == 16
    assert CONFIG.best_params['focal_gamma'] == 3.150
    assert CONFIG.best_params['focal_alpha'] == 0.727
