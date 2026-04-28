import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from src.core.ml_service import MLService

@pytest.fixture
def mock_mlflow():
    """Mock da dependência física do Registry MLFlow."""
    with patch("src.core.ml_service.mlflow") as mock_mlf:
        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = np.array([[0.5, 1.2, 0.0, 1.0]])
        mock_mlf.sklearn.load_model.return_value = mock_preprocessor

        mock_torch_model = MagicMock()
        mock_torch_model.return_value = torch.tensor([[1.5]])
        mock_torch_model.eval = MagicMock()
        mock_mlf.pytorch.load_model.return_value = mock_torch_model

        # Configura o mock do client
        mock_client_instance = MagicMock()
        mock_version_info = MagicMock()
        mock_version_info.run_id = "fake_run"
        mock_client_instance.get_model_version_by_alias.return_value = mock_version_info
        mock_mlf.tracking.MlflowClient.return_value = mock_client_instance

        yield mock_mlf

@pytest.fixture
def mock_clean_data():
    """Mock da função data-centric."""
    with patch("src.core.ml_service.clean_data") as mock_clean:
        mock_clean.side_effect = lambda df: df
        yield mock_clean

def test_load_model_artifacts(mock_mlflow):
    """Testa o carregamento de URI do Registry pro singleton."""
    service = MLService()
    service.load_model_artifacts(model_name="FakeModel", stage_or_alias="production", tracking_uri="sqlite:///fake.db")

    mock_mlflow.set_tracking_uri.assert_called_with("sqlite:///fake.db")
    mock_mlflow.sklearn.load_model.assert_called_with("runs:/fake_run/preprocessor")
    mock_mlflow.pytorch.load_model.assert_called_with("models:/FakeModel@production")

    assert service.preprocessor is not None
    assert service.model is not None

def test_predict_churn(mock_mlflow, mock_clean_data):
    """Valida o roteamento e predição correta do serviço acoplado."""
    service = MLService()
    service.load_model_artifacts(model_name="FakeModel", stage_or_alias="production")

    payload = {
        "customerID": "123",
        "gender": "Female",
        "TotalCharges": 100.0
    }

    response = service.predict_churn(payload)

    assert "churn_probability" in response
    assert "churn_prediction" in response
    assert isinstance(response["churn_probability"], float)
    assert response["churn_prediction"] == 1
