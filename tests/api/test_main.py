import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

with patch("src.api.ml_service.MLService.load_model_artifacts") as mock_load:
    from src.api.main import app

@pytest.fixture
def client():
    """Fixture geradora do client de teste do FastAPI em contexto lifespan."""
    with TestClient(app) as test_client:
        yield test_client

def test_health_check(client):
    """Testa o endpoint de readiness que atesta se a rede neural subiu na API."""
    with patch("src.api.main.ml_service.model", True), patch("src.api.main.ml_service.preprocessor", True):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json().get("model_loaded") is True

@patch("src.api.ml_service.MLService.predict_churn")
def test_predict_endpoint_success(mock_predict, client):
    """Smoke test da rota de predição injetando payload JSON perfeito."""
    mock_predict.return_value = {"churn_probability": 0.85, "churn_prediction": 1}

    payload = {
        "customerID": "1234-ABC",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "29.85"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["churn_probability"] == 0.85
    assert data["churn_prediction"] == 1

def test_predict_endpoint_validation_error(client):
    """Testa restrição imposta pelo Pydantic respondendo com erro HTTP 422."""
    payload = {
        "customerID": "1234-ABC",
        "gender": "Female"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "detail" in response.json()
