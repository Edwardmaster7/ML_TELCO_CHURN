import pytest
import pandas as pd
import pandera as pa
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.main import app

output_schema = pa.DataFrameSchema({
    "churn_probability": pa.Column(float, pa.Check.in_range(0.0, 1.0)),
    "churn_prediction": pa.Column(int, pa.Check.isin([0, 1]))
})

@patch("src.core.ml_service.MLService.predict_churn")
def test_api_output_respects_pandera_schema(mock_predict):
    """Testa se o retorno serializado da API REST é complacente ao schema numérico Pandas (Pandera).

    Exigência arquitetural rigorosa do Tech Challenge Fase 01 para garantir blindagem
    contra outputs defeituosos da rede neural.
    """
    mock_predict.return_value = {"churn_probability": 0.32, "churn_prediction": 0}

    with TestClient(app) as client:
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

        response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 200
    df_output = pd.DataFrame([response.json()])

    validated_df = output_schema.validate(df_output)
    assert not validated_df.empty
