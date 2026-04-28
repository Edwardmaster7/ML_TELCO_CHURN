import pytest
from pydantic import ValidationError
from src.api.schemas import ChurnPredictionRequest, ChurnPredictionResponse

def test_valid_request_schema():
    """Testa se o schema request aceita dados válidos rigorosamente.

    Valida se um payload completo com todos os campos originais do dataset
    é aceito pelo Pydantic, convertendo tipos quando apropriado.

    Asserts:
        O ID do cliente é mapeado corretamente.
        TotalCharges é convertido de string para float.
    """
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
    request = ChurnPredictionRequest(**payload)
    assert request.customerID == "1234-ABC"
    assert isinstance(request.TotalCharges, float)
    assert request.TotalCharges == 29.85

def test_empty_total_charges_coercion():
    """Testa coerção de string vazia em TotalCharges para 0.0.

    No dataset original da IBM, contas recentes podem ter TotalCharges
    como ' ', o que quebra o parser numérico.

    Asserts:
        A conversão de um espaço em branco resulta no float 0.0.
    """
    payload = {
        "customerID": "1234-ABC",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 0,
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
        "TotalCharges": " "
    }
    request = ChurnPredictionRequest(**payload)
    assert request.TotalCharges == 0.0

def test_invalid_literal_field():
    """Testa restrição Pydantic barrando dados fora do dicionário de dados literal.

    Asserts:
        Lança ValidationError quando uma feature categórica recebe
        um valor não listado em seus domínios Literals.
    """
    payload = {
        "customerID": "1234-ABC",
        "gender": "Alien",  # Invalid
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
    with pytest.raises(ValidationError):
        ChurnPredictionRequest(**payload)

def test_response_schema():
    """Testa modelo de validação da saída da API."""
    response = ChurnPredictionResponse(churn_probability=0.85, churn_prediction=1)
    assert response.churn_probability == 0.85
    assert response.churn_prediction == 1