# API de Inferência Churn Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar a API FastAPI de inferência em real-time que carrega artefatos (PyTorch e Scikit-Learn) do MLflow Model Registry, com docstrings formato Google.

**Architecture:** Padrão Controller-Service. FastAPI lidará com rotas e validação via Pydantic (`schemas.py` e `main.py`). A lógica do negócio e carregamento MLflow em Singleton via Lifespan reside em `ml_service.py`. A observabilidade será tratada por um `middlewares.py`. Todo o fluxo obedece à engenharia data-centric existente em `src/features/pipeline.py`.

**Tech Stack:** FastAPI, Pydantic, MLflow, PyTorch, Pandas, Pytest, Pandera.

---

### Task 1: Criar Schemas Pydantic de Entrada e Saída

**Files:**

- Create: `src/api/schemas.py`
- Create: `tests/api/test_schemas.py`

- [x] **Step 1: Write the failing test**

```python
# tests/api/test_schemas.py
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
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_schemas.py -v`
Expected: FAIL with ModuleNotFoundError for `src.api.schemas`

- [x] **Step 3: Write minimal implementation**

```python
# src/api/schemas.py
"""Contratos de Dados e validação usando Pydantic."""
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Union

class ChurnPredictionRequest(BaseModel):
    """Schema de requisição contendo as features originais do dataset Telco.
  
    Valida e tipa os atributos vindos do JSON de request para impedir Data Leakage
    e bugs por entrada malformada, com literais estritos para categóricas.
  
    Attributes:
        customerID (str): Identificador único do usuário.
        gender (Literal["Male", "Female"]): Gênero do cliente.
        SeniorCitizen (Literal[0, 1]): Indica se é idoso.
        Partner (Literal["Yes", "No"]): Possui parceiro.
        Dependents (Literal["Yes", "No"]): Possui dependentes.
        tenure (int): Meses de permanência na empresa.
        PhoneService (Literal["Yes", "No"]): Assina serviço de telefone.
        MultipleLines (Literal["Yes", "No", "No phone service"]): Múltiplas linhas telefônicas.
        InternetService (Literal["DSL", "Fiber optic", "No"]): Tipo de internet.
        OnlineSecurity (Literal["Yes", "No", "No internet service"]): Possui segurança online.
        OnlineBackup (Literal["Yes", "No", "No internet service"]): Possui backup online.
        DeviceProtection (Literal["Yes", "No", "No internet service"]): Possui proteção de aparelho.
        TechSupport (Literal["Yes", "No", "No internet service"]): Possui suporte técnico.
        StreamingTV (Literal["Yes", "No", "No internet service"]): Assina TV a cabo.
        StreamingMovies (Literal["Yes", "No", "No internet service"]): Assina filmes.
        Contract (Literal["Month-to-month", "One year", "Two year"]): Tipo de contrato.
        PaperlessBilling (Literal["Yes", "No"]): Fatura digital.
        PaymentMethod (Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]): Método de pagamento.
        MonthlyCharges (float): Cobrança mensal.
        TotalCharges (Union[float, str]): Cobrança total, que passará por coerção numérica.
    """
    customerID: str
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0.0)
    TotalCharges: Union[float, str]

    @field_validator('TotalCharges')
    @classmethod
    def coerce_total_charges(cls, v: Union[float, str]) -> float:
        """Coerce campos de TotalCharges vazios para 0.0 seguindo regra de EDA.

        Args:
            v (Union[float, str]): Valor de TotalCharges recebido no payload.

        Returns:
            float: O valor numérico formatado corretamente, garantindo float.

        Raises:
            ValueError: Se a string não puder ser convertida numérico real.
        """
        if isinstance(v, str):
            v_stripped = v.strip()
            if not v_stripped:
                return 0.0
            try:
                return float(v_stripped)
            except ValueError:
                raise ValueError("TotalCharges must be a valid float string or empty.")
        return float(v)

class ChurnPredictionResponse(BaseModel):
    """Schema de resposta representando a inferência do modelo campeão.
  
    Attributes:
        churn_probability (float): A probabilidade (0.0 a 1.0) calculada pela rede neural via sigmoid.
        churn_prediction (Literal[0, 1]): A classe consolidada (1 = sim, 0 = não) baseada em um limiar estrito de 0.5.
    """
    churn_probability: float = Field(ge=0.0, le=1.0)
    churn_prediction: Literal[0, 1]
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/api/test_schemas.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/api/schemas.py tests/api/test_schemas.py
git commit -m "feat: cria pydantic schemas para a request e response da api de inferencia"
```

### Task 2: Implementar Structured Logging Middleware

**Files:**

- Create: `src/api/middlewares.py`
- Create: `tests/api/test_middlewares.py`

- [x] **Step 1: Write the failing test**

```python
# tests/api/test_middlewares.py
import pytest
import logging
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api.middlewares import LoggingMiddleware

def test_logging_middleware_records_latency(caplog):
    """Testa se o middleware injeta com sucesso um log de monitoramento.
  
    Verifica se a resposta de uma rota qualquer dispara o interceptor
    que salva logs contendo os termos method, path, status e latency.
  
    Args:
        caplog (LogCaptureFixture): Objeto nativo do pytest para capturar emissões do logging.
    """
    app = FastAPI()
    app.add_middleware(LoggingMiddleware)
  
    @app.get("/test")
    def test_route():
        return {"msg": "ok"}
      
    client = TestClient(app)
  
    with caplog.at_level(logging.INFO):
        response = client.get("/test")
      
    assert response.status_code == 200
  
    logs = [record.message for record in caplog.records]
    log_found = any("method=GET" in log and "path=/test" in log and "status=200" in log and "latency=" in log for log in logs)
    assert log_found, f"Expected structured log not found. Logs: {logs}"
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_middlewares.py -v`
Expected: FAIL with ModuleNotFoundError for `src.api.middlewares`

- [x] **Step 3: Write minimal implementation**

```python
# src/api/middlewares.py
"""Módulo de Middlewares ASGI para observabilidade."""
import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    logger.addHandler(handler)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware FastAPI para mensuração de tráfego HTTP.
  
    Interceptor que avalia o tempo de cada requisição e lança um log
    estruturado no output sem uso de commands de print().
    """
  
    async def dispatch(self, request: Request, call_next):
        """Processa requisição e emite log formatado em pares key=value.
      
        Args:
            request (Request): O objeto de requisição do FastAPI.
            call_next (Callable): Função delegate para seguir na chain de middlewares/rotas.
          
        Returns:
            Response: Resposta da requisição original, inalterada.
        """
        start_time = time.time()
      
        response = await call_next(request)
      
        process_time = time.time() - start_time
        latency_ms = round(process_time * 1000, 2)
      
        log_msg = f"method={request.method} path={request.url.path} status={response.status_code} latency={latency_ms}ms"
        logger.info(log_msg)
      
        return response
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/api/test_middlewares.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/api/middlewares.py tests/api/test_middlewares.py
git commit -m "feat: adiciona logging middleware estruturado para latencia"
```

### Task 3: Criar Mock do Model Registry e Testes para ML Service

**Files:**

- Create: `tests/api/test_ml_service.py`
- Create: `src/api/ml_service.py`

- [x] **Step 1: Write the failing test**

```python
# tests/api/test_ml_service.py
import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from src.api.ml_service import MLService

@pytest.fixture
def mock_mlflow():
    """Mock da dependência física do Registry MLFlow."""
    with patch("src.api.ml_service.mlflow") as mock_mlf:
        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = np.array([[0.5, 1.2, 0.0, 1.0]])
        mock_mlf.sklearn.load_model.return_value = mock_preprocessor
      
        mock_torch_model = MagicMock()
        mock_torch_model.return_value = torch.tensor([[1.5]]) 
        mock_torch_model.eval = MagicMock()
        mock_mlf.pytorch.load_model.return_value = mock_torch_model
      
        yield mock_mlf

@pytest.fixture
def mock_clean_data():
    """Mock da função data-centric."""
    with patch("src.api.ml_service.clean_data") as mock_clean:
        mock_clean.side_effect = lambda df: df
        yield mock_clean

def test_load_model_artifacts(mock_mlflow):
    """Testa o carregamento de URI do Registry pro singleton."""
    service = MLService()
    service.load_model_artifacts(run_id="fake_run", tracking_uri="sqlite:///fake.db")
  
    mock_mlflow.set_tracking_uri.assert_called_with("sqlite:///fake.db")
    mock_mlflow.sklearn.load_model.assert_called_with("runs:/fake_run/preprocessor")
    mock_mlflow.pytorch.load_model.assert_called_with("runs:/fake_run/model")
  
    assert service.preprocessor is not None
    assert service.model is not None

def test_predict_churn(mock_mlflow, mock_clean_data):
    """Valida o roteamento e predição correta do serviço acoplado."""
    service = MLService()
    service.load_model_artifacts(run_id="fake_run")
  
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
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_ml_service.py -v`
Expected: FAIL with ModuleNotFoundError for `src.api.ml_service`

- [x] **Step 3: Write minimal implementation**

```python
# src/api/ml_service.py
"""Serviço MLOps.

Responsável por fazer cache (Singleton) dos modelos de produção e coordenar as inferências
pela rede MLP_Focal_KFold do Pytorch em conjunto com pipeline baseline de Data Science.
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features.pipeline import clean_data

logger = logging.getLogger(__name__)

class MLService:
    """Singleton class contendo pipelines de processamento e arquitetura da rede.
  
    Attributes:
        preprocessor (object): objeto Scikit-Learn fitado baixado do Tracking Server.
        model (torch.nn.Module): objeto torch baixado do Tracking Server.
        device (torch.device): device local disponível ('cuda' ou 'cpu').
    """
    _instance = None
  
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLService, cls).__new__(cls)
            cls._instance.preprocessor = None
            cls._instance.model = None
            cls._instance.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return cls._instance

    def load_model_artifacts(self, run_id: str, tracking_uri: str = "sqlite:///mlflow.db"):
        """Conecta ao MLflow Database e carrega artefatos do run em memória.
      
        Args:
            run_id (str): UUID do MLFlow Run contendo o registro do modelo de Prod.
            tracking_uri (str, optional): Caminho local / URL do Tracking Server.
          
        Raises:
            RuntimeError: Quando há erro na integridade dos artefatos contidos no mlartifacts ou sqlite.
        """
        logger.info(f"Conectando ao MLflow em {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
      
        try:
            logger.info(f"Carregando preprocessor da run {run_id}...")
            preprocessor_uri = f"runs:/{run_id}/preprocessor"
            self.preprocessor = mlflow.sklearn.load_model(preprocessor_uri)
          
            logger.info(f"Carregando PyTorch model da run {run_id}...")
            model_uri = f"runs:/{run_id}/model"
            self.model = mlflow.pytorch.load_model(model_uri)
            self.model.to(self.device)
            self.model.eval()
          
            logger.info("Modelos carregados com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao carregar modelos: {e}")
            raise RuntimeError(f"Falha ao iniciar MLService: {e}")

    def predict_churn(self, data: dict) -> dict:
        """Realiza Pipeline completa transformando payload unitário numérico em predição PyTorch.
      
        Aplica a função clean_data (data-centric engineer), executa o scaler do preprocessor 
        e finaliza na passagem feed-forward da rede neural finalizando na ativação sigmoid.
      
        Args:
            data (dict): Dict tipado proveniente do schema Pydantic.
          
        Returns:
            dict: Dicionário contendo "churn_probability" (float) e "churn_prediction" (int).
          
        Raises:
            RuntimeError: Se for chamado sem a invocação prévia (lifespans) de load_model_artifacts().
        """
        if self.preprocessor is None or self.model is None:
            raise RuntimeError("Modelos não carregados. Execute load_model_artifacts primeiro.")
          
        df = pd.DataFrame([data])
        df_clean = clean_data(df)
        features_array = self.preprocessor.transform(df_clean)
        features_tensor = torch.tensor(features_array, dtype=torch.float32).to(self.device)
      
        with torch.no_grad():
            logits = self.model(features_tensor)
            probability = torch.sigmoid(logits).cpu().numpy()[0][0]
          
        prediction = 1 if probability >= 0.5 else 0
      
        return {
            "churn_probability": float(probability),
            "churn_prediction": int(prediction)
        }
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/api/test_ml_service.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/api/ml_service.py tests/api/test_ml_service.py
git commit -m "feat: implementa ml_service para carregar artefatos do mlflow e executar inferencia pyrtorch"
```

### Task 4: Implementar o FastAPI App e Endpoints (Router)

**Files:**

- Create: `src/api/main.py`
- Create: `tests/api/test_main.py`

- [x] **Step 1: Write the failing test**

```python
# tests/api/test_main.py
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
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_loaded" in response.json()

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
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_main.py -v`
Expected: FAIL with ModuleNotFoundError for `src.api.main`

- [x] **Step 3: Write minimal implementation**

```python
# src/api/main.py
"""Ponto de entrada (Router) principal do FastAPI.

Responsável pelas rotas HTTP e inicialização assíncrona (Lifespan) do processo.
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.api.schemas import ChurnPredictionRequest, ChurnPredictionResponse
from src.api.ml_service import MLService
from src.api.middlewares import LoggingMiddleware

logger = logging.getLogger(__name__)
ml_service = MLService()

RUN_ID = os.getenv("MODEL_RUN_ID", "1")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Contexto de inicialização Cold-Start para FastAPI conectando na plataforma MLFlow.
  
    Garante que os artefatos de treinamento serão hidratados em memória da RAM antes 
    que qualquer requisição externa tente usar o endpoint de predição.
  
    Args:
        app (FastAPI): Instância da aplicação principal.
    """
    logger.info("Iniciando FastAPI e carregando modelos no MLService...")
    try:
        ml_service.load_model_artifacts(run_id=RUN_ID)
    except Exception as e:
        logger.error(f"Erro no Lifespan: {e}")
        pass
    yield
    logger.info("Desligando API.")

app = FastAPI(
    title="API de Previsão de Churn",
    description="Tech Challenge - ML Engineering API para inferência do modelo PyTorch (MLP_Focal_KFold)",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(LoggingMiddleware)

@app.get("/health")
def health_check():
    """Verifica a saúde básica do contêiner e prontidão dos artefatos scikit/torch.
  
    Returns:
        dict: Dicionário de status da API.
      
    Raises:
        HTTPException: Erro 503 Service Unavailable caso o modelo não tenha sido carregado.
    """
    model_loaded = ml_service.model is not None and ml_service.preprocessor is not None
    status = "ok" if model_loaded else "degraded"
  
    if not model_loaded:
         raise HTTPException(status_code=503, detail="Model Artifacts not loaded in MLService.")
       
    return {"status": status, "model_loaded": model_loaded}

@app.post("/predict", response_model=ChurnPredictionResponse)
def predict(request: ChurnPredictionRequest):
    """Invoca as pipelines de predição do modelo campeão (Pytorch Focal Loss).
  
    Args:
        request (ChurnPredictionRequest): Payload pydantic contendo atributos em string, inteiro ou float da operadora.
      
    Returns:
        ChurnPredictionResponse: Resposta serializada garantindo que um número real e a predição chegam no cliente.
      
    Raises:
        HTTPException: 503 Se ocorrer Runtime de Modelos offline.
        HTTPException: 500 Se ocorrer uma quebra silenciosa e imprevista do pandas.
    """
    try:
        result = ml_service.predict_churn(request.model_dump())
        return result
    except RuntimeError as re:
        logger.error(f"RuntimeError na predição: {re}")
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.error(f"Erro inesperado na predição: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during inference.")
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/api/test_main.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/api/main.py tests/api/test_main.py
git commit -m "feat: cria o router principal do fastapi com endpoints de predict e health"
```

### Task 5: Pandera Schema Test e Makefile Run

**Files:**

- Create: `tests/api/test_schema_pandera.py`
- Create/Update: `Makefile`

- [x] **Step 1: Write the failing test**

```python
# tests/api/test_schema_pandera.py
import pytest
import pandas as pd
import pandera as pa
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.api.main import app

output_schema = pa.DataFrameSchema({
    "churn_probability": pa.Column(float, pa.Check.in_range(0.0, 1.0)),
    "churn_prediction": pa.Column(int, pa.Check.isin([0, 1]))
})

@patch("src.api.ml_service.MLService.predict_churn")
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
      
        response = client.post("/predict", json=payload)
      
    assert response.status_code == 200
    df_output = pd.DataFrame([response.json()])
  
    validated_df = output_schema.validate(df_output)
    assert not validated_df.empty
```

- [x] **Step 2: Run test to verify it passes**

Run: `pytest tests/api/test_schema_pandera.py -v`
Expected: PASS

- [x] **Step 3: Update Makefile (se existir) / Instruções de Boot**

Create or update a `Makefile` at root (se não existir, crie um básico).

```makefile
# Makefile
.PHONY: test run

test:
	pytest tests/api/ -v

run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

- [x] **Step 4: Commit**

```bash
git add tests/api/test_schema_pandera.py Makefile
git commit -m "test: adiciona validacao de saida com pandera e makefile para execução do fastapi"
```
