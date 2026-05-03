"""Serviço MLOps.

Responsável por fazer cache (Singleton) dos modelos de produção e coordenar as inferências
pela rede MLP_Focal_KFold do Pytorch em conjunto com pipeline baseline de Data Science.
"""
import pandas as pd
import logging
import torch
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.tracking

from datetime import datetime, timezone
from typing import Optional, Any
from src.features.pipeline import clean_data

logger = logging.getLogger(__name__)

class MLService:
    """Singleton class contendo pipelines de processamento e arquitetura da rede.

    Attributes:
        preprocessor (object): objeto Scikit-Learn fitado baixado do Tracking Server.
        model (torch.nn.Module): objeto torch baixado do Tracking Server.
        device (torch.device): device local disponível ('cuda' ou 'cpu').
        model_name (str): Nome do modelo registrado no MLflow.
        model_version (str): Versão do modelo carregado (número de versão do Registry).
        run_id (str): Run ID do MLflow associado ao modelo em produção.
        loaded_at (datetime): Timestamp UTC de quando os artefatos foram carregados.
    """
    _instance = None
    preprocessor: Optional[Any] = None
    model: Optional[torch.nn.Module] = None
    device: Optional[torch.device] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    run_id: Optional[str] = None
    loaded_at: Optional[datetime] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLService, cls).__new__(cls)
            cls._instance.preprocessor = None
            cls._instance.model = None
            cls._instance.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
            cls._instance.model_name = None
            cls._instance.model_version = None
            cls._instance.run_id = None
            cls._instance.loaded_at = None
        return cls._instance

    def load_model_artifacts(self, model_name: str, stage_or_alias: str = "production", tracking_uri: str = "sqlite:///mlflow.db"):
        """Conecta ao MLflow Database e carrega artefatos do Model Registry em memória.

        Args:
            model_name (str): Nome do modelo registrado no MLflow (ex: 'MLP_Focal_KFold').
            stage_or_alias (str): Alias (ex: 'production' ou 'latest').
            tracking_uri (str, optional): Caminho local / URL do Tracking Server.

        Raises:
            RuntimeError: Quando há erro na integridade dos artefatos contidos no mlartifacts ou sqlite.
        """
        logger.info(f"Conectando ao MLflow em {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)

        try:
            # Novo formato de URI do Model Registry para Aliases
            base_uri = f"models:/{model_name}@{stage_or_alias}"

            logger.info(f"Buscando metadados do Model Registry para '{model_name}' (Alias: {stage_or_alias})...")
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

            # Nova abordagem de busca do MLflow >2.9.0 (Aliases em vez de Stages)
            model_version_info = client.get_model_version_by_alias(
                name=model_name,
                alias=stage_or_alias
            )

            run_id = model_version_info.run_id
            logger.info(f"Modelo localizado com sucesso! RUN_ID resolvido: {run_id}")

            logger.info("Carregando preprocessor acoplado da mesma Run...")
            preprocessor_uri = f"runs:/{run_id}/preprocessor"
            self.preprocessor = mlflow.sklearn.load_model(preprocessor_uri)

            logger.info(f"Carregando PyTorch model do Model Registry ({base_uri})...")
            self.model = mlflow.pytorch.load_model(base_uri)
            self.model.to(self.device)
            self.model.eval()

            # Armazena metadados para /health e logging estruturado
            self.model_name = model_name
            self.model_version = model_version_info.version
            self.run_id = run_id
            self.loaded_at = datetime.now(timezone.utc)

            logger.info("Modelos carregados com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao carregar modelos via Registry: {e}")
            raise RuntimeError(f"Falha ao iniciar MLService via Registry: {e}")

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
            # O PyTorch pode retornar shape [1, 1] ou flat [1].
            # Acessamos corretamente para extrair o valor float da GPU/CPU
            probability = torch.sigmoid(logits).item()

        prediction = 1 if probability >= 0.5 else 0

        return {
            "churn_probability": float(probability),
            "churn_prediction": int(prediction)
        }
