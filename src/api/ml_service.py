"""Serviço MLOps.

Responsável por fazer cache (Singleton) dos modelos de produção e coordenar as inferências
pela rede MLP_Focal_KFold do Pytorch em conjunto com pipeline baseline de Data Science.
"""
import os
import pandas as pd
import numpy as np
import torch
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import logging

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
