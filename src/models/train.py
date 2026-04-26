# src/models/train.py
import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Imports locais para features/config (o caminho foi ajustado conforme Clean Architecture)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config import CONFIG
from src.data.loader import load_and_merge_data
from src.features.pipeline import clean_data, get_preprocessor, prepare_target
from src.models.architectures import ChurnMLP, FocalLoss
from src.models.trainer import train_focal_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Desativa log de variáveis de ambiente do MLFlow para limpar output
    os.environ["MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING"] = "false"
    logging.getLogger("mlflow").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Treina o modelo de Churn (Focal Loss + K-Fold).")
    parser.add_argument("--customers", default="notebooks/data/raw/churn_customers.csv")
    parser.add_argument("--services", default="notebooks/data/raw/churn_services.csv")
    parser.add_argument("--contracts", default="notebooks/data/raw/churn_contracts.csv")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    # 1. Carregamento e Feature Engineering Base (Mantendo a compatibilidade do script)
    # Obs: Ajuste nos paths pro script rodar da raiz do projeto
    df_raw = load_and_merge_data(args.customers, args.services, args.contracts)
    df_clean = clean_data(df_raw)

    X = df_clean.drop(columns=[CONFIG.target_col, CONFIG.id_col], errors='ignore')
    y = prepare_target(df_clean[CONFIG.target_col])

    hidden_dims = [CONFIG.best_params['hidden_size_1'], CONFIG.best_params['hidden_size_2']]
    dropout_rate = CONFIG.best_params['dropout_rate']
    focal_gamma = CONFIG.best_params['focal_gamma']
    focal_alpha = CONFIG.best_params['focal_alpha']
    max_lr = CONFIG.best_params['max_lr']
    weight_decay = CONFIG.best_params['weight_decay']

    epochs_to_run = args.epochs
    batch_size = 64
    patience = 20
    n_splits = 3

    # Holdout Cego
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=CONFIG.random_state, stratify=y
    )

    # Preprocessamento (Fit no Treino Inteiro p/ uso no Teste e Deploy)
    preprocessor = get_preprocessor(
        cat_features=CONFIG.cat_features,
        num_features=CONFIG.num_features,
        available_columns=X_train_raw.columns.tolist()
    )

    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)

    # 2. Configurar o K-Fold no conjunto de treino
    logger.info("Iniciando Validação Cruzada K-Fold para estabilização de hiperparâmetros...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CONFIG.random_state)

    X_train_np = X_train_proc
    y_train_np = y_train.astype(np.float32)

    fold_scores = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    input_dim = X_train_np.shape[1]

    # Vamos retreinar um último modelo no conjunto todo (100% de X_train),
    # mas antes simulamos o K-Fold para report
    for train_idx, val_idx in skf.split(X_train_np, y_train_np):
        X_fold_tr = torch.tensor(X_train_np[train_idx], dtype=torch.float32)
        y_fold_tr = torch.tensor(y_train_np[train_idx], dtype=torch.float32)
        X_fold_val = torch.tensor(X_train_np[val_idx], dtype=torch.float32)
        y_fold_val = torch.tensor(y_train_np[val_idx], dtype=torch.float32)

        model_fold = ChurnMLP(input_dim, hidden_dims, dropout_rate).to(device)
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma).to(device)

        model_fold, history = train_focal_model(
            model=model_fold,
            criterion=criterion,
            X_tr_t=X_fold_tr,
            y_tr_t=y_fold_tr,
            X_val_t=X_fold_val,
            y_val_t=y_fold_val,
            device=device,
            max_lr=max_lr,
            weight_decay=weight_decay,
            n_epochs=epochs_to_run,
            batch_size=batch_size,
            patience=patience
        )

        hist_df = pd.DataFrame(history)
        if not hist_df.empty and 'val_pr_auc' in hist_df.columns:
            best_fold_pr_auc = hist_df['val_pr_auc'].max()
            fold_scores.append(best_fold_pr_auc)

    mean_kfold_pr_auc = np.mean(fold_scores) if fold_scores else 0.0
    logger.info(f"K-Fold concluído! PR-AUC Médio de Validação: {mean_kfold_pr_auc:.4f}")

    # 3. Retreino do Modelo Campeão em todo o X_train
    # Para o early stopping aqui simularemos um hold-out interno de 20% do X_train
    logger.info("Retreinando modelo final com hold-out simulado para Early Stopping...")
    X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
        X_train_proc, y_train_np, test_size=0.2, random_state=CONFIG.random_state, stratify=y_train_np
    )

    X_tr_t = torch.tensor(X_tr_sub, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_sub, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_sub, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_sub, dtype=torch.float32)

    final_model = ChurnMLP(input_dim, hidden_dims, dropout_rate).to(device)
    final_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma).to(device)

    final_model, _ = train_focal_model(
        model=final_model,
        criterion=final_criterion,
        X_tr_t=X_tr_t,
        y_tr_t=y_tr_t,
        X_val_t=X_val_t,
        y_val_t=y_val_t,
        device=device,
        max_lr=max_lr,
        weight_decay=weight_decay,
        n_epochs=epochs_to_run,
        batch_size=batch_size,
        patience=patience
    )

    # 4. Avaliação Rigorosa no Conjunto de Teste (Cego)
    logger.info("Avaliando modelo final no conjunto de Teste Cego...")
    final_model.eval()

    X_test_t = torch.tensor(X_test_proc, dtype=torch.float32).to(device)
    y_test_np = y_test.astype(np.float32)

    with torch.no_grad():
        test_logits = final_model(X_test_t)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_preds = (test_probs >= 0.5).astype(int)

    test_pr_auc = average_precision_score(y_test_np, test_probs)
    test_roc_auc = roc_auc_score(y_test_np, test_probs)
    test_f1 = f1_score(y_test_np, test_preds)
    test_precision = precision_score(y_test_np, test_preds, zero_division=0)
    test_recall = recall_score(y_test_np, test_preds)

    logger.info(f"Métricas no Teste -> PR-AUC: {test_pr_auc:.4f} | ROC-AUC: {test_roc_auc:.4f} | F1: {test_f1:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f}")

    # 5. MLflow Tracking (SQLite backend)
    # Comando exigido para iniciar ui: mlflow ui --backend-store-uri sqlite:///mlflow.db
    # Usando MLFLOW_TRACKING_URI da variável de ambiente ou do CONFIG para evitar erro do SQLite com artifact_uri
    tracking_uri = CONFIG.mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)

    pip_reqs = ["scikit-learn==1.3.2", "skops==0.8.0"] # Ajustado aos reqs do projeto (substitua conforme requirements.txt se necessário)

    with mlflow.start_run(run_name="MLP_Focal_KFold_Production"):
        # Loga os hiperparâmetros definidos do K-Fold
        mlflow.log_params(CONFIG.best_params)

        # Loga métricas
        metrics_dict = {
            "test_pr_auc": test_pr_auc,
            "test_roc_auc": test_roc_auc,
            "test_f1": test_f1,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "mean_kfold_pr_auc": mean_kfold_pr_auc
        }
        mlflow.log_metrics(metrics_dict)

        # Log do preprocessor como exigido pela restrição arquitetural da API
        mlflow.sklearn.log_model(
            sk_model=preprocessor,
            name="preprocessor",
            serialization_format="skops",
            skops_trusted_types=["numpy.dtype", "numpy.float64"],
            pip_requirements=pip_reqs
        )

        # Inferência de Signature e model log
        final_model.cpu()  # Evita erro de Tensores MPS/CPU ao salvar signature
        input_sample = X_test_proc[0:5].astype(np.float32)
        out_sig = final_model(torch.tensor(input_sample)).detach().numpy()

        from mlflow.models.signature import infer_signature
        sig = infer_signature(input_sample, out_sig)

        mlflow.pytorch.log_model(
            pytorch_model=final_model,
            name="model",
            registered_model_name="MLP_Focal_KFold_Script",
            signature=sig,
            pip_requirements=["torch==2.1.0"] # Ajustado aos reqs (substitua conforme o venv se diferir muito)
        )

        logger.info("Modelo (Focal Loss + K-Fold) e métricas registradas no MLflow.")

if __name__ == "__main__":
    main()
