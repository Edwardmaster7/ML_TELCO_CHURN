# src/ml_telco_churn/train.py
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from sklearn.model_selection import train_test_split

# Imports locais (depende de como o pacote foi construído, mas assumindo rodar via sys path)
from ml_telco_churn.config import CONFIG
from ml_telco_churn.data import load_and_merge_data
from ml_telco_churn.features import clean_data, get_preprocessor, prepare_target
from ml_telco_churn.model_nn import ChurnMLP

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Treina o modelo de Churn.")
    parser.add_argument("--customers", default="data/raw/churn_customers.csv")
    parser.add_argument("--services", default="data/raw/churn_services.csv")
    parser.add_argument("--contracts", default="data/raw/churn_contracts.csv")
    parser.add_argument("--epochs", type=int, default=10) # 10 p/ demo rapida, o nb usava 300
    args = parser.parse_args()

    # 1. Carregamento
    df_raw = load_and_merge_data(args.customers, args.services, args.contracts)
    df_clean = clean_data(df_raw)

    X = df_clean.drop(columns=[CONFIG.target_col, CONFIG.id_col], errors='ignore')
    y = prepare_target(df_clean[CONFIG.target_col])

    # 2. Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=CONFIG.random_state, stratify=y
    )

    # 3. Preprocessamento (Fit no Treino, Transform no Teste)
    preprocessor = get_preprocessor(
        cat_features=CONFIG.cat_features,
        num_features=CONFIG.num_features,
        available_columns=X_train_raw.columns.tolist()
    )
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)

    # Converte p/ Tensores
    X_train_t = torch.tensor(X_train_proc, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    # 4. Treinamento da Rede
    input_dim = X_train_t.shape[1]
    model = ChurnMLP(
        input_dim=input_dim,
        hidden_dims=CONFIG.mlp_hidden_dims,
        dropout_rate=CONFIG.mlp_dropout_rate
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG.learning_rate)

    logger.info("Iniciando treinamento PyTorch...")
    model.train()

    epochs_to_run = args.epochs if args.epochs != 10 else CONFIG.epochs
    for epoch in range(epochs_to_run):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    logger.info("Treinamento finalizado.")

    # 5. MLflow Tracking (SQLite backend)
    # Importante: Como usaremos http://127.0.0.1:5000, o MLflow UI precisa estar rodando
    # com o comando: mlflow ui --backend-store-uri sqlite:///mlflow.db
    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    mlflow.set_experiment(CONFIG.mlflow_experiment_name)

    with mlflow.start_run(run_name="mlp_pytorch_refactored"):
        # Log artifacts (A decisão principal da arquitetura)
        mlflow.sklearn.log_model(preprocessor, "preprocessor")
        mlflow.pytorch.log_model(model, "pytorch_model")

        mlflow.log_param("epochs", args.epochs)
        logger.info("Modelos registrados no MLflow.")

if __name__ == "__main__":
    main()