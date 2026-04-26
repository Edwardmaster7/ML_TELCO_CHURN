# MLflow Metrics Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adicionar o cálculo e log das métricas completas (ROC-AUC, PR-AUC, F1, Precision, Recall) no bloco de avaliação de teste nos notebooks 03, 06 e 07 e retreiná-los para povoar o MLflow Registry.

**Architecture:** Editaremos diretamente os três notebooks via `NotebookEdit`, adicionando os cálculos de métrica baseados no `threshold` de `0.5` nas células finais que realizam a validação cega, e em seguida adicionaremos os comandos `mlflow.log_metrics()` ao bloco do experimento. Após a edição, executaremos os notebooks em background.

**Tech Stack:** PyTorch, Scikit-Learn, MLflow.

---

### Task 1: Atualizar o Notebook 03 (MLP_Vanilla_KFold)

**Files:**
- Modify: `notebooks/03_mlp_pytorch.ipynb`

- [ ] **Step 1: Adicionar o bloco de métricas completo no código fonte da última célula**

Edite a última célula do notebook `03_mlp_pytorch.ipynb` que contém a avaliação do `final_kfold_model` e o log no `MLP_Tuned_KFold`. Certifique-se de importar e calcular `f1_score`, `precision_score`, `recall_score`, `roc_auc_score` usando um threshold de `0.5` (que pode ser ajustado dependendo da otimização posterior).

```python
# A célula editada deve ficar similar a isso:

import mlflow
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

best_params_kf = study_kfold.best_params
best_hidden_dims_kf = [best_params_kf["hidden_size_1"], best_params_kf["hidden_size_2"]]
best_pos_w_kf = POS_WEIGHT if best_params_kf["use_class_weights"] else 1.0

# Instanciar o modelo final com os hiperparâmetros campeões do K-Fold
final_kfold_model = ChurnMLP(INPUT_DIM, best_hidden_dims_kf, best_params_kf["dropout_rate"]).to(device)

final_kfold_model, _ = train_mlp(
    model=final_kfold_model,
    X_tr_np=X_tr.values,
    y_tr_np=y_tr.values.astype(np.float32),
    X_val_np=X_val.values,
    y_val_np=y_val.values.astype(np.float32),
    pos_weight=best_pos_w_kf,
    lr=best_params_kf["lr"],
    weight_decay=best_params_kf["weight_decay"],
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    patience=PATIENCE
)

# ----------------- Avaliação Final no Test Set -----------------
final_kfold_model.eval()
with torch.no_grad():
    X_test_t = torch.FloatTensor(X_test.values).to(device)
    y_test_t = torch.FloatTensor(y_test.values.astype(np.float32)).to(device)
    
    test_logits_kf = final_kfold_model(X_test_t).squeeze()
    test_probs_kf = torch.sigmoid(test_logits_kf).cpu().numpy()
    
    test_preds_kf = (test_probs_kf >= 0.5).astype(int)
    
    test_pr_auc_kf = average_precision_score(y_test.values, test_probs_kf)
    test_roc_auc_kf = roc_auc_score(y_test.values, test_probs_kf)
    test_f1_kf = f1_score(y_test.values, test_preds_kf)
    test_precision_kf = precision_score(y_test.values, test_preds_kf, zero_division=0)
    test_recall_kf = recall_score(y_test.values, test_preds_kf)

print(f"Test PR-AUC do Modelo Vencedor (K-Fold): {test_pr_auc_kf:.4f}")

# ----------------- Registro Seguro no MLflow -----------------
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name="MLP_Tuned_KFold"):
    mlflow.log_params(best_params_kf)
    
    # NOVAS MÉTRICAS LOGADAS AQUI
    mlflow.log_metrics({
        "test_pr_auc": test_pr_auc_kf,
        "test_roc_auc": test_roc_auc_kf,
        "test_f1": test_f1_kf,
        "test_precision": test_precision_kf,
        "test_recall": test_recall_kf
    })
    
    # Tratamento contra o bug de RuntimeError e Mismatch Dtype do MLflow MPS/CPU
    final_kfold_model.cpu()
    
    # Gerando a signature exata
    input_sample = X_test.head(1).values.astype(np.float32)
    output_sample = final_kfold_model(torch.tensor(input_sample)).detach().numpy()
    
    from mlflow.models.signature import infer_signature
    sig_kfold = infer_signature(input_sample, output_sample)
    
    mlflow.pytorch.log_model(
        final_kfold_model,
        name="model",
        registered_model_name="MLP_Vanilla_KFold",
        signature=sig_kfold
    )
```

### Task 2: Atualizar o Notebook 06 (MLP_Focal_KFold e MLP_Focal_OneCycleLR)

**Files:**
- Modify: `notebooks/06_mlp_advanced_loss.ipynb`

*Note for agent:* The notebook 06 has two MLflow cells. One logs `MLP_Focal_OneCycleLR` (single split) and the other logs `MLP_Advanced_KFold` (K-Fold). Add the metrics array to BOTH evaluation blocks.

- [ ] **Step 1: Adicionar o bloco de métricas no Optuna Single Split (`MLP_Focal_OneCycleLR`)**
Edite a célula que faz a avaliação do Single Split. 
Importe `f1_score, precision_score, recall_score`. Calcule `test_roc_auc`, `test_f1`, `test_precision` e `test_recall` usando `test_preds = (test_probs >= 0.5).astype(int)`. No bloco MLflow, substitua `mlflow.log_metric("test_pr_auc", test_pr_auc)` pelo `mlflow.log_metrics({})` com todas as 5 chaves.

- [ ] **Step 2: Adicionar o bloco de métricas no K-Fold Split (`MLP_Advanced_KFold`)**
Edite a última célula do notebook 06. Calcule `test_roc_auc_kf`, `test_f1_kf`, `test_precision_kf` e `test_recall_kf` da mesma forma. Registre usando `mlflow.log_metrics({})`.

### Task 3: Atualizar o Notebook 07 (MLP_ResNet_KFold)

**Files:**
- Modify: `notebooks/07_mlp_resnet_embeddings.ipynb`

- [ ] **Step 1: Adicionar o bloco de métricas no K-Fold Final**
Edite a última célula de orquestração do Notebook 07.

Importe `f1_score, precision_score, recall_score, roc_auc_score`.
Abaixo de onde `test_pr_auc` é calculado, insira:

```python
    test_preds = (test_probs >= 0.5).astype(int)
    test_roc_auc = roc_auc_score(ds_test_final.y.numpy(), test_probs)
    test_f1 = f1_score(ds_test_final.y.numpy(), test_preds)
    test_precision = precision_score(ds_test_final.y.numpy(), test_preds, zero_division=0)
    test_recall = recall_score(ds_test_final.y.numpy(), test_preds)
```

E no MLflow run (`with mlflow.start_run(run_name="MLP_ResNet_KFold"):`) troque o `log_metric` unitário por:

```python
    mlflow.log_metrics({
        "test_pr_auc": test_pr_auc,
        "test_roc_auc": test_roc_auc,
        "test_f1": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall
    })
```

### Task 4: Executar os 3 Notebooks em Background

**Files:**
- Modify: (None, execution only)

- [ ] **Step 1: Rodar `03_mlp_pytorch.ipynb`**
Use a tool Bash `jupyter nbconvert --to notebook --execute --inplace notebooks/03_mlp_pytorch.ipynb`
- [ ] **Step 2: Rodar `06_mlp_advanced_loss.ipynb`**
Use a tool Bash `jupyter nbconvert --to notebook --execute --inplace notebooks/06_mlp_advanced_loss.ipynb`
- [ ] **Step 3: Rodar `07_mlp_resnet_embeddings.ipynb`**
Use a tool Bash `jupyter nbconvert --to notebook --execute --inplace notebooks/07_mlp_resnet_embeddings.ipynb`

*Note for agent:* These processes take a very long time. You MUST run them sequentially (one after another) and wait for each to complete successfully to ensure the SQLite MLflow database is not corrupted by concurrent locks. Do not use background execution, block and wait.