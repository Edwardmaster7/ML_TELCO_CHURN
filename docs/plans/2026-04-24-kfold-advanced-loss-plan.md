# Advanced Loss & K-Fold Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adicionar validação cruzada K-Fold no Optuna do notebook 06 (`06_mlp_advanced_loss.ipynb`) para combater o Generalization Gap observado nas estratégias de Focal Loss e OneCycleLR.

**Architecture:** Sem alterar o código existente, adicionaremos uma nova seção ao final do notebook. Essa seção criará uma função `objective_kfold` para o Optuna, que usa `StratifiedKFold` (n_splits=3) no conjunto `X_train` inteiro, avaliando o desempenho do modelo na média da validação PR-AUC. O melhor modelo será retreinado e gravado seguramente no MLflow.

**Tech Stack:** PyTorch, Scikit-Learn (StratifiedKFold), Optuna, MLflow.

---

### Task 1: Seção de K-Fold Cross-Validation

**Files:**
- Modify: `notebooks/06_mlp_advanced_loss.ipynb` (Apenas adicionar novas células ao final do arquivo). *Worker hint: Use `NotebookEdit` tool with `edit_mode="insert"` and an omitted or large `cell_id`.*

- [ ] **Step 1: Adicionar célula de documentação (Markdown)**

Adicione uma nova célula markdown ao final do notebook para documentar a transição lógica.

```markdown
---
## Correção Metodológica: Validação Cruzada (K-Fold) na Arquitetura Avançada

Assim como ocorreu no MLP Vanilla, a arquitetura avançada sofria de *Hyperparameter Overfitting* por testar repetidas vezes o mesmo conjunto de validação (`X_val`). Para aferirmos o real poder da `FocalLoss` combinada com o `AdamW` e `OneCycleLR`, precisamos submeter o Optuna a um `StratifiedKFold` sobre o conjunto de treino inteiro. O objetivo passa a ser a maximização da **média** de PR-AUC nos Folds.
```

- [ ] **Step 2: Implementar a função `objective_kfold` e executar o Optuna**

Adicione uma célula de código que define a otimização com K-Fold e reaproveita as instâncias já criadas no notebook.

```python
from sklearn.model_selection import StratifiedKFold

# Constantes K-Fold
N_SPLITS = 3
N_TRIALS_KFOLD = 15

def objective_kfold(trial):
    """
    Função objetivo do Optuna utilizando Validação Cruzada K-Fold para a Arquitetura Focal.
    O Optuna tentará otimizar os parâmetros que maximizam a média de PR-AUC dos 3 folds.
    """
    # 1. Sugestão de Hiperparâmetros (Restritos para mitigar Overfitting)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    hidden_size_1 = trial.suggest_categorical("hidden_size_1", [32, 64])
    hidden_size_2 = trial.suggest_categorical("hidden_size_2", [16, 32])
    focal_gamma = trial.suggest_float("focal_gamma", 0.0, 5.0)
    focal_alpha = trial.suggest_float("focal_alpha", 0.1, 0.9)
    max_lr = trial.suggest_float("max_lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 5e-3, log=True)

    hidden_dims = [hidden_size_1, hidden_size_2]
    
    # 2. Configurar o K-Fold no conjunto de treino original
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []
    
    X_train_np = X_train.values
    y_train_np = y_train.values.astype(np.float32)
    
    # 3. Iterar sobre cada Fold
    for train_idx, val_idx in skf.split(X_train_np, y_train_np):
        X_fold_tr, y_fold_tr = X_train_np[train_idx], y_train_np[train_idx]
        X_fold_val, y_fold_val = X_train_np[val_idx], y_train_np[val_idx]
        
        # Instanciar nova rede a cada fold
        model_fold = ChurnMLP(INPUT_DIM, hidden_dims, dropout_rate).to(device)
        
        # Treinar usando train_mlp_advanced
        model_fold, history = train_mlp_advanced(
            model=model_fold,
            X_tr_np=X_fold_tr,
            y_tr_np=y_fold_tr,
            X_val_np=X_fold_val,
            y_val_np=y_fold_val,
            loss_type="focal",
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            max_lr=max_lr,
            weight_decay=weight_decay,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            patience=PATIENCE
        )
        
        # Coletar pico de PR-AUC do fold
        hist_df = pd.DataFrame(history)
        best_fold_pr_auc = hist_df['val_pr_auc'].max()
        fold_scores.append(best_fold_pr_auc)
        
    return np.mean(fold_scores)

# Executar o Estudo
study_kfold = optuna.create_study(direction="maximize", study_name="focal_loss_kfold")
study_kfold.optimize(objective_kfold, n_trials=N_TRIALS_KFOLD)

print(f"Melhor PR-AUC Médio (K-Fold): {study_kfold.best_value:.4f}")
print("Melhores Hiperparâmetros:", study_kfold.best_params)
```

### Task 2: Retreino e Registro Seguro do Modelo Vencedor

**Files:**
- Modify: `notebooks/06_mlp_advanced_loss.ipynb`

- [ ] **Step 1: Adicionar célula de Avaliação no Teste Cego e MLflow Logging**

Adicione esta última célula de código. Ela deve instanciar a rede com os melhores parâmetros, rodar `train_mlp_advanced` e fazer o registro limpo no MLflow movendo o PyTorch pra CPU.

```python
best_params_kf = study_kfold.best_params
best_hidden_dims_kf = [best_params_kf["hidden_size_1"], best_params_kf["hidden_size_2"]]

# Instanciar modelo campeão do K-Fold
final_kfold_model = ChurnMLP(INPUT_DIM, best_hidden_dims_kf, best_params_kf["dropout_rate"]).to(device)

# Treinamento simulando hold-out com X_tr e X_val para preservar early stopping original
final_kfold_model, _ = train_mlp_advanced(
    model=final_kfold_model,
    X_tr_np=X_tr.values,
    y_tr_np=y_tr.values.astype(np.float32),
    X_val_np=X_val.values,
    y_val_np=y_val.values.astype(np.float32),
    loss_type="focal",
    focal_gamma=best_params_kf["focal_gamma"],
    focal_alpha=best_params_kf["focal_alpha"],
    max_lr=best_params_kf["max_lr"],
    weight_decay=best_params_kf["weight_decay"],
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    patience=PATIENCE
)

# Avaliação rigorosa no Test Set (Hold-out Cego)
final_kfold_model.eval()
with torch.no_grad():
    X_test_t = torch.FloatTensor(X_test.values).to(device)
    y_test_t = torch.FloatTensor(y_test.values.astype(np.float32)).to(device)
    
    test_logits_kf = final_kfold_model(X_test_t).squeeze()
    test_probs_kf = torch.sigmoid(test_logits_kf).cpu().numpy()
    
    test_pr_auc_kf = average_precision_score(y_test.values, test_probs_kf)

print(f"Test PR-AUC do Modelo Vencedor Avançado (K-Fold): {test_pr_auc_kf:.4f}")

# Registro MLOps no MLflow
import mlflow
from mlflow.models.signature import infer_signature

mlflow.set_tracking_uri("sqlite:///../mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="MLP_Advanced_KFold"):
    mlflow.log_params(best_params_kf)
    mlflow.log_metric("test_pr_auc", test_pr_auc_kf)
    
    # Move para CPU para evitar Tensor Error RuntimeError('Tensor for argument input is on cpu but expected on mps')
    final_kfold_model.cpu()
    
    input_sample = X_test.head(1).values.astype(np.float32)
    output_sample = final_kfold_model(torch.tensor(input_sample)).detach().numpy()
    sig_kfold = infer_signature(input_sample, output_sample)
    
    mlflow.pytorch.log_model(
        final_kfold_model,
        name="model",
        registered_model_name="MLP_Focal_KFold",
        signature=sig_kfold
    )
```