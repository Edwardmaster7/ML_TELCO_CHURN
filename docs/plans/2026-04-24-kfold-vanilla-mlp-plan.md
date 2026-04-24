# Correção Metodológica: K-Fold no MLP Vanilla Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provar a superioridade metodológica da validação cruzada K-Fold sobre um único split de validação no framework Optuna, utilizando a arquitetura original (MLP Vanilla) no notebook 03.

**Architecture:** Adicionaremos uma nova seção ao final do notebook `03_mlp_pytorch.ipynb` sem modificar nenhuma célula anterior. Criaremos um novo objeto de estudo do Optuna que instanciará a rede 3 vezes por iteração usando `StratifiedKFold`, retornando a média do PR-AUC. 

**Tech Stack:** PyTorch, Scikit-Learn (StratifiedKFold), Optuna.

---

### Task 1: Adicionar Seção de K-Fold Cross-Validation

**Files:**
- Modify: `notebooks/03_mlp_pytorch.ipynb` (Apenas adicionar novas células ao final do arquivo JSON). *Dica para workers: Use o `NotebookEdit` com `edit_mode="insert"` para adicionar células ao final de forma segura.*

- [ ] **Step 1: Adicionar célula de documentação (Markdown)**

Adicione uma nova célula markdown ao final do notebook explicando a transição.

```markdown
---
## Correção Metodológica: Validação Cruzada (K-Fold) no Optuna

Diagnosticamos um **Generalization Gap** nos testes anteriores. O Optuna estava memorizando o conjunto de validação estático (`X_val`), atingindo picos irrealistas que caíam bruscamente no teste cego.

Para provar se o teto de performance do MLP Vanilla decorre da limitação estrutural da rede ou apenas de um *hyperparameter overfitting*, aplicaremos a **Validação Cruzada K-Fold (StratifiedKFold)** diretamente no loop do Optuna, forçando-o a maximizar a **média** de PR-AUC de múltiplos splits randômicos do conjunto de treino.
```

- [ ] **Step 2: Implementar a função `objective_kfold` e executar o Optuna**

Adicione uma nova célula de código contendo o K-Fold, reaproveitando as constantes, a função `train_mlp` e a classe `ChurnMLP` originais que já existem no escopo do notebook.

```python
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

# Definindo parâmetros do K-Fold
N_SPLITS = 3
N_TRIALS_KFOLD = 15

def objective_kfold(trial):
    """
    Função objetivo do Optuna que utiliza Validação Cruzada K-Fold.
    O Optuna tentará otimizar os parâmetros que maximizam a média de PR-AUC dos 3 folds.
    """
    # 1. Sugestão de Hiperparâmetros
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    hidden_size_1 = trial.suggest_categorical("hidden_size_1", [32, 64, 128])
    hidden_size_2 = trial.suggest_categorical("hidden_size_2", [16, 32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    use_class_weights = trial.suggest_categorical("use_class_weights", [True, False])
    
    hidden_dims = [hidden_size_1, hidden_size_2]
    pos_w = POS_WEIGHT if use_class_weights else 1.0
    
    # 2. Configurar o K-Fold no conjunto de treino original
    # Usaremos X_train e y_train que contêm 80% do dataset original (antes do split de X_tr e X_val)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    fold_scores = []
    
    # Convertendo para arrays NumPy para o indexamento do KFold e compatibilidade do train_mlp
    X_train_np = X_train.values
    y_train_np = y_train.values.astype(np.float32)
    
    # 3. Iterar sobre cada Fold
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np)):
        X_fold_tr = X_train_np[train_idx]
        y_fold_tr = y_train_np[train_idx]
        
        X_fold_val = X_train_np[val_idx]
        y_fold_val = y_train_np[val_idx]
        
        # O model deve ser reiniciado a cada fold para não vazar pesos
        model_fold = ChurnMLP(INPUT_DIM, hidden_dims, dropout_rate).to(device)
        
        # Treinar o fold usando a função train_mlp original já definida neste notebook
        model_fold, history = train_mlp(
            model=model_fold,
            X_tr_np=X_fold_tr,
            y_tr_np=y_fold_tr,
            X_val_np=X_fold_val,
            y_val_np=y_fold_val,
            pos_weight=pos_w,
            lr=lr,
            weight_decay=weight_decay,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            patience=PATIENCE
        )
        
        # Obter o melhor val_pr_auc alcançado no early stopping deste fold
        hist_df = pd.DataFrame(history)
        best_fold_pr_auc = hist_df['val_pr_auc'].max()
        fold_scores.append(best_fold_pr_auc)
        
    # O valor final deste Trial é a média do PR-AUC em todos os folds
    return np.mean(fold_scores)

# Executar o Estudo
study_kfold = optuna.create_study(direction="maximize", study_name="mlp_vanilla_kfold")
study_kfold.optimize(objective_kfold, n_trials=N_TRIALS_KFOLD)

print(f"Melhor PR-AUC Médio (K-Fold): {study_kfold.best_value:.4f}")
print("Melhores Hiperparâmetros:", study_kfold.best_params)
```

### Task 2: Treino Final e Avaliação do Modelo K-Fold

**Files:**
- Modify: `notebooks/03_mlp_pytorch.ipynb`

- [ ] **Step 1: Adicionar célula de retreino e MLflow Tracking**

Adicione esta última célula de código ao notebook para treinar o modelo na base toda de treino (`X_train`) e gerar a pontuação oficial no `X_test` cego.

```python
import mlflow
from sklearn.metrics import average_precision_score

best_params_kf = study_kfold.best_params
best_hidden_dims_kf = [best_params_kf["hidden_size_1"], best_params_kf["hidden_size_2"]]
best_pos_w_kf = POS_WEIGHT if best_params_kf["use_class_weights"] else 1.0

# Instanciar o modelo final com os hiperparâmetros campeões do K-Fold
final_kfold_model = ChurnMLP(INPUT_DIM, best_hidden_dims_kf, best_params_kf["dropout_rate"]).to(device)

# Vamos re-treinar o modelo usando X_tr e X_val como se fossem o hold-out de early stopping tradicional
# Apenas para obtermos um modelo final treinado. (Uma prática comum pós-K-Fold é treinar em 100% de X_train com épocas fixas, 
# mas manteremos a compatibilidade com a função train_mlp existente).
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
    
    test_pr_auc_kf = average_precision_score(y_test.values, test_probs_kf)

print(f"Test PR-AUC do Modelo Vencedor (K-Fold): {test_pr_auc_kf:.4f}")

# ----------------- Registro Seguro no MLflow -----------------
mlflow.set_tracking_uri("sqlite:///../mlflow.db")
mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name="MLP_Tuned_KFold"):
    mlflow.log_params(best_params_kf)
    mlflow.log_metric("test_pr_auc", test_pr_auc_kf)
    
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