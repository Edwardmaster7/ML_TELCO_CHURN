# MLP Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tune the PyTorch MLP model hyperparameters, apply class weights, analyze the decision threshold based on business costs, and improve probability calibration to address low precision and optimize for the specific FP/FN cost trade-off.

**Architecture:** We will modify `notebooks/03_mlp_pytorch.ipynb` to include hyperparameter optimization using Optuna. We will also add explicit steps for threshold analysis (matching baselines) and probability calibration (using Isotonic Regression).

**Tech Stack:** PyTorch, Optuna, Scikit-Learn (IsotonicRegression), MLflow.

---

### Task 0: Setup Dependencies

- [x] **Step 1: Add Optuna**
  Adicionar a dependência do Optuna utilizando a ferramenta `uv`.
  ```bash
  uv add optuna
  git add pyproject.toml uv.lock
  git commit -m "chore: adiciona optuna as dependencias"
  ```

### Task 1: Setup Optuna and Tuning Function

**Files:**
- Modify: `notebooks/03_mlp_pytorch.ipynb`

- [ ] **Step 1: Add Optuna imports**
  Modify the first cell to include Optuna:
  ```python
  import optuna
  from optuna.integration.mlflow import MLflowCallback
  ```

- [ ] **Step 2: Create objective function for Optuna**
  Add a cell to define the Optuna objective function. This function will sample hyperparameters (learning rate, layers, dropout, class_weight), train the MLP, and return the PR-AUC (or custom metric) to optimize.
  ```python
  def objective(trial):
      # Define hyperparameters to tune
      lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
      dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
      hidden_size_1 = trial.suggest_categorical("hidden_size_1", [32, 64, 128])
      hidden_size_2 = trial.suggest_categorical("hidden_size_2", [16, 32, 64])
      weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
      
      # Determine class weights based on trial (e.g., balance vs default)
      use_class_weights = trial.suggest_categorical("use_class_weights", [True, False])
      
      # Note: Implementation of training loop inside objective goes here
      # You'll need to adapt the existing training loop to use these hyperparameters
      # and return the validation PR-AUC.
      
      return val_pr_auc
  ```

### Task 2: Implement Class Weights in PyTorch

**Files:**
- Modify: `notebooks/03_mlp_pytorch.ipynb`

- [ ] **Step 1: Calculate and apply class weights**
  In the data preparation section or inside the Optuna objective, calculate the class weights based on the training data distribution.
  ```python
  # Example calculation
  from sklearn.utils.class_weight import compute_class_weight
  class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
  pos_weight = class_weights[1] / class_weights[0]
  criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(device))
  ```
  Ensure this is integrated into the training loop, conditionally based on `use_class_weights`.

### Task 3: Threshold Optimization based on Business Costs

**Files:**
- Modify: `notebooks/03_mlp_pytorch.ipynb`

- [ ] **Step 1: Add cost-based threshold analysis**
  After tuning the model, add a cell to analyze the optimal decision threshold, given that False Negatives (FN) cost 10x more than False Positives (FP).
  ```python
  def find_optimal_threshold(y_true, y_prob, cost_fp=1, cost_fn=10):
      thresholds = np.linspace(0, 1, 100)
      costs = []
      for t in thresholds:
          y_pred = (y_prob >= t).astype(int)
          fp = np.sum((y_pred == 1) & (y_true == 0))
          fn = np.sum((y_pred == 0) & (y_true == 1))
          cost = (fp * cost_fp) + (fn * cost_fn)
          costs.append(cost)
      
      optimal_idx = np.argmin(costs)
      return thresholds[optimal_idx], costs[optimal_idx]
      
  # Apply to validation set
  opt_thresh, min_cost = find_optimal_threshold(y_val, y_prob_val)
  print(f"Optimal Threshold: {opt_thresh:.4f}, Min Cost: {min_cost}")
  ```

### Task 4: Probability Calibration

**Files:**
- Modify: `notebooks/03_mlp_pytorch.ipynb`

- [ ] **Step 1: Evaluate current calibration**
  Plot the calibration curve (reliability diagram) for the tuned MLP.
  ```python
  from sklearn.calibration import calibration_curve
  prob_true, prob_pred = calibration_curve(y_test, y_prob_test, n_bins=10)
  plt.plot(prob_pred, prob_true, marker='o')
  plt.plot([0, 1], [0, 1], linestyle='--')
  plt.show()
  ```

- [ ] **Step 2: Apply Isotonic Regression**
  Use `IsotonicRegression` from `sklearn.calibration` (or `CalibratedClassifierCV` if wrapping) to calibrate the probabilities. Note that for PyTorch, we fit the calibrator on validation probabilities.
  ```python
  from sklearn.isotonic import IsotonicRegression
  
  iso_reg = IsotonicRegression(out_of_bounds='clip')
  iso_reg.fit(y_prob_val, y_val)
  
  y_prob_test_calibrated = iso_reg.predict(y_prob_test)
  ```
  Re-evaluate Precision, Recall, PR-AUC and ROC-AUC after calibration.

### Task 5: MLflow Integration

**Files:**
- Modify: `notebooks/03_mlp_pytorch.ipynb`

- [ ] **Step 1: Log Optuna study and Best Model**
  Ensure the Optuna study is logged to MLflow, and the best hyperparameters, best threshold, and calibration metrics are all logged.
  ```python
  with mlflow.start_run(run_name="Tuned_MLP_Calibrated"):
      mlflow.log_params(study.best_params)
      mlflow.log_metric("optimal_threshold", opt_thresh)
      mlflow.log_metric("min_cost", min_cost)
      # Log calibrated metrics (PR-AUC, F1, Precision, Recall at optimal threshold)
      
      # Log the model (including the calibrator if necessary, or the PyTorch model)
      mlflow.pytorch.log_model(best_model, "mlp_model")
      # Optional: Save and log the calibrator using sklearn
  ```
