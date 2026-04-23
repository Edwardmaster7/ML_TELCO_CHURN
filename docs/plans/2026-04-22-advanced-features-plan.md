# Advanced Feature Engineering & MLP Retraining Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement new financial and engagement features to provide more predictive power to the MLP model, breaking the current PR-AUC performance ceiling. To avoid clutter, this will be done in brand new notebooks.

**Architecture:** 
1. `notebooks/04_advanced_feature_engineering.ipynb` -> Reads the original raw data (or the intermediate merge), creates the new features, and outputs `churn_processed_advanced.csv`.
2. `notebooks/05_mlp_pytorch_advanced_features.ipynb` -> Reads the new advanced dataset, implements the PyTorch MLP (with Optuna and baseline comparison), and logs to a new MLflow experiment (`03_PyTorch_Advanced_FE`).

---

### Task 1: Create the Advanced Feature Engineering Notebook

**Files:**
- Create: `notebooks/04_advanced_feature_engineering.ipynb`

- [ ] **Step 1: Setup and Data Loading**
  - Import pandas, numpy, sklearn preprocessing tools.
  - Load the raw tables (`customers`, `services`, `contracts`) and merge them exactly as in `01_eda_feature_engineering.ipynb`.

- [ ] **Step 2: Base Preprocessing**
  - Fix `TotalCharges` (to numeric, impute with median).
  - Map the target `Churn` to 0/1.
  - Map simple Yes/No columns to 0/1.

- [ ] **Step 3: Advanced Feature Engineering**
  - Re-implement previous derived features: `is_monthly_contract`, `is_new_customer`.
  - **New Feature 1:** `charges_per_tenure` = `TotalCharges / (Tenure + 1)`
  - **New Feature 2:** `is_high_spender` = 1 if `MonthlyCharges` > 75th percentile, else 0
  - **New Feature 3:** `total_services_count` = sum of flags for `['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']` (where value is 'Yes')
  - **New Feature 4:** `has_protection_services` = 1 if any of `['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']` is 'Yes', else 0

- [ ] **Step 4: Sklearn Pipeline and Export**
  - Create the `ColumnTransformer` (Scale numerics, OHE multi-class categoricals).
  - Apply the transformation.
  - Export to `notebooks/data/processed/churn_processed_advanced.csv`.

### Task 2: Create the Advanced MLP Notebook

**Files:**
- Create: `notebooks/05_mlp_pytorch_advanced_features.ipynb`

- [ ] **Step 1: Setup MLflow and Data**
  - Set MLflow URI to `http://127.0.0.1:5000` and experiment to `03_PyTorch_Advanced_FE`.
  - Load `churn_processed_advanced.csv`.
  - Split into train, validation, and test sets.

- [ ] **Step 2: Implement the PyTorch MLP**
  - Copy the PyTorch model class, DataLoader logic, and training loop from `03_mlp_pytorch.ipynb`.

- [ ] **Step 3: Train Vanilla Advanced MLP (Baseline)**
  - Train the model using default hyperparameters (no class weights, static LR).
  - Evaluate on Test set, calculating PR-AUC, ROC-AUC, F1, Recall, Precision.
  - Log to MLflow under run name `mlp_advanced_baseline`.

- [ ] **Step 4: Optuna Tuning for Advanced MLP**
  - Implement Optuna objective maximizing validation PR-AUC.
  - Train the best model found by Optuna.
  - Log to MLflow under run name `mlp_advanced_tuned`.

- [ ] **Step 5: Comparison and Commit**
  - Print a DataFrame comparing the `mlp_advanced_baseline` vs `mlp_advanced_tuned`.
  - Clear notebook outputs (`jupyter nbconvert --clear-output`).
  - Commit changes with message `feat: cria notebooks para feature engineering avancada e retreino da mlp`.