# Logistic Regression vs Advanced MLP Comparison Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan.

**Goal:** Provide visual evidence in the MLflow UI comparing the newly tuned advanced MLP with the original baseline (Logistic Regression) by logging comparative plots (ROC Curve, Precision-Recall Curve).

**Architecture:**
Modify `notebooks/05_mlp_pytorch_advanced_features.ipynb` to:
1. Re-train (or load, if easier to just re-train quickly) the best Logistic Regression baseline on the advanced (or base) dataset.
2. Generate matplotlib figures that plot both the Logistic Regression curves and the Advanced MLP curves on the same axes.
3. Use `mlflow.log_figure()` to attach these plots directly to the active MLflow run of the advanced MLP.

---

### Task 1: Generate and Log Comparison Plots

**Files:**
- Modify: `notebooks/05_mlp_pytorch_advanced_features.ipynb`

- [ ] **Step 1: Train Logistic Regression Baseline in the notebook**
  - Add a cell near the end (before clearing outputs) that imports `LogisticRegression` from `sklearn.linear_model`.
  - Train the baseline model (e.g., `solver='liblinear'`, `max_iter=1000`) on `X_train_np`, `y_train_np`.
  - Get probability scores (`y_scores_lr`) on `X_test_np`.

- [ ] **Step 2: Generate Comparative Plots**
  - Import `roc_curve`, `precision_recall_curve`, `auc`, `average_precision_score`.
  - Create a ROC Curve figure plotting both `Logistic Regression` and `Tuned MLP`.
  - Create a PR Curve figure plotting both `Logistic Regression` and `Tuned MLP`.

- [ ] **Step 3: Log Figures to MLflow**
  - Ensure this is done inside an active MLflow run or by starting a new one specifically for the comparison (or appending to the `mlp_advanced_tuned` run by getting its run ID). *Hint:* If the MLflow run is closed, reopen it using `with mlflow.start_run(run_id=...):` or just re-run the tuned MLP cell with the plotting appended.
  - Use `mlflow.log_figure(fig_roc, "comparison_roc_curve.png")`
  - Use `mlflow.log_figure(fig_pr, "comparison_pr_curve.png")`

- [ ] **Step 4: Cleanup and Commit**
  - Execute the notebook to generate the plots and send them to MLflow.
  - Clear notebook outputs.
  - Commit changes with message `feat: adiciona graficos comparativos com regressao logistica ao mlflow`.