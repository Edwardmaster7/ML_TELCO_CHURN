# ADR-004: Abandono da Calibração de Probabilidade Isotonic/Platt

**Status:** Aceito
**Data:** 23 de Abril de 2026

## Contexto
Durante o tuning do MLP PyTorch (Iteração 2), o modelo resultante exibia um altíssimo Recall (~85%) e baixa Precision (~48%). Em uma tentativa de corrigir a distribuição enviesada das saídas logit, aplicamos a `IsotonicRegression` como método de calibração post-hoc no conjunto de validação.

## Decisão
Decidimos **abandonar e remover a calibração de probabilidade post-hoc** na arquitetura do modelo de Churn. O problema do desbalanceamento será tratado "na raiz" da rede (via *Loss Function* e amostragens) e não por ajustes matemáticos da saída final.

## Consequências
*   **Positivas:** Previne a quebra do ranqueamento relativo. A calibração de probabilidade estava achatando a curva, destruindo o Recall (que caiu de 85% para 49.5%) e reduzindo o PR-AUC global (0.654 para 0.619). Ao abandonar a calibração, preservamos a capacidade agressiva de detecção do MLP, cuja precisão poderá ser refinada movendo o limiar de decisão de negócio (Threshold Optimization) em vez de recalibrar a distribuição estática.
*   **Negativas (Trade-offs):** As probabilidades de saída (saída da função Sigmoid) geradas pelo modelo não podem ser interpretadas como probabilidades reais estritas (ex: "tem 72% de chance exata matemática de cancelar"), devendo ser tratadas apenas como **Scores de Risco** ordinais.