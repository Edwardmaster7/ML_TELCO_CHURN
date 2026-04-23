# ADR-004: Escolha da Métrica de Avaliação Primária (PR-AUC)

**Status:** Aceito
**Data:** 23 de Abril de 2026

## Contexto
O dataset IBM Telco Customer Churn apresenta um desbalanceamento moderado, onde a classe positiva (Churn) representa aproximadamente 26% das amostras. Em cenários de desbalanceamento, métricas como Acurácia são enganosas (um modelo que preveja "Não Churn" para todos teria ~74% de acurácia). Além disso, a curva ROC-AUC tende a ser excessivamente otimista porque é inflada pela grande quantidade de Verdadeiros Negativos (clientes que não cancelam e são fáceis de prever).

## Decisão
Estabelecemos a **PR-AUC (Area Under the Precision-Recall Curve)** como o Indicador Chave de Desempenho (KPI) primário para avaliação e tuning de todos os modelos (juntamente com o F1-Score para avaliação de ponto de corte). A curva ROC-AUC será apenas uma métrica secundária de acompanhamento.

## Consequências
*   **Positivas:** A PR-AUC foca exclusivamente na performance da classe minoritária (Churn), avaliando o quão bem o modelo consegue ranquear os churners reais sem disparar falsos alarmes excessivos (Precision vs Recall).
*   **Negativas (Trade-offs):** Torna o desafio de otimização muito mais difícil, especialmente para redes neurais vanilla, evidenciando gargalos no aprendizado que a ROC-AUC esconderia.