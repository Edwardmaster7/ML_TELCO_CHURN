# ADR-005: Estratégia de Engenharia de Features Avançada (Data-Centric)

**Status:** Aceito
**Data:** 22 de Abril de 2026

## Contexto
Durante a Fase 2, os modelos estagnaram em PR-AUC mesmo após extenso tuning de hiperparâmetros (model-centric). Havia a necessidade de injetar features derivadas focadas no comportamento financeiro e de engajamento do cliente. Surgiu o questionamento se deveríamos simplesmente adicionar essas colunas no topo do dataset já processado (`churn_processed.csv`) ou refazer o pipeline.

## Decisão
Decidimos **reconstruir o pipeline de dados a partir das tabelas brutas** criando o dataset `churn_processed_advanced.csv` em um novo notebook (`04_advanced_feature_engineering.ipynb`), isolando o experimento da Fase 1. 

## Consequências
*   **Positivas:** 
    *   Mantém a integridade física/matemática: Evita que cálculos como divisões (ex: `TotalCharges / Tenure`) sejam feitos em cima de números que já sofreram distorções matemáticas por um `StandardScaler` prévio.
    *   Evita perda de informação que ocorre quando variáveis categóricas já foram transformadas via One-Hot Encoding antes de criar interações semânticas combinadas.
*   **Negativas (Trade-offs):** Aumenta a duplicidade de notebooks exploratórios e scripts de processamento na base de código durante a fase de prototipagem.