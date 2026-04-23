# ADR-001: Frameworks de Rastreamento e Serviço de ML

**Status:** Aceito
**Data:** 23 de Abril de 2026

## Contexto
O projeto faz parte de um Tech Challenge que exige a demonstração de maturidade em MLOps, governança de modelos e engenharia de software para entrega de inferências em produção. O ciclo de vida tradicional de exportar modelos como arquivos puros em disco (ex: `.pkl` ou `.joblib`) dificulta a reprodutibilidade e o rastreamento do histórico de treinamento.

## Decisão
1. Adotamos o **MLflow** para Rastreamento de Experimentos e "Model Registry". O salvamento dos modelos (`pytorch_model` e baselines do `sklearn`), bem como de pré-processadores associados, acontecerá nativamente via `mlflow.log_model`.
2. Adotamos o **FastAPI** como o framework web responsável por expor a API de inferência de previsão de churn.
3. A API de inferência buscará o modelo a ser utilizado utilizando exclusivamente o `RUN_ID` ou o nome registrado no Model Registry do MLflow.

## Consequências
*   **Positivas:** Teremos total rastreabilidade entre o código exato que treinou um modelo, as métricas geradas, e o artefato que será exposto pela API. Fica explícito a progressão de MLOps no repositório.
*   **Negativas (Trade-offs):** Introduz complexidade no carregamento da API (precisa estar ciente de credenciais/URI do tracking server do MLflow) e exige iniciar um servidor MLflow (`mlflow ui`) local ou remoto sempre que a pipeline for executada.