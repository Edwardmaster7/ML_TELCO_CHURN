# ADR-002: Arquitetura Base do Modelo Preditivo (MLP)

**Status:** Aceito
**Data:** 23 de Abril de 2026

## Contexto
O problema central é a classificação binária de Churn em dados tabulares oriundos do banco de dados IBM Telco Customer. A literatura de Machine Learning consolida que modelos baseados em árvores com gradient boosting (XGBoost, LightGBM, CatBoost) são consistentemente os "Estado da Arte" para matrizes de dados tabulares não uniformes, superando quase sempre Redes Neurais Feed-Forward (MLPs) de prateleira, tanto em performance quanto em facilidade de uso.

## Decisão
Foi imposta uma **restrição estrita do Tech Challenge** onde o modelo final a ser produtizado **DEVE** ser uma **Rede Neural Multi-Layer Perceptron (MLP) baseada em PyTorch**. O uso de ensembles de árvore foi aceito em caráter meramente exploratório (na fase de baselines), mas não servirá como a solução final de engenharia.

## Consequências
*   **Positivas:** Permite ao aluno comprovar o domínio profundo sobre a construção manual de backpropagation, criação de classes `nn.Module`, loops de treinamento otimizados e arquiteturas complexas via tensores (PyTorch).
*   **Negativas (Trade-offs):** Estamos forçando o uso de um método sub-ótimo para dados estruturados. Isso provocou a estagnação prematura observada no modelo (teto local em PR-AUC 0.649), sendo consistentemente derrotado por um algoritmo muito mais simples: a Regressão Logística (PR-AUC 0.662). Isso exigirá uma carga extra dramática na micro-arquitetura e regularização da rede para forçá-la a competir de forma realista com a regressão ou modelos baseados em árvore.