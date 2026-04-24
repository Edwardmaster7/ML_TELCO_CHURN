# ADR-007: Correção Metodológica de Validação (K-Fold Cross-Validation)

**Status:** Aceito
**Data:** 24 de Abril de 2026

## Contexto
Após o desenvolvimento do pipeline inicial do PyTorch MLP e da aplicação de estratégias avançadas (Focal Loss, Optuna Tuning), percebemos um padrão recorrente: os modelos atingiam altos valores de PR-AUC na Validação (ex: 0.695), mas sofriam quedas brutais no conjunto de Teste Cego (ex: 0.640). 

Diagnosticamos que isso é causado por um "Generalization Gap" (Overfitting de Hiperparâmetros). Estávamos usando um único corte estático de validação (`X_val`) tanto para o *Early Stopping* de cada época quanto como métrica-objetivo da busca Bayesiana do Optuna. Ao escalar a complexidade arquitetural do modelo em busca de contornar esse teto, ferimos o Princípio da Parcimônia (Navalha de Occam): a falha principal não estava na falta de capacidade da rede (arquitetura), mas sim na nossa metodologia de validação falha que permitia o Optuna memorizar o `X_val`.

## Decisão
Decidimos **dar um passo arquitetural para trás** antes de introduzir complexidade severa (como ResNets e Embeddings). 
Retornaremos ao primeiro notebook de PyTorch (`03_mlp_pytorch.ipynb`), que contém a MLP Vanilla com dados OHE padrão, e introduziremos uma nova seção aplicando **K-Fold Cross-Validation** no loop do Optuna. O Optuna passará a treinar 3 vezes a mesma rede e sua recompensa será a *Média* da PR-AUC nos Folds.

## Consequências
*   **Positivas:** Permite isolar e responder cientificamente se o teto de performance do MLP em comparação à Regressão Logística (0.655) era uma deficiência da rede em si ou apenas uma otimização enviesada. Demonstra altíssima maturidade analítica no histórico do projeto.
*   **Negativas (Trade-offs):** Aumenta o tempo de processamento computacional do notebook base de forma linear (cada trial do Optuna passa a treinar $K$ redes em vez de 1). Retarda temporariamente o avanço para redes neurais mais exóticas.