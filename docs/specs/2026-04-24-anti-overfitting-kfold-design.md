# Especificação de Design: Fase 4 - Combate Severo a Overfitting e Governança MLOps

**Data:** 24 de Abril de 2026
**Tópico:** Implementação de K-Fold CV, Ruído Gaussiano e Refatoração de Código (ADR-007)
**Arquivos Alvo:** `notebooks/07_mlp_resnet_embeddings.ipynb` (Refatorado) e `src/models/tabular_resnet.py`

## 1. Visão Geral

Após o sucesso matemático da Focal Loss, Embeddings e arquitetura ResNet (Iteração 5/Notebook 07), o modelo exibiu o que chamamos de *Generalization Gap* — o Optuna atingiu 0.695 de PR-AUC na Validação, mas o Teste caiu para 0.640. O modelo sofre de Overfitting aos hiperparâmetros por conta da altíssima capacidade expressiva avaliada sempre sobre o mesmo split. Adicionalmente, as classes PyTorch foram inseridas no arquivo do Jupyter, o que fere a governança do repositório (apontado pelo Agente de Code Review). Esta especificação traça as defesas contra overfitting e quita o débito arquitetural.

## 2. Governança e Refatoração (Clean Architecture)

Todas as estruturas matemáticas e de Deep Learning serão fisicamente retiradas dos *notebooks* de experimentação.

* **Destino:** Um novo módulo será criado em `src/models/tabular_resnet.py`.
* **Componentes Exportados:** A classe da função de perda (`FocalLoss`), o bloco residual tabulado (`ResNetBlock`) e a rede neural aglutinadora (`AdvancedChurnMLP`).
* **Impacto:** Os Jupyter Notebooks passarão a apenas invocar a classe, focando exclusivamente na orquestração do treinamento e hiperparametrização via Optuna, alinhando-se aos princípios S.O.L.I.D. e às restrições do Tech Challenge.

## 3. Otimização de MLflow (Pipeline Serializado)

Para sanar o defeito arquitetural que impediria a construção da camada de serving (FastAPI) no futuro:

* A classe `ColumnTransformer` que hospeda as imputações numéricas e o `OrdinalEncoder` (com `unknown_value=-1`) deverá ser logada no registro do experimento através do comando `mlflow.sklearn.log_model(preprocessor, "preprocessor")`.
* O carregamento seguro da rede na API dependerá tanto dos pesos em PyTorch quanto da máquina de tradução categórica gerada pelo pipeline acima.

## 4. O Regimento "Anti-Overfitting" (Data e Treinamento)

### 4.1. Camada de Ruído Gaussiano (Gaussian Noise)

A literatura comprova que injetar ruído randômico de baixa escala nas features numéricas atua como uma forte regularização estrutural, forçando a rede a focar no "desenho das distribuições" ao invés de decorar valores flutuantes exatos (os "pontos") do conjunto de treino.

* **Implementação:** Uma classe customizada `GaussianNoise(nn.Module)`.
* **Comportamento:** Somente se ativará na passagem direta quando o estado da rede for `model.train()`. Somará uma amostra $\mathcal{N}(0, \sigma^2)$ ao tensor das features numéricas (onde $\sigma \approx 0.05$).

### 4.2. Validação Cruzada K-Fold (A Defesa do Optuna)

Ao invés de delegar a decisão de *early-stopping* e melhoria da função-objetivo para uma única repartição estática de validação (`X_val`), mudaremos o modelo de re-avaliação do Optuna:

* Apenas criaremos o Split de Teste (Cego) com 20% dos dados. Os outros 80% servirão como base para uma segmentação dinâmica.
* Implementaremos o `StratifiedKFold(n_splits=3)` do Scikit-Learn no Optuna.
* Para CADA *trial* da varredura, a rede neural será estanciada e treinada **3 vezes**, cada vez testada num *fold* diferente de validação.
* O retorno `return` do Optuna não será o pico do PR-AUC de uma passagem isolada, mas a **média do PR-AUC dos 3 cortes**. Hiperparâmetros que não generalizam receberão uma média baixa e serão varridos da seleção.

### 4.3. Pruning do Espaço de Busca

Para combater a enorme capacidade da rede em esgotar graus de liberdade em memorização, o Optuna será restrito a fornecer apenas redes de menor complexidade e maior penalização nos pesos:

* `hidden_dim`: Restrito ao teto de 32 ou 64 neurônios (abandonado 128/256).
* `num_blocks`: Restrito a redes rasas (1 ou 2 blocos de salto) ao invés de redes profundas (4 blocos).
* `weight_decay` (Regularização $L2$ no `AdamW`): Limite de tolerância elevado para o intervalo restrito de $[1e-4, 5e-3]$, forçando uma penalização real por complexidade excessiva.
