# ADR-006: Estratégias Avançadas para MLP Tabular (Quebrando o Baseline)

**Status:** Proposto
**Data:** 23 de Abril de 2026

## Contexto
Após o estabelecimento dos baselines com as novas features de engajamento (Iteração 3), a Regressão Logística isolou-se na liderança com um PR-AUC de 0.662. O PyTorch MLP otimizado pelo Optuna não conseguiu escalar além de 0.649. O MLP vanilla treinado com `BCEWithLogitsLoss` cai na armadilha do desbalanceamento, forçando o aprendizado a inflar Recall às custas de Precision em vez de traçar uma fronteira de probabilidade bem ranqueada. Com a imposição definida pelo `ADR-002` (deve ser MLP puro), precisamos aplicar técnicas acadêmicas de alto nível para tornar nossa rede competitiva sem usar arquiteturas externas como TabNet.

## Decisão (Técnicas Propostas)
Para extrair performance otimizada de dados tabulares no PyTorch MLP, implementaremos quatro frentes conjuntas (*Regularization Cocktail*):
1.  **Tratamento de Categóricas:** Substituir One-Hot Encoding por `nn.Embedding` (Entity Embeddings) para capturar a semântica relacional das variáveis nas camadas lineares.
2.  **Métrica e Penalidade (Loss):** Transitar para Focal Loss (ou uso severo de `pos_weight` tunado). O objetivo é penalizar a rede focando seu aprendizado em exemplos difíceis/fronteiriços de churn, ao invés de acertar exemplos triviais da classe majoritária.
3.  **Modernização da Camada Linear:** Inserir blocos tipo ResNet (Skip Connections) no MLP com Layer Normalization e ativações avançadas (GELU/Mish) em vez do tradicional ReLU, prevenindo mortes de gradiente e facilitando matrizes tabulares esparsas.
4.  **Escape de Mínimos Locais:** Aplicação de Schedulers oscilantes intensos (ex: `CosineAnnealingWarmRestarts` acoplados ao `AdamW`) para empurrar os gradientes de estagnação precoce.

## Consequências
*   **Positivas:** Oferece as bases teóricas de última geração documentadas na literatura ("Well-Tuned Simple Nets Excel on Tabular Datasets") para forçar a flexibilidade expressiva da Rede Neural a efetivamente superar o teto linear encontrado pela regressão logística.
*   **Negativas (Trade-offs):** Eleva imensamente a complexidade do loop de treinamento e da classe PyTorch (passando de dezenas para centenas de linhas de customização da `nn.Module`), além de adicionar inúmeros novos hiperparâmetros (Alpha e Gamma da Focal Loss, Dimensões dos Embeddings) exigindo longas janelas de tuning no Optuna.