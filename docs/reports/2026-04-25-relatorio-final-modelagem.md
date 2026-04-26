# Relatório Final da Fase de Modelagem: Predição de Churn

## 1. Introdução
Este relatório consolida os resultados finais da fase de modelagem do Tech Challenge para o problema de predição de churn de clientes em telecomunicações. O objetivo principal deste documento é apresentar o comparativo definitivo entre as abordagens testadas (desde baselines até redes neurais profundas), justificar a escolha do modelo final para produção e concluir oficialmente a etapa de experimentação.

## 2. Tabela de Métricas Finais

A tabela abaixo apresenta os resultados finais extraídos do MLflow, refletindo o desempenho dos modelos no conjunto de validação. As métricas estão ordenadas pelo PR-AUC (Precision-Recall Area Under the Curve), que foi estabelecida como a métrica principal de otimização devido ao desbalanceamento inerente da classe alvo (Churn).

| Modelo | Versão | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| **LogisticRegression_Advanced** | v2 | **0.6624** | **0.8480** | 0.5952 | **0.6711** | 0.5348 |
| MLP_Focal_KFold | v3 | 0.6539 | 0.8456 | 0.5877 | 0.6300 | 0.5508 |
| MLP_Focal_OneCycleLR | v13 | 0.6534 | 0.8460 | 0.5812 | 0.6566 | 0.5214 |
| MLP_Vanilla_KFold | v2 | 0.6495 | 0.8441 | **0.6220** | 0.4968 | **0.8316** |
| MLP_Tuned_Advanced | v2 | 0.6487 | 0.8449 | 0.6152 | 0.4920 | 0.8209 |
| MLP_Baseline_Advanced | v2 | 0.6476 | 0.8439 | 0.6141 | 0.6243 | 0.6043 |
| MLP_ResNet_Embeddings | v7 | 0.6450 | 0.8388 | 0.5853 | 0.6503 | 0.5321 |

## 3. Análise dos Resultados

A análise das métricas finais revela insights importantes sobre a modelagem deste conjunto de dados tabular:

*   **O Vencedor Absoluto:** A `LogisticRegression_Advanced` obteve o melhor desempenho geral, alcançando o maior PR-AUC (0.6624) e ROC-AUC (0.8480). Este resultado destaca que, com engenharia de features robusta, modelos lineares simples podem superar redes neurais complexas em dados tabulares estruturados.
*   **O Desempenho das Redes Neurais:** Dentre as arquiteturas de redes neurais, o `MLP_Focal_KFold` consolidou-se como a melhor abordagem pura (PR-AUC 0.6539). A combinação de K-Fold Cross Validation e Focal Loss demonstrou ser a estratégia mais eficaz para lidar com o desbalanceamento das classes dentro do paradigma de Deep Learning para este caso.
*   **A Ilusão da Complexidade (Overfitting):** O modelo mais complexo testado, a arquitetura `MLP_ResNet_Embeddings`, obteve uma pontuação inferior (PR-AUC 0.6450) e apresentou claros sinais de overfitting. Este resultado comprova empiricamente a hipótese levantada anteriormente: datasets tabulares deste porte muitas vezes carecem dos relacionamentos hierárquicos complexos necessários para justificar arquiteturas profundas com embeddings.

## 4. Decisão Final para Produção

Apesar da regressão logística avançada ter vencido numericamente, o projeto está sujeito a restrições específicas do Tech Challenge, que exigem o deploy de uma arquitetura de rede neural do tipo MLP.

Portanto, levando em consideração as restrições impostas e o desempenho validado, o modelo **`MLP_Focal_KFold`** foi escolhido como o modelo final de produção. Esta versão será empacotada e servida através do endpoint da API FastAPI na próxima fase do projeto.

## 5. Comparação Visual

Para uma análise detalhada do comportamento dos modelos, consulte as curvas ROC e Precision-Recall geradas durante a avaliação do overfitting da ResNet:

[Visualizar Comparativo de Curvas](../notebooks/resnet_comparison.png)
