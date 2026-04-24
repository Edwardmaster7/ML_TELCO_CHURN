# Relatório de Estratégias e Desempenho: PyTorch MLP vs Regressão Logística em Previsão de Churn

## 1. Contexto do Problema
- **Dataset:** IBM Telco Customer Churn (Dados Tabulares).
- **Problema:** Classificação Binária (Previsão de Cancelamento / Churn).
- **Característica:** Dataset moderadamente desbalanceado (Churn representa a classe minoritária).
- **Métrica Principal:** PR-AUC (Area Under the Precision-Recall Curve), devido ao desbalanceamento, seguida de F1-Score e ROC-AUC.
- **Objetivo:** Fazer com que um modelo Multi-Layer Perceptron (MLP) em PyTorch supere consistentemente o baseline de Regressão Logística, especialmente na métrica PR-AUC.

---

## 2. Histórico Cronológico de Estratégias e Resultados

### Fase 1: Baselines Tradicionais (Experimento 02)
Começamos treinando modelos tradicionais do Scikit-Learn utilizando as 31 features originais (após imputação, One-Hot Encoding e StandardScaler).
- **Modelos testados:** Logistic Regression, Random Forest, Gradient Boosting.
- **Vencedor dos baselines:** Regressão Logística.
  - *Resultados LogReg:* ROC-AUC: 0.846 | **PR-AUC: 0.652** | F1: 0.595 | Precision: 0.671 | Recall: 0.535
- **Observação:** Modelos baseados em árvore (RF, GB) tiveram desempenho muito inferior em PR-AUC (0.614 e 0.628, respectivamente).

### Fase 2: Implementação do Primeiro PyTorch MLP (Experimento 02)
Criamos uma rede neural simples para tentar capturar padrões não lineares complexos que a Regressão Logística pudesse estar perdendo.
- *Resultados MLP Baseline:* ROC-AUC: 0.847 | **PR-AUC: 0.659** | F1: 0.618 | Precision: 0.498 | Recall: 0.816
- **Análise:** O MLP obteve um ganho marginal no PR-AUC em relação à LogReg. Ele demonstrou um comportamento extremamente agressivo na detecção de churn (Recall altíssimo de 81%), mas com o custo de derrubar a Precision para menos de 50%.

### Fase 3: Hyperparameter Tuning e Calibração de Probabilidades (Experimento 02)
Para tentar equilibrar o modelo e aumentar o PR-AUC, introduzimos o Optuna e calibração.
- **Estratégia:** Otimização via Optuna focada em maximizar o PR-AUC no conjunto de validação. Aplicamos Calibração via Isotonic Regression.
- *Resultados MLP Tuned (Sem Calibração):* ROC-AUC: 0.847 | PR-AUC: 0.655 | Precision: 0.483 | Recall: 0.850. (A otimização apenas maximizou o Recall, piorando o PR-AUC).
- *Resultados MLP Tuned (Calibrado):* PR-AUC: 0.619 | Precision: 0.673 | Recall: 0.495.
- **Análise:** A calibração foi um desastre comportamental. Ela reverteu a rede neural para se comportar como a Regressão Logística, afundando o PR-AUC.

### Fase 4: Engenharia de Features Avançada - Abordagem Data-Centric (Experimento 03)
Adicionamos 5 novas features de **Engajamento** e **Financeiras** e reconstruímos o dataset do zero. Total de features: 36.

### Fase 5: Retreinamento com Features Avançadas (Experimento 03)
- *Resultados LogReg (Advanced):* ROC-AUC: 0.848 | **PR-AUC: 0.662** | F1: 0.595 | Precision: 0.671 | Recall: 0.535
  - *Comentário:* A LogReg soube aproveitar as novas features, elevando o teto do PR-AUC.
- *Resultados MLP Tuned (Advanced):* ROC-AUC: 0.845 | PR-AUC: 0.649 | Precision: 0.492 | Recall: 0.821
  - *Comentário:* Voltou ao comportamento super agressivo e falhou em bater o novo teto de 0.662 da LogReg.

---

## 3. O Gargalo Atual e a Constatação
1. **Modelos Lineares estão dominando:** A Regressão Logística extrai mais valor preditivo seguro (PR-AUC 0.662) das novas features.
2. **Resistência Tabular:** Confirma-se a dificuldade histórica de redes neurais "vanilla" baterem modelos lineares em datasets tabulares de tamanho moderado sem truques avançados.

---

## 4. O Que Precisamos (Prompt para o Deep Research)

"Estou enfrentando um teto de performance em um problema de detecção de churn tabular (IBM Telco Churn, ~7k linhas). Uma Regressão Logística estabeleceu o teto com um PR-AUC de 0.662. Um PyTorch MLP otimizado com Optuna falhou em quebrar esse limite, estacionando em 0.649. O MLP sofre com um trade-off rígido de Recall muito alto e Precision baixa. 

**MUITO IMPORTANTE - RESTRIÇÕES DO PROJETO:** 
Estou trabalhando em um "Tech Challenge" acadêmico/profissional onde os requisitos são estritos: **preciso obrigatoriamente utilizar uma arquitetura baseada em Multi-Layer Perceptron (MLP) puro (Feed-Forward Neural Network) desenvolvida em PyTorch**. 
- Não posso utilizar frameworks baseados em árvores (XGBoost, LightGBM, Random Forest).
- Não posso apelar para arquiteturas tabulares especializadas complexas de artigos de pesquisa (como TabNet, FT-Transformer ou SAINT). 
O objetivo do desafio é mostrar domínio de engenharia, otimização e MLOps construindo um MLP tradicional (Linear layers, activations, dropout).

Dadas essas restrições restritas à arquitetura de um MLP, qual é a melhor estratégia algorítmica e matemática para fazer minha rede neural destruir o baseline da Regressão Logística em PR-AUC?

Busco sugestões aplicáveis EXCLUSIVAMENTE a um PyTorch MLP padrão, abordando:
1. **Tratamento de Entradas:** É melhor manter o One-Hot Encoding ou implementar Entity Embeddings (`nn.Embedding`) para as variáveis categóricas para melhorar a representação perante as camadas lineares?
2. **Função de Perda:** Considerando o desbalanceamento e foco em PR-AUC, devo abandonar a `BCEWithLogitsLoss`? O uso de Focal Loss (com parâmetros Alpha e Gamma otimizados) ou Dice Loss tem comprovação de melhorar PR-AUC em MLPs tabulares?
3. **Micro-arquitetura Interna:** Vale a pena adicionar skip-connections (Tornando-o um ResNet-like MLP)? Substituir ReLU por ativações mais modernas (GELU, Mish)? O que é melhor para MLPs tabulares: Batch Normalization ou Layer Normalization?
4. **Regime de Treinamento:** A adoção de schedulers agressivos como `OneCycleLR` ou `CosineAnnealingWarmRestarts` ajuda na fuga desses mínimos locais estruturais que limitam a precisão?

Por favor, forneça o raciocínio de quais dessas técnicas combinadas têm a maior probabilidade de fazer um MLP puro vencer um modelo linear neste cenário."