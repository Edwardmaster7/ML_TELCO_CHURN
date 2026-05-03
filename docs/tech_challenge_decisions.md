# Decisões de Projeto — ML Telco Churn

> **Tipo:** Narrativa histórica de decisões arquiteturais e de modelagem
> **Projeto:** Tech Challenge Fase 01 — ML Engineering (FIAP Pós-Tech, Grupo 21)
> **Atualizado em:** 2026-05-02

Este documento consolida as decisões arquiteturais, estratégicas e as restrições impostas pelo **Tech Challenge**, detalhando o caminho percorrido através da evolução do código (notebooks 01 a 07) e a escolha do modelo campeão para deploy.

## 1. Restrições e Escopo do Projeto (Tech Challenge - Fase 1)
O projeto visa demonstrar a aplicação de boas práticas de Engenharia de Machine Learning e MLOps. A evolução deve mimetizar o processo de maturação técnica diretamente no histórico de commits (de scripts modulares simples para MLflow Projects e Docker).

### 1.1 Restrições Rigorosas (Memória e CLAUDE.md)
*   **Arquitetura do Modelo Final:** É *obrigatório* o uso de uma **Rede Neural Multi-Layer Perceptron (MLP) puro (Feed-Forward Neural Network)** implementada em **PyTorch**.
*   **Modelos Proibidos (para a versão final):** Modelos baseados em árvores (Random Forest, Gradient Boosting, XGBoost, LightGBM) ou arquiteturas tabulares complexas de pesquisa recente. Tais modelos foram testados como baselines (Exp 02), mas não compõem a entrega de Deep Learning.
*   **Framework de Serviço:** A inferência em produção deve ser servida exclusivamente usando **FastAPI**.
*   **Versionamento e Rastreabilidade (MLflow):** O MLflow atua como o "Model Registry" nativo (usando `mlflow.sklearn.log_model` e `mlflow.pytorch.log_model`). O carregamento na API usará o `RUN_ID`, banindo o uso de manipulação de arquivos `.joblib` brutos.
*   **Idiomas e Padrões de Commit:** Todos os commits do repositório DEVEM ser em **Português (pt-BR)** seguindo *Conventional Commits* (`feat:`, `fix:`, `chore:`).
*   **Estrutura do Projeto:** Código de produção vive restritamente em `src/` (com módulos `data/`, `features/`, `models/`, `api/`). Documentação vive *exclusivamente* em `docs/specs/` e `docs/plans/`. Abordagem de Testes será Detalhada.

## 2. Histórico e Evolução do Desenvolvimento (Notebooks 01 a 05)

A jornada da modelagem seguiu uma abordagem iterativa focada em resolver o problema do aprendizado com dados tabulares desbalanceados (26% de churn). O KPI primário escolhido foi a **PR-AUC** (Área sob a curva Precision-Recall).

### Iteração 1: Análise e Baselines (`01_eda_feature_engineering.ipynb` e `02_baselines.ipynb`)
*   **O Que Foi Feito:** Limpeza básica (ex: conversão de `TotalCharges`), One-Hot Encoding e StandardScaler nas 31 variáveis categóricas/numéricas originais. Treinamos Logistic Regression, Random Forest e Gradient Boosting.
*   **O Que Aprendemos:** A Regressão Logística estabeleceu um "teto linear" muito forte e robusto (PR-AUC: 0.652), batendo os modelos de árvore em dados originais sem vazamentos.

### Iteração 2: A Incursão do PyTorch e Tuning (`03_mlp_pytorch.ipynb`)
*   **O Que Foi Feito:** Construção do primeiro MLP em PyTorch, integração profunda com o Optuna para tuning (learning rate, camadas, weight decay) e tentativa de plugar `IsotonicRegression` para calibração.
*   **O Que Aprendemos:**
    *   O MLP baseline era agressivo (alta revocação / Recall 81%, baixa precisão).
    *   *Falha da Calibração:* A calibração de probabilidade não funcionou como esperado para melhorar a PR-AUC. Ela conteve a agressividade do modelo elevando a precisão, mas esmagando o Recall (caindo para 49.5%). O MLP ficou funcionalmente idêntico à Regressão Logística, perdendo a sua vantagem exploratória inicial (PR-AUC caiu para 0.619).

### Iteração 3: Feature Engineering Avançada (`04_advanced_feature_engineering.ipynb` e `05_mlp_pytorch_advanced_features.ipynb`)
*   **A Decisão (Memória da Equipe):** Devido à estagnação do tuning do MLP, a abordagem migrou de otimização de modelo para enriquecimento semântico dos dados. Optou-se por separar a base de código em novos notebooks para não corromper o histórico inicial.
*   **O Que Foi Feito:** Processamento a partir dos dados crus inserindo novos construtos comportamentais: `charges_per_tenure`, `is_high_spender`, `total_services_count` e `has_protection_services`.
*   **O Que Aprendemos:**
    *   As features foram um sucesso absoluto para o modelo linear. A Regressão Logística subiu o teto do PR-AUC para **0.662**.
    *   O PyTorch MLP Tuned se afundou novamente em um trade-off rígido. Voltou a priorizar Recall alto e parou em PR-AUC **0.649**, sendo ofuscado novamente pela simplicidade da regressão.

### Iteração 4: Fase 1 das Estratégias Avançadas (Focal Loss e OneCycleLR) (`06_mlp_advanced_loss.ipynb`)
*   **O Que Foi Feito:** Implementação do início das táticas baseadas no estado-da-arte para dados tabulares, aplicando a **Focal Loss** em conjunto com **AdamW** e o otimizador oscilante **OneCycleLR**. Reutilizamos o mesmo dataset base processado do experimento 05.
*   **O Que Aprendemos:**
    *   Conseguimos dominar a arquitetura técnica da Focal Loss em PyTorch mantendo um PR-AUC validado elevado no Optuna (0.691).
    *   No entanto, sofremos do *Generalization Gap* quando medido contra os dados puros de Teste, amargando um PR-AUC final de **0.647**.
    *   A constatação dessa distorção nos levou ao questionamento metodológico solucionado na Iteração 5.

## 3. A Crise de Generalização e a Correção Metodológica (K-Fold)

Ao avaliarmos friamente a Iteração 4, o princípio da Navalha de Occam tornou-se evidente: estávamos escalando drasticamente a complexidade matemática da rede (Losses e Schedulers) para combater um sintoma de um problema que era metodológico. 

O *Generalization Gap* (queda brutal de PR-AUC da Validação para o Teste) não era primariamente uma falha de arquitetura, mas sim um **Hyperparameter Overfitting**. O framework de tuning (Optuna) estava avaliando milhares de épocas e dezenas de iterações sobre o mesmo corte estático de validação (`X_val`), decorando as flutuações amostrais e falhando ao enfrentar o Teste Cego.

### Iteração 5: O Passo Metodológico Atrás (`03_mlp_pytorch.ipynb` e `06_mlp_advanced_loss.ipynb`)
Decidimos frear a introdução de *ResNet Blocks* e *Embeddings* e retornar para os dois experimentos anteriores com o objetivo de higienizá-los estatisticamente.
*   **O Que Foi Feito:** Retornamos ao notebook base (`03`) e ao notebook avançado de perda (`06`), adicionando seções finais onde injetamos o rigor da **Validação Cruzada K-Fold (StratifiedKFold)** diretamente no coração do Optuna. O framework passou a avaliar o modelo por sua *Média de PR-AUC* em múltiplos folds cegos.
*   **O Que Aprendemos:** 
    *   No MLP base (Notebook 03), a métrica ajustou para 0.639, erradicando o overfitting, mas solidificando o teto de 0.64 como limite máximo daquela arquitetura pobre.
    *   Na MLP Avançada com Focal Loss (Notebook 06), a técnica brilhou: sem o overfitting, o modelo consolidado e validado por K-Fold atingiu a sua melhor pontuação histórica no Teste Cego: **0.6512**.
*   **A Racionalidade:** O K-Fold encostou nossa rede neural na Regressão Logística (0.655), atestando o funcionamento perfeito da engenharia da função Focal Loss. O último gargalo agora restrito para vencer a LogReg de vez reside no One-Hot Encoding.

## 4. Próximos Passos Estratégicos (Quebrando o Teto Linear)

Como a Focal Loss otimizada com a validação K-Fold atestou o pleno funcionamento da otimização Bayesiana, a arquitetura avançará para o estágio de modernização estrutural:
1.  **Embeddings em vez de One-Hot:** Destruição do pipeline Scikit-Learn de *OneHotEncoder* e aplicação de `nn.Embedding` (Entity Embeddings) para capturar a geografia semântica.
2.  **Modernização Intra-Camada (Micro-arquitetura):** Incorporação de ResNet-like blocks (Skip Connections) no MLP com Layer Normalization e ativação GELU.

## 5. Conclusão da Modelagem e Escolha para Deploy (O Veredito Final)

Com a execução da iteração final, a fase de modelagem foi oficialmente encerrada.

Os testes extensivos comprovaram que a **Regressão Logística** atua como o melhor modelo absoluto para este dataset tabular específico. A tentativa de aplicar **ResNet com Embeddings** (mesmo com regularização) resultou em overfitting. Isso confirmou nossa hipótese analítica: o dataset do IBM Telco carece da hierarquia complexa de features (deep feature hierarchy) que as redes profundas como ResNets precisam para extrair valor, fazendo com que o modelo decore o ruído.

**Decisão Final para Deploy:**
Apesar da superioridade simples da Regressão Logística neste contexto, as restrições do **Tech Challenge** exigem o deploy de uma arquitetura baseada em MLP puramente em PyTorch.

Desta forma, o modelo **`MLP_Focal_KFold`** (com PR-AUC de 0.6539) foi selecionado como o modelo campeão. Ele atende de forma irretocável às restrições acadêmicas da entrega e garantiu a melhor capacidade de generalização e robustez encontrada nas arquiteturas de redes neurais testadas. É este modelo que será embarcado e servido em nossa API utilizando **FastAPI**.