# Diretrizes e Decisões de Modelagem: Previsão de Churn

## Visão Geral
Este documento consolida as decisões arquiteturais, estratégicas e as restrições impostas pelo **Tech Challenge**, detalhando o caminho percorrido através da evolução do código (notebooks 01 a 05) e as próximas etapas na modelagem de Churn para o dataset IBM Telco.

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

## 3. Próximos Passos Estratégicos (Quebrando o Teto Linear)

A atual restrição do projeto exige superação arquitetural. Como um PyTorch MLP padrão falhou em captar o valor intrínseco das features avançadas na mesma medida em que a Regressão Logística captou, a modelagem neural sofrerá mutações baseadas no estado da arte de *Deep Learning for Tabular Data* sem infligir as regras de entrega do Tech Challenge:

1.  **Função de Perda Agressiva (Loss):** Transição de `BCEWithLogitsLoss` normal para Focal Loss ou uso estrito do fator `pos_weight`, focando explicitamente a atenção do gradiente nas fronteiras difíceis do ranqueamento e não apenas na predição de probabilidade.
2.  **Modernização Intra-Camada (Micro-arquitetura):** Incorporação de ResNet-like blocks (Skip Connections) no MLP, Layer Normalization em substituição ao Batch Normalization em batch sizes pequenos e transição de ativação ReLU para GELU/Mish.
3.  **Embeddings em vez de One-Hot:** As variáveis categóricas serão processadas por `nn.Embedding` (Entity Embeddings) para capturar a geografia semântica (ex: proximidade entre os diferentes tipos de contrato ou formas de pagamento eletrônico), fornecendo insumos mais ricos para a Linear Layer inicial.
4.  **Regime de Escape (Schedulers):** Adoção de schedulers oscilantes intensos (OneCycleLR, CosineAnnealingWarmRestarts) associados ao AdamW para expulsar o treinamento do buraco negro do ótimo local precoce induzido pelo desbalanceamento natural da base.