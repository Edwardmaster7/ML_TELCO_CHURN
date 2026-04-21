# Specification: Refatoração Modular Inicial (Tech Challenge Fase 1)

## Objetivo

Transformar os notebooks de experimentação (Data Science) em uma base de código modular e sustentável (Software Engineering). Esta é a primeira etapa (o "simples") do processo de maturação técnica do repositório, focada apenas em migrar código procedimental do Jupyter para scripts Python em `src/`, sem alterar orquestração avançada por enquanto.

## Estrutura Alvo da Refatoração (`src/`)

```text
src/
├── data/
│   └── preprocess.py      # Lógica de carregamento de dados e data cleaning (drop NAs, types)
├── features/
│   └── build_features.py  # Construção de features, Scikit-learn Pipeline, Scalers e OneHot
├── models/
│   └── train.py           # O script principal que une data -> features -> treino da MLP
└── utils/
    └── mlflow_utils.py    # (Opcional) Helpers para lidar com o registro de corridas
```

## Diretrizes de Implementação

1. **Ponto de Partida:** Extrair lógicas dos notebooks `01_eda_feature_engineering.ipynb` e `03_mlp_pytorch.ipynb`. O `02_baselines.ipynb` servirá como consulta.
2. **Pipelines do Scikit-Learn:** Isolar todo o processamento de features categóricas e numéricas no `ColumnTransformer`.
3. **Tracking e Artefatos (MLflow):** O script de treinamento (`train.py`) continuará usando o MLflow ativamente.
   - Deve executar `mlflow.sklearn.log_model(preprocessor, "preprocessor")` para registrar os transformadores.
   - Deve executar `mlflow.pytorch.log_model(model, "pytorch_model")` para registrar o modelo treinado final.
4. **Acoplamento Frouxo:** O módulo `features` não deve ter dependências do PyTorch, ele apenas cuida de transformar DataFrames para Arrays/Tensores. O módulo `models` é quem assume o controle do deep learning.

## Fora de Escopo

- **APIs:** A criação da API do FastAPI não será feita nesta especificação.
- **Testes Unitários:** A suite do Pytest será incluída na etapa 2 deste processo de maturação.
- **Docker / MLproject:** Orquestração avançada e conteinerização vêm depois.

## Critérios de Sucesso

- Conseguir rodar o pipeline completo apenas invocando `python src/models/train.py`.
- O MLflow UI deve conseguir exibir o experimento rodado pelo script de forma íntegra.
- Nenhuma lógica nova inserida; apenas o que já funcionava nos notebooks deve estar na pasta `src/`.
