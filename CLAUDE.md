# Guia de Desenvolvimento - ML_TELCO_CHURN

Este arquivo serve como base para orientar o desenvolvimento e as interações com o projeto.

## Comandos de Setup
- **Ativar o ambiente virtual:** `source venv/bin/activate` (ou ative o ambiente conda correspondente caso esteja utilizando o Anaconda/Miniconda)
- **Instalar dependências:** `uv sync`

## Comandos de Teste
- **Executar a suíte de testes:** `pytest tests/`
  *(Nota: O projeto segue uma abordagem de "Testes Detalhados").*

## Comandos Executáveis
- **Visualizar o tracking de experimentos:** `mlflow ui`

## Padrões de Código e Diretrizes

### 1. Evolução e Arquitetura MLOps
- O projeto simulará o **processo de maturação MLOps**, partindo de scripts procedurais iniciais, evoluindo para uma API estruturada e, finalmente, migrando para orquestração e containerização.

### 2. Estrutura de Diretórios
- Todo o código refatorado de produção deve ficar restrito às pastas modulares dentro do diretório `src/`:
  - `src/data/` - Ingestão e manipulação de dados
  - `src/features/` - Engenharia de features e pré-processamento
  - `src/models/` - Treinamento, avaliação e definições de modelos
  - `src/api/` - Código relacionado à inferência e endpoints da API

### 3. Serving de Inferência
- A inferência em produção/API será servida utilizando o framework **FastAPI**.

### 4. Integração com MLflow
- Salvaremos os artefatos (como o `preprocessor` e o `pytorch_model`) obrigatoriamente através do **tracking do MLflow** (evitar o uso de exportação direta com joblib puro para modelos finais).
- Na etapa da API, o carregamento dos modelos e pré-processadores deve ser feito utilizando o `RUN_ID` de registro do MLflow.

### 5. Controle de Versão (GIT)
- **REGRA RÍGIDA:** Absolutamente todas as mensagens de commits DEVEM ser escritas em **Português (pt-BR)**. Siga o padrão de conventional commits (ex: `feat:`, `fix:`, `chore:`, etc.).

### 6. Documentação
- Arquivos de documentação sobre escopos, planos arquiteturais e especificações devem sempre ser armazenados exclusivamente nos diretórios:
  - `docs/specs/`
  - `docs/specs/adrs/` (Para Architecture Decision Records)
  - `docs/plans/`
- **Decisões Técnicas:** Utilize o formato **ADR (Architecture Decision Record)** para registrar escolhas arquiteturais ou de modelagem significativas. Um ADR é um documento curto (~1 página) que registra o contexto, a decisão tomada e as alternativas consideradas/trade-offs (Ex: *ADR-001: Escolhemos usar MLP em vez de XGBoost porque...*).
- **Relatório Consolidado:** Mantenha as atualizações macro do histórico do projeto no arquivo global de narrativa `docs/tech_challenge_decisions.md`.
- *Nota: É terminantemente proibida a criação ou o uso de pastas não-padronizadas (ex: não usar `docs/superpowers`).*
