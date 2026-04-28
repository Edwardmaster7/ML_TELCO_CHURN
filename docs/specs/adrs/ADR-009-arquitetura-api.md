# ADR-009: Arquitetura da API de Inferência (FastAPI e MLflow)

## Status
Aceito

## Contexto
Atendendo aos requisitos da Etapa 3 do Tech Challenge, o projeto deve expor a arquitetura de modelo final (`MLP_Focal_KFold`) através de uma API. A API deve realizar predições de Churn garantindo a reprodutibilidade dos dados de entrada (Data Contract), validação restrita, logging estruturado e integração contínua com os artefatos de treinamento para evitar *training-serving skew*.

## Decisão
Foi desenhada uma arquitetura baseada no padrão "Controller-Service" utilizando FastAPI e a plataforma MLflow (Model Registry) como fonte de verdade para os artefatos da rede neural.

1. **Estratégia de Inferência Síncrona:** A API atuará no modo *Real-Time Unitário*, recebendo um único payload JSON na rota `POST /predict`. Embora o processamento Batch seja mais eficiente, a requisição unitária atende ao caso de uso primário da predição pontual exigida.
2. **Ciclo de Vida de Inicialização (Cold Start):** Durante o *lifespan* do FastAPI, a aplicação conectará no servidor MLflow e baixará **dois artefatos** na memória (singleton) da última *run* de produção (gerada pelo script `src/models/train.py`):
   - O pré-processador treinado (Scikit-Learn).
   - O modelo PyTorch (Arquitetura ChurnMLP).
3. **Data Contract Estrito (Pydantic):** Para prevenir Data Leakage e corrupção de tipos, o Schema Pydantic de entrada espelhará os 20 atributos categóricos e numéricos crus do *dataset* original. Haverá coerção explícita de `TotalCharges` (tratando strings vazias ou nulas).
4. **Acoplamento do Pipeline Data-Centric:** Após a validação do Payload Pydantic, os dados serão convertidos em um DataFrame Pandas de linha única, passando pela função `clean_data()` da pipeline antes do objeto `preprocessor.transform()`.
5. **Observabilidade Obrigatória:** Foi definido um middleware de requisição que calculará a latência da rota de predição, registrando em um sistema de *structured logging*.

## Consequências
- **Positivas:** Permite que a ciência de dados itere nos modelos PyTorch sem necessitar tocar no repositório da API (desacoplamento via Model Registry). Garante tipagem forte desde o edge (rede HTTP) até o tensor de inferência.
- **Negativas:** Exige que a base local do MLflow SQLite (`mlflow.db`) e a pasta `mlartifacts` estejam íntegras e acessíveis no deploy para que a API inicie, aumentando o peso da esteira de empacotamento Docker futuramente.