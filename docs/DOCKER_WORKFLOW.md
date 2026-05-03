# Docker Workflow - Training e Deploy sem Stress

## 📋 Pré-requisitos

- Docker e Docker Compose instalados
- Dependências Python via `uv sync` (já instaladas localmente)
- dados em `notebooks/data/raw/` (churn_customers.csv, churn_services.csv, churn_contracts.csv)

---

## 🚀 Passo a Passo Completo

### 1️⃣ Iniciar a Stack Docker

Suba todos os serviços (MLflow, API, Prometheus, Grafana):

```bash
make docker-up
```

Aguarde ~5 segundos para MLflow ficar pronto. Verifique:

- MLflow UI: http://localhost:5001
- API: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

**Nota:** o MLflow server roda com `--serve-artifacts` para permitir treino local
com tracking no Docker (fluxo hibrido).

---

### 2️⃣ Treinar o Modelo DENTRO do Container

No fluxo Docker puro, treine sempre dentro do container:

```bash
make docker-train EPOCHS=5
```

Equivalente ao comando direto:

```bash
docker compose exec -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e PYTHONPATH=/app \
  api python src/models/train.py --epochs 5
```

**O que acontece:**

- ✅ Modelo treina conectado ao MLflow Server (Docker)
- ✅ Artifacts (modelos, preprocessor) salvos em `/app/mlruns` (mapeado para `./mlruns`)
- ✅ SQLite DB atualizado em `/app/mlflow.db` (mapeado para `./mlflow.db`)
- ✅ Modelo registrado no Model Registry como versão

**Saída esperada:**

```
Registered model 'MLP_Focal_KFold_Script' already exists. Creating a new version of this model...
Created version 'N' of model 'MLP_Focal_KFold_Script'.
🏃 View run MLP_Focal_KFold_Production at: http://mlflow:5000/#/experiments/1/runs/...
🧪 View experiment at: http://mlflow:5000/#/experiments/1
```

---

### 3️⃣ Configurar Alias 'production'

Após treinar, configure o modelo como produção:

```bash
docker compose exec api python << 'EOF'
import mlflow
client = mlflow.tracking.MlflowClient("http://mlflow:5000")
versions = client.search_model_versions('name="MLP_Focal_KFold_Script"')
if versions:
    latest = sorted(versions, key=lambda x: int(x.version))[-1]
    client.set_registered_model_alias("MLP_Focal_KFold_Script", "production", latest.version)
    print(f"✅ Alias 'production' → versão {latest.version}")
else:
    print("❌ Nenhuma versão encontrada")
EOF
```

---

### 4️⃣ Reiniciar a API

A API carrega o modelo durante startup. Reinicie para pegar a nova versão com alias:

```bash
docker compose restart api
```

Verifique nos logs:

```bash
make docker-logs
```

Busque por: `"Modelos carregados com sucesso"` ✅

Se vir erro sobre alias não encontrado, volte ao passo 3.

---

### 5️⃣ Testar a Predição

```bash
curl -s http://localhost:8000/health | jq .
```

Resposta esperada (model_loaded: true):

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "MLP_Focal_KFold_Script",
  "model_version": "N",
  "loaded_at": "2026-05-03T16:03:16.747707+00:00",
  "uptime_seconds": 45.2,
  "mlflow_uri": "http://mlflow:5000"
}
```

---

### 6️⃣ Down e Up (Persistência)

Modelos persistem em `./mlruns` e `./mlflow.db` (bind mounts):

```bash
docker compose down
docker compose up -d
sleep 5
docker compose logs api | tail -20
```

Logs mostram: `"Modelos carregados com sucesso"` ✅ (automático, sem retreinar)

---

## 🔄 Atalho: Treinar + Alias + Reiniciar

Use o target do Makefile:

```bash
make docker-train-full EPOCHS=5
```

## ⏱️ Parametros de EPOCHS

Formas suportadas (de acordo com o Makefile):

```bash
# Docker (treino dentro do container)
make docker-train EPOCHS=50

# One-step (treino + alias + restart)
make docker-train-full EPOCHS=50

# Hibrido (treino local -> registra no Docker)
make train-and-sync-docker EPOCHS=50
```

Ou rode manualmente:

```bash
# 1. Treinar
docker compose exec -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e PYTHONPATH=/app \
  api python src/models/train.py --epochs 5 && \

# 2. Alias
docker compose exec api python << 'EOF'
import mlflow
client = mlflow.tracking.MlflowClient("http://mlflow:5000")
versions = client.search_model_versions('name="MLP_Focal_KFold_Script"')
if versions:
    latest = sorted(versions, key=lambda x: int(x.version))[-1]
    client.set_registered_model_alias("MLP_Focal_KFold_Script", "production", latest.version)
    print(f"✅ Versão {latest.version} como production")
EOF
&& \

# 3. Reiniciar
docker compose restart api && sleep 5 && \
docker compose logs api | grep "Modelos carregados"
```

---

## ⚠️ Erros Comuns & Soluções

### ❌ "Modelos não carregados. Execute load_model_artifacts primeiro"

**Causa:** Alias 'production' não configurado
**Solução:** Rode o passo 3 (Configurar Alias)

### ❌ "RESOURCE_DOES_NOT_EXIST: Registered Model not found"

**Causa:** Modelo não foi treinado ou está em outra versão
**Solução:** Rode passo 2 (Treinar) → passo 3 (Alias)

### ❌ "Read-only file system: /app"

**Causa:** Treino local apontando para o MLflow do Docker sem artifact serving
**Solução:** Garanta `--serve-artifacts` no `docker-compose.yml` e recrie o servico:
`docker compose up -d --force-recreate mlflow`

### ❌ "Connection refused" ao treinar

**Causa:** MLflow container não subiu
**Solução:** Verifique com `docker compose ps` e `docker compose logs mlflow`

### ❌ Modelo carrega, mas after `down/up` sumiu

**Causa:** Volumes não configurados como bind mounts
**Solução:** Verifique `docker-compose.yml` tem `./mlruns:/app/mlruns` e `./mlflow.db:/app/mlflow.db`

---

## 📊 Comandos Úteis

```bash
# Status dos containers
make docker-ps

# Ver logs da API
make docker-logs

# Limpar tudo e recomeçar
make docker-clean

# Acessar MLflow UI (no navegador)
open http://localhost:5001

# Treinar no Docker com epochs custom
make docker-train EPOCHS=50

# Treino hibrido (GPU local -> registra no Docker)
make train-and-sync-docker EPOCHS=50

# Ver versões do modelo no registry
docker compose exec api python << 'EOF'
import mlflow
client = mlflow.tracking.MlflowClient("http://mlflow:5000")
versions = client.search_model_versions('name="MLP_Focal_KFold_Script"')
for v in sorted(versions, key=lambda x: int(x.version)):
    print(f"Versão {v.version}: stage={v.current_stage}, status={v.status}")
EOF

# Deletar e recomeçar do zero
docker compose down -v && rm -rf mlruns mlflow.db && docker compose up -d
```

---

## 📝 Resumo da Stack

| Serviço                | URL                   | Porta | Propósito                |
| ----------------------- | --------------------- | ----- | ------------------------- |
| **MLflow Server** | http://localhost:5001 | 5001  | Model Registry + Tracking |
| **FastAPI**       | http://localhost:8000 | 8000  | Inference API             |
| **Prometheus**    | http://localhost:9090 | 9090  | Métricas                 |
| **Grafana**       | http://localhost:3000 | 3000  | Dashboards                |

---

## 🎯 Fluxo Recomendado (Diário)

```bash
# 1. Acordou, inicia tudo
make docker-up

# 2. Treina modelo novo
make docker-train EPOCHS=100

# 3. Configura como produção
docker compose exec api python << 'EOF'
import mlflow
client = mlflow.tracking.MlflowClient("http://mlflow:5000")
versions = client.search_model_versions('name="MLP_Focal_KFold_Script"')
if versions:
    latest = sorted(versions, key=lambda x: int(x.version))[-1]
    client.set_registered_model_alias("MLP_Focal_KFold_Script", "production", latest.version)
EOF

# 4. Reinicia API
docker compose restart api

# 5. Verifica
curl http://localhost:8000/health | jq .model_loaded

# 6. Vai embora
docker compose down
```

---

## ✅ Checklist Final

- [ ] `docker compose up -d` → tudo green no `docker compose ps`
- [ ] `make docker-logs` → sem erros críticos
- [ ] Treinou modelo via `docker compose exec ... python src/models/train.py`
- [ ] Configurou alias com MLflow client
- [ ] `docker compose restart api` → logs mostram "Modelos carregados com sucesso"
- [ ] `curl http://localhost:8000/health` → `model_loaded: true`
- [ ] `docker compose down` + `docker compose up -d` → modelo carrega automaticamente

---

**Sucesso! 🎉 Modelo está em produção no Docker.**
