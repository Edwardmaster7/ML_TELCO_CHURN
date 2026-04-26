# Clean Architecture Migration: Refactoring the Monolithic Pipeline & Isolating ResNet

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
1. Isolar totalmente o código experimental (ResNet, Entity Embeddings e seus DataLoaders específicos) dentro da pasta `notebooks/resnet/`.
2. Migrar o código de produção que estava acoplado no pacote monolítico `src/ml_telco_churn/` para as devidas camadas de Clean Architecture (`src/data/`, `src/features/`, `src/models/`), sem o uso de underscores nos nomes dos arquivos de script (`pipeline-features.py` não é uma boa prática Python por causa de imports, então usaremos nomes de módulos como `pipeline.py`).
3. Refatorar o script de treinamento de produção para consumir a arquitetura campeã e os módulos limpos de feature engineering.

**Architecture:** A arquitetura ResNet pertencerá somente ao escopo do `notebook 07` via a pasta local `resnet`. O pacote da API ficará purificado em `src/`, fornecendo as bases para o `train.py`.

**Tech Stack:** PyTorch, Python Modules, Scikit-learn.

---

### Task 1: Isolar Resquícios da Arquitetura ResNet em `notebooks/resnet`

**Files:**
- Move: `src/data/datasets.py` -> `notebooks/resnet/datasets.py`
- Move: `src/features/preprocessing.py` -> `notebooks/resnet/preprocessing.py`
- Modify: `notebooks/07_mlp_resnet_embeddings.ipynb`

- [ ] **Step 1: Mover os arquivos de data e pre-processamento do ResNet**

O arquivo `datasets.py` (contém `ChurnEmbeddingDataset`) e `preprocessing.py` (contém `build_ordinal_preprocessor`) foram criados especificamente para suprir a demanda da rede ResNet e não serão usados em produção.

Run:
```bash
mv src/data/datasets.py notebooks/resnet/datasets.py
mv src/features/preprocessing.py notebooks/resnet/preprocessing.py
```

- [ ] **Step 2: Atualizar imports no Notebook 07**

O notebook 07 deve importar os loaders e as features de `resnet` em vez de `src.data` e `src.features` (e também corrigir a importação de `src_resnet` para `resnet` que fizemos localmente). E atualizar a importação do `CONFIG`.

Crie um script em python `fix_nb07_imports.py` para alterar a cell do import e rode-o:
```python
import json
path = 'notebooks/07_mlp_resnet_embeddings.ipynb'
with open(path, 'r') as f: nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            line = line.replace('from src.features.preprocessing', 'from resnet.preprocessing')
            line = line.replace('from src.data.datasets', 'from resnet.datasets')
            line = line.replace('from src_resnet.', 'from resnet.')
            line = line.replace('from src.ml_telco_churn.config', 'from src.config')
            new_source.append(line)
        cell['source'] = new_source

with open(path, 'w') as f: json.dump(nb, f, indent=1)
```

Run:
```bash
python3 fix_nb07_imports.py && rm fix_nb07_imports.py
```

---

### Task 2: Distribuir o Monólito `src/ml_telco_churn` para a Clean Architecture Sem Underscores Extras

**Files:**
- Move: `src/ml_telco_churn/config.py` -> `src/config.py`
- Move: `src/ml_telco_churn/data.py` -> `src/data/loader.py`
- Move: `src/ml_telco_churn/features.py` -> `src/features/pipeline.py`
- Move: `src/models/train_focal_model.py` -> `src/models/trainer.py`
- Move: `src/models/tabular_focal_kfold.py` -> `src/models/architectures.py`
- Delete: `src/ml_telco_churn`

- [ ] **Step 1: Renomear e Mover módulos do monólito e de deep learning**

Vamos remover os underscores extra dos nomes de arquivo, adotando nomes simples de modulo Python (para evitar traços que quebram o import syntax).

Run:
```bash
mv src/ml_telco_churn/config.py src/config.py
mv src/ml_telco_churn/data.py src/data/loader.py
mv src/ml_telco_churn/features.py src/features/pipeline.py
mv src/models/train_focal_model.py src/models/trainer.py
mv src/models/tabular_focal_kfold.py src/models/architectures.py
```

- [ ] **Step 2: Atualizar imports internos e limpar monólito**

Substitua a referência antiga em `src/features/pipeline.py` (antigo features.py) e reserve o `train.py`.

Run:
```bash
python3 -c "with open('src/features/pipeline.py', 'r') as f: t = f.read().replace('from ml_telco_churn.config import CONFIG', 'from src.config import CONFIG');
with open('src/features/pipeline.py', 'w') as f: f.write(t)"

mv src/ml_telco_churn/train.py src/train_temp.py
rm -rf src/ml_telco_churn
```

---

### Task 3: Criar o Script Definitivo de Treino em Produção (`train.py`)

**Files:**
- Create: `src/models/train.py`
- Delete: `src/train_temp.py`

- [ ] **Step 1: Escrever o script de treino final**

Escreva (via Write Tool) o conteúdo para `src/models/train.py`.
O script deve recriar o pipeline de treino unindo a configuração global (`src.config`), carregamento de base de dados (`src.data.loader.load_and_merge_data`), features em OHE (`src.features.pipeline.get_preprocessor`), o modelo PyTorch focal (`src.models.architectures.ChurnMLP`) e a rotina de treino (`src.models.trainer.train_focal_model`).

Em seguida, apague o temporário:
```bash
rm src/train_temp.py
```

- [ ] **Step 2: Validar o fluxo End-to-End**

Execute o pipeline inteiro para garantir que não haja vazamento nem imports quebrados, rodando apenas 2 epochs para confirmar o funcionamento completo de Logging e PyTorch.

Run:
```bash
source .venv/bin/activate && python3 src/models/train.py --epochs 2
```

Expected: Log final exibindo "Modelo campeão (Focal Loss + K-Fold) e métricas registradas no MLflow."

- [ ] **Step 3: Commit da Migração**

```bash
git add src/ notebooks/
git commit -m "refactor: purifica arquitetura solid em src com modelos isolados e remove monólito"
```
