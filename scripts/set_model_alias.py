#!/usr/bin/env python3
"""Set 'production' alias to latest model version."""

import mlflow
import sys

try:
    client = mlflow.tracking.MlflowClient("http://mlflow:5000")
    versions = client.search_model_versions('name="MLP_Focal_KFold_Script"')
    
    if not versions:
        print("❌ Nenhuma versão do modelo encontrada. Treina primeiro: make docker-train")
        sys.exit(1)
    
    latest = sorted(versions, key=lambda x: int(x.version))[-1]
    client.set_registered_model_alias("MLP_Focal_KFold_Script", "production", latest.version)
    print(f"✅ Alias 'production' → versão {latest.version}")
    
except Exception as e:
    print(f"❌ Erro ao configurar alias: {e}")
    sys.exit(1)
