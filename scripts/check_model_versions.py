#!/usr/bin/env python3
"""List all model versions in registry."""

import mlflow

try:
    client = mlflow.tracking.MlflowClient("http://mlflow:5000")
    versions = client.search_model_versions('name="MLP_Focal_KFold_Script"')
    
    if not versions:
        print("❌ Nenhuma versão encontrada")
    else:
        print("\n")
        for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
            print(f"  Versão {v.version}: stage={v.current_stage}, status={v.status}")
        print("\n")
        
except Exception as e:
    print(f"❌ Erro: {e}")
