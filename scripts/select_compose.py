#!/usr/bin/env python3
"""Auto-select docker-compose file based on GPU availability."""

import subprocess
import sys
import torch

def detect_gpu():
    """Detect GPU type."""
    if torch.cuda.is_available():
        return "nvidia"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def main():
    gpu_type = detect_gpu()
    
    if gpu_type == "nvidia":
        compose_file = "docker-compose.gpu.yml"
        print("✅ NVIDIA GPU detectada → usando docker-compose.gpu.yml")
    else:
        compose_file = "docker-compose.yml"
        print(f"✅ {gpu_type.upper()} detectado → usando docker-compose.yml (sem GPU)")
    
    # Retorna o arquivo
    print(compose_file)

if __name__ == "__main__":
    main()
