#!/bin/bash
# Auto-detect GPU type (CUDA, MPS, or CPU)

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detectada"
    echo "nvidia"
    exit 0
fi

# Check for Apple Silicon (MPS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    CHIP=$(uname -m)
    if [[ "$CHIP" == "arm64" ]]; then
        echo "✅ Apple Silicon (MPS) detectado"
        echo "mps"
        exit 0
    fi
fi

# Default to CPU
echo "✅ CPU detectado (sem GPU)"
echo "cpu"
exit 0
