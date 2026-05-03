#!/usr/bin/env python3
"""Detect available GPU (CUDA or MPS) and return device info."""

import torch
import json
import sys

def detect_gpu():
    """Return GPU info as JSON."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "device": None,
        "device_name": None,
    }
    
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["device_name"] = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        info["device"] = "mps"
        info["device_name"] = "Apple Silicon (MPS)"
    else:
        info["device"] = "cpu"
        info["device_name"] = "CPU only"
    
    return info

if __name__ == "__main__":
    info = detect_gpu()
    print(json.dumps(info, indent=2))
