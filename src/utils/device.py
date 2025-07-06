"""Utility to pick the best available torch.device."""
import torch

def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
