"""CLIP embedding helpers (text + image)."""
from typing import List
import torch
from open_clip import create_model_from_pretrained, tokenize
from PIL import Image

from src.utils.device import get_device

_DEVICE = get_device()
_MODEL, _PREPROCESS, _ = create_model_from_pretrained("ViT-L/14")
_MODEL.eval().to(_DEVICE)

def embed_text(texts: List[str]) -> torch.Tensor:
    with torch.no_grad():
        tokens = tokenize(texts).to(_DEVICE)
        return _MODEL.encode_text(tokens).cpu()

def embed_images(images: List[Image.Image]) -> torch.Tensor:
    with torch.no_grad():
        tensors = torch.stack([_PREPROCESS(img) for img in images]).to(_DEVICE)
        return _MODEL.encode_image(tensors).cpu()
