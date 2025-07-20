import torch
from PIL import Image
from open_clip import create_model_from_pretrained, tokenize
from src.utils.device import get_device

# -------- GLOBAL VARIABLES --------------
_DEVICE = get_device()  # 'cuda' / 'mps' / 'cpu'
_MODEL = None           # CLIP model
_PREPROCESS = None      # img -> tensor
TXT_DIM = 768           # ViT-L-14 produces 768 dim for text
IMG_DIM = 768           # idem for img

# ---------- LAZY LOAD -------------------
def _lazy_load(model_name="ViT-L-14", weights='openai'):
    """
    Downloads and sets the model in _DEVICE only the first time.
    For Streamlit to launch instantly if embed is not called.
    """
    global _MODEL, _PREPROCESS
    if _MODEL is not None:
        return # Already loaded
    
    _MODEL, _PREPROCESS, _ = create_model_from_pretrained(model_name, weights)
    _MODEL.eval().to(_DEVICE) # Inference mode

