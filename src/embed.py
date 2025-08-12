import torch
from PIL import Image
from open_clip import create_model_from_pretrained, tokenize
from utils.device import get_device

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
    
    ret = create_model_from_pretrained(model_name, weights)
    if len(ret) == 3:
        _MODEL, _PREPROCESS, _ = ret # Older API version
    elif len(ret) == 2:
        _MODEL, _PREPROCESS = ret
    else:
        raise RuntimeError("Unexpected return from create_model_from_pretrained")
    
    _MODEL.eval().to(_DEVICE) # Inference mode

# --------------- L2 Normalization --------------------------
def _l2_normalize(t: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(t, dim=1) # NxD size -> norm=1 row by row

# --------------- EMBED TEXT -------------------------------
def embed_text(text_list, batch_size=32):
    """
    text_list: List[str] -> return: torch:Tensor (N x 768)
    """
    _lazy_load()
    all_vecs = []

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        tokens = tokenize(batch).to(_DEVICE) # converts to CLIPS ids

        with torch.no_grad():
            vec = _MODEL.encode_text(tokens)
        all_vecs.append(vec.cpu()) # to cpu to be processed

    vecs = torch.cat(all_vecs)
    return _l2_normalize(vecs) # N x 768

# --------------- EMBED IMAGES ----------------------------
def embed_images(pil_images, batch_size=16):
    """
    pil_images: List[PIL.Image] -> torch.Tensor (N x 768)
    """
    _lazy_load()
    tensors = []

    for img in pil_images:
        tensors.append(_PREPROCESS(img)) # resize, center-crop, normalize
    
    vecs = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i:i+batch_size]).to(_DEVICE)
        with torch.no_grad():
            v = _MODEL.encode_image(batch)
        vecs.append(v.cpu())
    
    vecs = torch.cat(vecs)
    return _l2_normalize(vecs)

# -------------------- EMBED PAGE -------------------------
def embed_page(text: str, image: Image.Image) -> torch.Tensor:
    """
    Fusion text+image; simplest: concat (768+768 = 1536).
    Other option: mean (768)
    """
    v_text = embed_text([text])[0] # 1 x 768 -> tensor 768
    v_image = embed_images([image])[0]
    return torch.cat([v_text, v_image]) # 1536-dim