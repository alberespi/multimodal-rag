from __future__ import annotations
from typing import Optional, TypedDict, List
import torch
from PIL import Image

from embed import embed_text, embed_images
from vector_store import VectorStore

class Hit(TypedDict):
    id: int
    score: float
    source: str
    page: int
    text: str
    image: str # filename (e.g. "page_012.png")

_STORE: Optional[VectorStore] = None

def init_store(store: VectorStore) -> None:
    """
    Regists the global VectorStore for the queries.
    """
    global _STORE
    _STORE = store

def _query_vec(
        question: str,
        image: Optional[Image.Image] = None,
        *,
        alpha: float = 0.7, # 1.0 = just text; if image in future, low the weight
) -> torch.Tensor:
    """
    Builds the query vector according to the index dimension.
    - If the index is 1536 (text + image): concatenate [text(768), image(768)].
      For now, if there is no image -> we use 768 zeros instead.
    - If the indez is 768: use only the text embedding.

    Returns 1xdim tensor.
    """
    assert _STORE is not None, "VectorStore not initialized; call init_store() first"

    # 1) Embedding de texto (siempre)
    v_text = embed_text([question])[0]           # shape: (768,)

    # 2) Determinar la dimensión esperada del índice de forma segura
    dim_index = int(getattr(_STORE, "dim", 0))
    dim_text  = int(v_text.numel())

    if dim_index == dim_text:
        # Índice solo-texto (768)
        return v_text.unsqueeze(0)               # shape: (1, 768)

    elif dim_index == 2 * dim_text:
        # Índice texto+imagen (1536)
        if image is not None and alpha < 1.0:
            v_img = embed_images([image])[0]     # (768,)
        else:
            v_img = torch.zeros_like(v_text)     # placeholder if no image
        v = torch.cat([alpha * v_text, (1.0 - alpha) * v_img], dim=0)  # (1536,)
        return v.unsqueeze(0)                    # shape: (1, 1536)

    else:
        # Mensaje de error informativo
        raise ValueError(
            f"Index dim {dim_index} not compatible with text dim {dim_text}. "
            f"Did you create the index with concat (1536) or just texto (768)?"
        )


def retrieve(question: str, k: int = 5, image: Optional[Image.Image] = None, *, alpha = 1.0) -> List[Hit]:
    """
    Runs a query and returns a hit list with practical metadata.
    """
    assert _STORE is not None, "VectorStore not initialized; call init_store() first"

    q = _query_vec(question, image=image, alpha=alpha) # 1xdim
    rows = _STORE.search(q, k=k)[0]                    # dicts list (id, score, meta)

    hits: List[Hit] = []
    for r in rows:
        m = r.get("meta") or {}
        hits.append(Hit(
            id=int(r["id"]),
            score=float(r["score"]),
            source=str(m.get("source", "")),
            page=int(m.get("page", -1)),
            text=str(m.get("text", "")),
            image=str(m.get("image", "")),
        ))
    return hits
