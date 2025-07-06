"""User-facing retrieval API."""
from typing import List
from src.embed import embed_text
from src.vector_store import VectorStore

_STORE: VectorStore | None = None  # global singleton for demo

class Hit(dict):
    """{score, source, location, text}"""

def init_store(store: VectorStore):
    global _STORE
    _STORE = store

def retrieve(query: str, k: int = 5) -> List[Hit]:
    assert _STORE is not None, "VectorStore not initialised"
    # TODO: search _STORE for top-k and return structured hits
    raise NotImplementedError
