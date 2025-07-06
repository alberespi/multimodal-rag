"""Thin wrapper around a FAISS index plus SQLite metastore."""
from pathlib import Path
from typing import Sequence, Dict, Any
import faiss, sqlite3, torch

class VectorStore:
    def __init__(self, dim: int, index_path: Path, meta_path: Path):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine via inner-product on normalised vecs
        self.index_path = index_path
        self.db = sqlite3.connect(meta_path)
        self._init_meta()

    def _init_meta(self):
        cur = self.db.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS meta (
                   id INTEGER PRIMARY KEY,
                   source TEXT,
                   location TEXT,
                   text TEXT
               )"""
        )
        self.db.commit()

    def add(self, vecs: torch.Tensor, metas: Sequence[Dict[str, Any]]):
        self.index.add(vecs.numpy())
        cur = self.db.cursor()
        for m in metas:
            cur.execute(
                "INSERT INTO meta (source, location, text) VALUES (:source, :location, :text)",
                m,
            )
        self.db.commit()

    # TODO: save/load, search
