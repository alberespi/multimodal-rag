import faiss, sqlite3, json, numpy as np, torch
from pathlib import Path
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, dim: int, index_path: Path, meta_path: Path):
        self.dim = dim
        self.index_path = Path(index_path)
        self.db = sqlite3.connect(str(meta_path))
        self._init_meta_table()

        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            assert self.index.d == dim, "dim mismatch"
        else:
            self.index = faiss.IndexFlatIP(dim)

    def _init_meta_table(self) -> None:
        cur = self.db.cursor()
        cur.execute("""
                    CREATE TABLE IF NOT EXISTS meta (
                        id INTEGER PRIMARY KEY,
                    source TEXT,
                    page INTEGER,
                    payload TEXT
                    )
                """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_source ON meta(source)")
        self.db.commit()