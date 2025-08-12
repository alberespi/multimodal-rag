import faiss, sqlite3, json, numpy as np, torch
from pathlib import Path
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, dim: int, index_path: Path, meta_path: Path):
        self.dim = dim
        self.index_path = Path(index_path)
        self.db = sqlite3.connect(str(meta_path))
        self._init_meta_table()

        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
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

    def add(self, vecs: torch.Tensor | np.ndarray, metas: List[Dict[str, Any]]) -> List[int]:
        assert len(metas) == len(vecs), "vecs/metas length mismatch"
        if isinstance(vecs, torch.Tensor):
            vecs = vecs.detach().cpu().to(torch.float32).numpy()
        else:
            vecs = vecs.astype("float32", copy=False)
        assert vecs.shape[1] == self.dim

        faiss.normalize_L2(vecs)
        start = self.index.ntotal
        n = vecs.shape[0]
        ids = list(range(start, start + n))
        self.index.add(vecs)

        rows = []
        for i, m in enumerate(metas):
            rid = start + i
            src = m.get("source")
            page = int(m.get("page", -1))
            payload = json.dumps(m, ensure_ascii=False)
            rows.append((rid, src, page, payload))
        cur = self.db.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO meta (id, source, page, payload) VALUES (?, ?, ?, ?)",
            rows,
        )
        self.db.commit()
        return ids

    def _fetch_meta_by_ids(self, ids: List[int]) -> Dict[int, Dict[str, Any]]:
        ids = [int(i) for i in ids if i is not None and i >= 0]
        if not ids:
            return {}
        qmarks = ",".join("?" for _ in ids)
        cur = self.db.cursor()
        cur.execute(f"SELECT id, payload FROM meta WHERE id IN ({qmarks})", ids)
        out: Dict[int, Dict[str, Any]] = {}
        for rid, payload in cur.fetchall():
            out[int(rid)] = json.loads(payload)
        return out

    def search(self, query_vecs: torch.Tensor | np.ndarray, k: int = 5):
        if isinstance(query_vecs, torch.Tensor):
            q = query_vecs.detach().cpu().to(torch.float32).numpy()
        else:
            q = query_vecs.astype("float32", copy=False)
        assert q.shape[1] == self.dim
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)

        results = []
        for row_ids, row_scores in zip(I, D):
            metas = self._fetch_meta_by_ids(list(row_ids))
            hits = []
            for id_, score in zip(row_ids, row_scores):
                if id_ == -1:
                    continue
                hits.append({
                    "id": int(id_),
                    "score": float(score),
                    "meta": metas.get(int(id_)),
                })
            results.append(hits)
        return results

    def save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()
    
