import faiss, sqlite3, json, numpy as np, torch
from pathlib import Path
from typing import List, Dict, Any
import threading

class VectorStore:
    def __init__(self, dim: int, index_path: Path, meta_path: Path):
        self._lock = threading.RLock()
        self.dim = dim
        self.index_path = Path(index_path)
        self.db = sqlite3.connect(str(meta_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row  # (optional) dict type rows
        self._init_meta_table()

        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            assert self.index.d == dim, "dim mismatch"
        else:
            self.index = faiss.IndexFlatIP(dim)

    def _init_meta_table(self) -> None:
        with self._lock:
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

        # ---- 1) Load existing keys from SQLite ----
        cur = self.db.cursor()
        cur.execute("SELECT payload FROM meta")
        existing_keys = set()
        for (payload,) in cur.fetchall():
            try:
                m = json.loads(payload)
                key = (
                    m.get("source"),
                    int(m.get("page", -1)),
                    m.get("sha_image"),   # can be None if there was no one
                    m.get("sha_text"),
                )
            except Exception:
                continue
            existing_keys.add(key)

        # ---- 2) Filter what element are actually new ----
        keep_idx = []
        new_metas = []
        for i, m in enumerate(metas):
            key = (
                m.get("source"),
                int(m.get("page", -1)),
                m.get("sha_image"),
                m.get("sha_text"),
            )
            # Fallback just inca se the meta have no hashes
            if key[2] is None and key[3] is None:
                key = (m.get("source"), int(m.get("page", -1)), None, None)

            if key in existing_keys:
                continue
            existing_keys.add(key)
            keep_idx.append(i)
            new_metas.append(m)

        if not keep_idx:
            return []  # there is no new elements to add

        # ---- 3) Prepare vectors (just the new ones) ----
        if isinstance(vecs, torch.Tensor):
            vecs_np = vecs.detach().cpu().to(torch.float32).numpy()[keep_idx]
        else:
            vecs_np = vecs.astype("float32", copy=False)[keep_idx]

        assert vecs_np.shape[1] == self.dim
        faiss.normalize_L2(vecs_np)

        # ---- 4) Add to FAISS with implicit ids (start..start+n-1) ----
        start = self.index.ntotal
        n = vecs_np.shape[0]
        ids = list(range(start, start + n))
        self.index.add(vecs_np)

        # ---- 5) Insert alinged metadata with those ids ----
        rows = []
        for off, m in enumerate(new_metas):
            rid = start + off
            src = m.get("source")
            page = int(m.get("page", -1))
            payload = json.dumps(m, ensure_ascii=False)
            rows.append((rid, src, page, payload))

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
        with self._lock:
            cur = self.db.cursor()
            cur.execute(f"SELECT id, payload FROM meta WHERE id IN ({qmarks})", ids)
            rows = cur.fetchall()

        out = {int(r["id"] if isinstance(r, sqlite3.Row) else r[0]):
           json.loads(r["payload"] if isinstance(r, sqlite3.Row) else r[1])
           for r in rows}
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
    
    def stats(self) -> dict:
        n_faiss = self.index.ntotal
        n_sql = self.db.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
        return {"faiss": int(n_faiss), "sqlite": int(n_sql), "dim": int(self.index.d)}

    def reset(self):
        # just closes; to delete files, outside (more explicit)
        if self.index_path.exists(): self.index_path.unlink()
        # the sqlite is deleted deleting the file outside


    def save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        with self._lock:
            self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()
    
