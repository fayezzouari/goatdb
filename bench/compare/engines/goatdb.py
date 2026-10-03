"""goatdb adapter: REST/JSON over HTTP keep-alive (goatdb has no gRPC/binary API)."""

from __future__ import annotations

import json
import time

import numpy as np
import requests

from .base import COLLECTION, Engine, Searcher, env


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["Content-Type"] = "application/json"
    adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=4)
    s.mount("http://", adapter)
    return s


class _GoatSearcher(Searcher):
    def __init__(self, base: str, ef_ref):
        self.base = base
        self.s = _session()
        self._ef_ref = ef_ref

    def search(self, vec: np.ndarray, k: int):
        return _search(self.s, self.base, vec, k, self._ef_ref())

    def close(self):
        self.s.close()


def _search(s: requests.Session, base: str, vec: np.ndarray, k: int, ef: int | None):
    body = {"embeddings": vec.tolist(), "top_k": k, "include_metadata": False}
    if ef is not None:
        body["ef"] = ef
    r = s.post(f"{base}/collections/{COLLECTION}/search", data=json.dumps(body))
    if r.status_code != 200:
        raise RuntimeError(f"goatdb search failed: {r.status_code} {r.text[:200]}")
    return [int(x["id"]) for x in r.json()["results"]]


class GoatDB(Engine):
    name = "goatdb"

    def __init__(self) -> None:
        super().__init__()
        self.base = env("GOATDB_URL", "http://goatdb:8080").rstrip("/")
        self.s = _session()

    def _wait_healthy(self, timeout: float = 120) -> None:
        deadline = time.time() + timeout
        while True:
            try:
                if self.s.get(f"{self.base}/health", timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            if time.time() > deadline:
                raise RuntimeError(f"goatdb at {self.base} not healthy after {timeout}s")
            time.sleep(1)

    def setup(self, dim, metric, m, ef_construction):
        self._wait_healthy()
        self.s.delete(f"{self.base}/collections/{COLLECTION}")
        body = {
            "name": COLLECTION,
            "dim": dim,
            "metric": metric,  # "euclidean" | "cosine"
            "index_type": "hnsw",
            # Honoured by servers that support tunable HNSW params (feat/index-params);
            # older servers ignore unknown JSON fields.
            "hnsw": {"m": m, "ef_construction": ef_construction, "ef_search": 128},
        }
        r = self.s.post(f"{self.base}/collections", data=json.dumps(body))
        if r.status_code not in (200, 201):
            raise RuntimeError(f"goatdb create collection failed: {r.status_code} {r.text}")

    def insert(self, ids, vectors):
        body = {"vectors": [{"id": str(int(i)), "embeddings": v} for i, v in zip(ids, vectors.tolist())]}
        r = self.s.post(f"{self.base}/collections/{COLLECTION}/vectors/batch", data=json.dumps(body))
        if r.status_code not in (200, 201):
            raise RuntimeError(f"goatdb batch insert failed: {r.status_code} {r.text[:300]}")

    def finish_build(self):
        # goatdb's HNSW is built incrementally during insert; the batch endpoint
        # returns only after the vectors are in the graph, so nothing to wait for.
        r = self.s.get(f"{self.base}/collections/{COLLECTION}")
        return {"collection": r.json() if r.ok else r.text}

    def search(self, vec, k):
        return _search(self.s, self.base, vec, k, self.ef)

    def make_searcher(self):
        return _GoatSearcher(self.base, lambda: self.ef)

    def version(self):
        try:
            r = self.s.get(f"{self.base}/health", timeout=2)
            return r.json().get("version", "unknown")
        except Exception:
            return "unknown"

    def teardown(self):
        try:
            self.s.delete(f"{self.base}/collections/{COLLECTION}")
        finally:
            self.s.close()
