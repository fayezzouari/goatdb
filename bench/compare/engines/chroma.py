"""Chroma adapter: official chromadb-client (HTTP).

HNSW parameters are collection-level in Chroma. On Chroma >= 1.0 they are set
through the collection `configuration` (the successor of the legacy
`hnsw:M` / `hnsw:construction_ef` / `hnsw:search_ef` metadata keys, which the
client still maps onto it). search_ef has no per-query override, so set_ef()
modifies the collection configuration before each sweep step.

Set CHROMA_LEGACY_METADATA=1 to use the legacy metadata keys instead (Chroma < 1.0).
"""

from __future__ import annotations

import time

import chromadb

from .base import COLLECTION, Engine, Searcher, env

_SPACE = {"euclidean": "l2", "cosine": "cosine"}


def _client():
    return chromadb.HttpClient(host=env("CHROMA_HOST", "chroma"), port=int(env("CHROMA_PORT", "8000")))


def _search(col, vec, k):
    res = col.query(query_embeddings=[vec], n_results=k, include=[])
    return [int(x) for x in res["ids"][0]]


class _CSearcher(Searcher):
    def __init__(self):
        self.c = _client()
        self.col = self.c.get_collection(COLLECTION, embedding_function=None)

    def search(self, vec, k):
        return _search(self.col, vec, k)


class Chroma(Engine):
    name = "chroma"

    def __init__(self):
        super().__init__()
        deadline = time.time() + 120
        while True:
            try:
                self.c = _client()
                self.c.heartbeat()
                break
            except Exception:
                if time.time() > deadline:
                    raise
                time.sleep(1)
        self.legacy = env("CHROMA_LEGACY_METADATA", "0") == "1"
        self.col = None

    def setup(self, dim, metric, m, ef_construction):
        self.dim = dim
        try:
            self.c.delete_collection(COLLECTION)
        except Exception:
            pass
        if self.legacy:
            self.col = self.c.create_collection(
                COLLECTION,
                metadata={
                    "hnsw:space": _SPACE[metric],
                    "hnsw:M": m,
                    "hnsw:construction_ef": ef_construction,
                    "hnsw:search_ef": 128,
                },
                embedding_function=None,
            )
        else:
            self.col = self.c.create_collection(
                COLLECTION,
                configuration={
                    "hnsw": {
                        "space": _SPACE[metric],
                        "max_neighbors": m,
                        "ef_construction": ef_construction,
                        "ef_search": 128,
                    }
                },
                embedding_function=None,
            )

    def insert(self, ids, vectors):
        self.col.add(ids=[str(int(i)) for i in ids], embeddings=vectors)

    def finish_build(self):
        # Single-node Chroma applies its write-ahead log to the HNSW segment
        # asynchronously in batches; a query forces any pending log entries to be
        # applied, so count() + one query bounds "index ready".
        n = self.col.count()
        self.col.query(query_embeddings=[[0.0] * self.dim], n_results=1, include=[])
        return {"count": n, "configuration": self.col.configuration_json}

    def set_ef(self, ef):
        self.ef = ef
        if self.legacy:
            # Pre-1.0 servers accepted hnsw:search_ef changes via metadata.
            self.col.modify(metadata={"hnsw:search_ef": ef})
        else:
            self.col.modify(configuration={"hnsw": {"ef_search": ef}})
        # Re-fetch so later queries use a handle carrying the new configuration.
        self.col = self.c.get_collection(COLLECTION, embedding_function=None)
        applied = (self.col.configuration_json or {}).get("hnsw", {}) if not self.legacy else {}
        if not self.legacy and applied.get("ef_search") not in (None, ef):
            raise RuntimeError(f"chroma: ef_search={ef} not applied (got {applied})")

    def search(self, vec, k):
        return _search(self.col, vec, k)

    def make_searcher(self):
        return _CSearcher()

    def version(self):
        try:
            return self.c.get_version()
        except Exception:
            return "unknown"

    def teardown(self):
        try:
            self.c.delete_collection(COLLECTION)
        except Exception:
            pass
