"""Qdrant adapter: official qdrant-client, gRPC (prefer_grpc=True, Qdrant's recommended fast path)."""

from __future__ import annotations

import time

from qdrant_client import QdrantClient, models

from .base import COLLECTION, Engine, Searcher, env

_DIST = {"euclidean": models.Distance.EUCLID, "cosine": models.Distance.COSINE}


def _client() -> QdrantClient:
    return QdrantClient(
        host=env("QDRANT_HOST", "qdrant"),
        port=int(env("QDRANT_HTTP_PORT", "6333")),
        grpc_port=int(env("QDRANT_GRPC_PORT", "6334")),
        prefer_grpc=True,
        timeout=600,
    )


def _search(client: QdrantClient, vec, k, ef):
    res = client.query_points(
        collection_name=COLLECTION,
        query=vec.tolist(),
        limit=k,
        search_params=models.SearchParams(hnsw_ef=ef, exact=False),
        with_payload=False,
        with_vectors=False,
    )
    return [int(p.id) for p in res.points]


class _QSearcher(Searcher):
    def __init__(self, ef_ref):
        self.c = _client()
        self._ef_ref = ef_ref

    def search(self, vec, k):
        return _search(self.c, vec, k, self._ef_ref())

    def close(self):
        self.c.close()


class Qdrant(Engine):
    name = "qdrant"

    def __init__(self):
        super().__init__()
        self.c = _client()

    def setup(self, dim, metric, m, ef_construction):
        deadline = time.time() + 120
        while True:
            try:
                self.c.get_collections()
                break
            except Exception:
                if time.time() > deadline:
                    raise
                time.sleep(1)
        if self.c.collection_exists(COLLECTION):
            self.c.delete_collection(COLLECTION)
        self.c.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=_DIST[metric], on_disk=False),
            hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construction),
            # Default optimizer settings: HNSW is built by the background optimizer
            # once segments exceed indexing_threshold. finish_build() waits for it.
        )

    def insert(self, ids, vectors):
        self.c.upsert(
            collection_name=COLLECTION,
            points=models.Batch(ids=[int(i) for i in ids], vectors=vectors.tolist()),
            wait=True,
        )

    def finish_build(self):
        # Qdrant indexes asynchronously. Wait until status is GREEN (optimizers idle)
        # for several consecutive polls; a single green read right after the last
        # upsert can precede the optimizer starting.
        stable, info = 0, None
        while stable < 3:
            time.sleep(1)
            info = self.c.get_collection(COLLECTION)
            if info.status == models.CollectionStatus.GREEN:
                stable += 1
            else:
                stable = 0
        return {
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "segments_count": info.segments_count,
        }

    def search(self, vec, k):
        return _search(self.c, vec, k, self.ef)

    def make_searcher(self):
        return _QSearcher(lambda: self.ef)

    def version(self):
        try:
            return self.c.info().version
        except Exception:
            return "unknown"

    def teardown(self):
        try:
            self.c.delete_collection(COLLECTION)
        finally:
            self.c.close()
