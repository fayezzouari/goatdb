"""Weaviate adapter: official weaviate-client v4 (gRPC for import and search, REST for schema).

Weaviate's HNSW `ef` is a collection-level setting (vectorIndexConfig.ef), not a
per-query parameter, so set_ef() reconfigures the collection and waits until the
new value is visible in the schema. Object UUIDs encode the integer row index
(uuid.UUID(int=i)) so no properties have to be fetched at query time.
"""

from __future__ import annotations

import time
import uuid

import weaviate
from weaviate.classes.config import Configure, Reconfigure, VectorDistances
from weaviate.classes.data import DataObject

from .base import Engine, Searcher, env

NAME = "Bench"  # Weaviate collection names must start with an upper-case letter
VEC = "default"
_DIST = {"euclidean": VectorDistances.L2_SQUARED, "cosine": VectorDistances.COSINE}


def _client():
    return weaviate.connect_to_custom(
        http_host=env("WEAVIATE_HOST", "weaviate"),
        http_port=int(env("WEAVIATE_HTTP_PORT", "8080")),
        http_secure=False,
        grpc_host=env("WEAVIATE_HOST", "weaviate"),
        grpc_port=int(env("WEAVIATE_GRPC_PORT", "50051")),
        grpc_secure=False,
        additional_config=weaviate.classes.init.AdditionalConfig(
            timeout=weaviate.classes.init.Timeout(init=60, query=120, insert=600)
        ),
    )


def _search(col, vec, k):
    res = col.query.near_vector(
        near_vector=vec.tolist(),
        limit=k,
        target_vector=VEC,
        return_properties=[],
        return_metadata=None,
        include_vector=False,
    )
    return [o.uuid.int for o in res.objects]


class _WSearcher(Searcher):
    def __init__(self):
        self.c = _client()
        self.col = self.c.collections.get(NAME)

    def search(self, vec, k):
        return _search(self.col, vec, k)

    def close(self):
        self.c.close()


class Weaviate(Engine):
    name = "weaviate"

    def __init__(self):
        super().__init__()
        deadline = time.time() + 120
        while True:
            try:
                self.c = _client()
                if self.c.is_ready():
                    break
                self.c.close()
            except Exception:
                if time.time() > deadline:
                    raise
            time.sleep(1)
        self.col = None

    def setup(self, dim, metric, m, ef_construction):
        if self.c.collections.exists(NAME):
            self.c.collections.delete(NAME)
        self.col = self.c.collections.create(
            NAME,
            vector_config=Configure.Vectors.self_provided(
                name=VEC,
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=_DIST[metric],
                    max_connections=m,
                    ef_construction=ef_construction,
                    ef=128,
                ),
            ),
        )

    def insert(self, ids, vectors):
        objs = [
            DataObject(properties={}, vector={VEC: v}, uuid=uuid.UUID(int=int(i)))
            for i, v in zip(ids, vectors.tolist())
        ]
        res = self.col.data.insert_many(objs)
        if res.has_errors:
            first = next(iter(res.errors.values()))
            raise RuntimeError(f"weaviate insert_many: {len(res.errors)} errors, first: {first}")

    def finish_build(self):
        # Synchronous indexing is the default (ASYNC_INDEXING=false), so the graph is
        # complete when insert_many returns. If async indexing were enabled, wait for
        # the per-shard vector queue to drain.
        deadline = time.time() + 3600
        while time.time() < deadline:
            nodes = self.c.cluster.nodes(collection=NAME, output="verbose")
            shards = [s for n in nodes for s in (n.shards or [])]
            queue = sum((s.vector_queue_length or 0) for s in shards)
            if queue == 0 and all(s.vector_indexing_status == "READY" for s in shards):
                break
            time.sleep(1)
        total = self.col.aggregate.over_all(total_count=True).total_count
        return {"object_count": total}

    def set_ef(self, ef):
        self.ef = ef
        self.col.config.update(
            vector_config=Reconfigure.Vectors.update(
                name=VEC, vector_index_config=Reconfigure.VectorIndex.hnsw(ef=ef)
            )
        )
        deadline = time.time() + 60
        while time.time() < deadline:
            cfg = self.col.config.get()
            if cfg.vector_config[VEC].vector_index_config.ef == ef:
                time.sleep(0.5)  # let the shard pick up the schema change
                return
            time.sleep(0.2)
        raise RuntimeError(f"weaviate: ef={ef} not applied after 60s")

    def search(self, vec, k):
        return _search(self.col, vec, k)

    def make_searcher(self):
        return _WSearcher()

    def version(self):
        try:
            return self.c.get_meta().get("version", "unknown")
        except Exception:
            return "unknown"

    def teardown(self):
        try:
            self.c.collections.delete(NAME)
        finally:
            self.c.close()
