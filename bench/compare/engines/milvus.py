"""Milvus standalone adapter: official pymilvus ORM API (gRPC).

Milvus' recommended bulk path: insert into a collection without an index, then
flush -> create_index(HNSW) -> wait for index build -> load. All of that is
timed as the build phase by run.py.
"""

from __future__ import annotations

import itertools
import time

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from .base import COLLECTION, Engine, Searcher, env

_METRIC = {"euclidean": "L2", "cosine": "COSINE"}
_alias_seq = itertools.count()


def _connect() -> str:
    alias = f"bench{next(_alias_seq)}"
    connections.connect(alias=alias, uri=env("MILVUS_URI", "http://milvus:19530"), timeout=120)
    return alias


def _search(col: Collection, metric: str, vec, k, ef):
    res = col.search(
        data=[vec.tolist()],
        anns_field="vector",
        param={"metric_type": metric, "params": {"ef": max(ef or 64, k)}},
        limit=k,
        output_fields=[],
        consistency_level="Eventually",  # data is static and flushed before queries
    )
    return [int(h.id) for h in res[0]]


class _MSearcher(Searcher):
    def __init__(self, metric, ef_ref):
        self.alias = _connect()
        self.col = Collection(COLLECTION, using=self.alias)
        self.metric = metric
        self._ef_ref = ef_ref

    def search(self, vec, k):
        return _search(self.col, self.metric, vec, k, self._ef_ref())

    def close(self):
        connections.disconnect(self.alias)


class Milvus(Engine):
    name = "milvus"

    def __init__(self):
        super().__init__()
        deadline = time.time() + 180
        while True:
            try:
                self.alias = _connect()
                break
            except Exception:
                if time.time() > deadline:
                    raise
                time.sleep(2)
        self.col = None

    def setup(self, dim, metric, m, ef_construction):
        self.metric = _METRIC[metric]
        self.index_params = {
            "index_type": "HNSW",
            "metric_type": self.metric,
            "params": {"M": m, "efConstruction": ef_construction},
        }
        if utility.has_collection(COLLECTION, using=self.alias):
            utility.drop_collection(COLLECTION, using=self.alias)
        schema = CollectionSchema(
            [
                FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=dim),
            ]
        )
        self.col = Collection(COLLECTION, schema=schema, using=self.alias, shards_num=1)

    def insert(self, ids, vectors):
        self.col.insert([[int(i) for i in ids], vectors.tolist()])

    def finish_build(self):
        timings = {}
        t = time.perf_counter()
        self.col.flush()
        timings["flush_s"] = time.perf_counter() - t
        t = time.perf_counter()
        self.col.create_index("vector", self.index_params)
        utility.wait_for_index_building_complete(COLLECTION, using=self.alias)
        timings["create_index_s"] = time.perf_counter() - t
        t = time.perf_counter()
        self.col.load()
        utility.wait_for_loading_complete(COLLECTION, using=self.alias)
        timings["load_s"] = time.perf_counter() - t
        timings["num_entities"] = self.col.num_entities
        return timings

    def search(self, vec, k):
        return _search(self.col, self.metric, vec, k, self.ef)

    def make_searcher(self):
        return _MSearcher(self.metric, lambda: self.ef)

    def version(self):
        try:
            return utility.get_server_version(using=self.alias)
        except Exception:
            return "unknown"

    def teardown(self):
        try:
            utility.drop_collection(COLLECTION, using=self.alias)
        finally:
            connections.disconnect(self.alias)
