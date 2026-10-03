"""Engine adapters. Imported lazily so the client libraries of engines that are
not being benchmarked do not need to be installed."""

import importlib

ENGINES = {
    "goatdb": "engines.goatdb:GoatDB",
    "qdrant": "engines.qdrant:Qdrant",
    "weaviate": "engines.weaviate:Weaviate",
    "milvus": "engines.milvus:Milvus",
    "pgvector": "engines.pgvector:PgVector",
    "chroma": "engines.chroma:Chroma",
}


def load(name: str):
    try:
        target = ENGINES[name]
    except KeyError:
        raise SystemExit(f"unknown engine {name!r}; choose from {', '.join(ENGINES)}")
    mod, cls = target.split(":")
    return getattr(importlib.import_module(mod), cls)
