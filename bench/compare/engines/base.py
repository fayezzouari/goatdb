"""Common interface every engine adapter implements.

Lifecycle driven by run.py:

    eng = Engine()
    eng.setup(dim, metric, m, ef_construction)    # create an empty collection/table
    for ids, vecs in batches:                       # ids: np.int64[n], vecs: np.float32[n, dim]
        eng.insert(ids, vecs)
    eng.finish_build()                              # block until the HNSW index is built and queryable
    for ef in efs:
        eng.set_ef(ef)                              # server-side ef where the engine needs it
        eng.search(vec, k)                          # -> list[int] ids, best first
    s = eng.make_searcher()                         # independent connection for a client thread
    eng.teardown()

`metric` is "euclidean" or "cosine". Returned ids must be the integer row index
of the vector in the dataset's train matrix (that is what the ground truth uses).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

COLLECTION = "bench"


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


class Searcher(ABC):
    """Something that can answer a k-NN query. Engines are Searchers themselves."""

    @abstractmethod
    def search(self, vec: np.ndarray, k: int) -> Sequence[int]: ...

    def close(self) -> None:  # pragma: no cover - optional
        pass


class Engine(Searcher):
    name: str = "base"

    def __init__(self) -> None:
        self.ef: int | None = None

    @abstractmethod
    def setup(self, dim: int, metric: str, m: int, ef_construction: int) -> None: ...

    @abstractmethod
    def insert(self, ids: np.ndarray, vectors: np.ndarray) -> None: ...

    def finish_build(self) -> dict:
        """Block until all inserted data is indexed and searchable.

        Returns optional engine-specific details (e.g. indexed counts) that are
        stored in the result JSON.
        """
        return {}

    def set_ef(self, ef: int) -> None:
        """Default: per-request ef; just remember it."""
        self.ef = ef

    def make_searcher(self) -> Searcher:
        """Return a searcher with its own connection, for one client thread.

        Must honour the ef currently set via set_ef(). The default shares the
        adapter itself, which is fine only for thread-safe clients; adapters
        whose client is not thread-safe override this.
        """
        return self

    def version(self) -> str:
        return "unknown"

    def teardown(self) -> None:
        pass
