"""pgvector adapter: psycopg 3 + pgvector-python.

Strategy: load all rows first (binary COPY, one COPY per 1000-row batch), then
CREATE INDEX ... USING hnsw. Building after load is pgvector's documented
recommendation and is much faster than maintaining the graph per insert.
ef is per session: SET hnsw.ef_search.
"""

from __future__ import annotations

import time

import psycopg
from pgvector.psycopg import register_vector

from .base import Engine, Searcher, env

TABLE = "items"
_OPS = {"euclidean": ("vector_l2_ops", "<->"), "cosine": ("vector_cosine_ops", "<=>")}


def _connect() -> psycopg.Connection:
    conn = psycopg.connect(
        host=env("PG_HOST", "pgvector"),
        port=int(env("PG_PORT", "5432")),
        user=env("PG_USER", "postgres"),
        password=env("PG_PASSWORD", "postgres"),
        dbname=env("PG_DB", "postgres"),
        autocommit=True,
    )
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


class _PgSearcher(Searcher):
    def __init__(self, op: str, ef: int | None):
        self.conn = _connect()
        self.sql = f"SELECT id FROM {TABLE} ORDER BY embedding {op} %s LIMIT %s"
        if ef is not None:
            self.conn.execute(f"SET hnsw.ef_search = {int(ef)}")

    def search(self, vec, k):
        rows = self.conn.execute(self.sql, (vec, k), prepare=True).fetchall()
        return [r[0] for r in rows]

    def close(self):
        self.conn.close()


class PgVector(Engine):
    name = "pgvector"

    def __init__(self):
        super().__init__()
        deadline = time.time() + 120
        while True:
            try:
                self.conn = _connect()
                break
            except psycopg.OperationalError:
                if time.time() > deadline:
                    raise
                time.sleep(1)
        self.searcher: _PgSearcher | None = None

    def setup(self, dim, metric, m, ef_construction):
        self.opclass, self.op = _OPS[metric]
        self.m, self.efc = m, ef_construction
        self.conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
        self.conn.execute(f"CREATE TABLE {TABLE} (id int4 PRIMARY KEY, embedding vector({dim}) NOT NULL)")
        # Index-only data path: avoid TOAST compression on the vector column.
        self.conn.execute(f"ALTER TABLE {TABLE} ALTER COLUMN embedding SET STORAGE PLAIN")

    def insert(self, ids, vectors):
        with self.conn.cursor() as cur:
            with cur.copy(f"COPY {TABLE} (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
                copy.set_types(["int4", "vector"])
                for i, v in zip(ids, vectors):
                    copy.write_row((int(i), v))

    def finish_build(self):
        mwm = env("PG_MAINTENANCE_WORK_MEM", "3GB")
        workers = int(env("PG_INDEX_WORKERS", "7"))  # + leader = 8 = container CPU limit
        self.conn.execute(f"SET maintenance_work_mem = '{mwm}'")
        self.conn.execute(f"SET max_parallel_maintenance_workers = {workers}")
        t = time.perf_counter()
        self.conn.execute(
            f"CREATE INDEX {TABLE}_hnsw ON {TABLE} USING hnsw (embedding {self.opclass}) "
            f"WITH (m = {self.m}, ef_construction = {self.efc})"
        )
        index_s = time.perf_counter() - t
        t = time.perf_counter()
        self.conn.execute(f"VACUUM ANALYZE {TABLE}")
        vacuum_s = time.perf_counter() - t
        size = self.conn.execute(
            f"SELECT pg_relation_size('{TABLE}_hnsw'), pg_total_relation_size('{TABLE}')"
        ).fetchone()
        return {
            "create_index_s": index_s,
            "vacuum_analyze_s": vacuum_s,
            "maintenance_work_mem": mwm,
            "parallel_workers": workers,
            "index_bytes": size[0],
            "table_total_bytes": size[1],
        }

    def set_ef(self, ef):
        self.ef = ef
        if self.searcher is not None:
            self.searcher.close()
        self.searcher = _PgSearcher(self.op, ef)

    def search(self, vec, k):
        if self.searcher is None:
            self.searcher = _PgSearcher(self.op, self.ef)
        return self.searcher.search(vec, k)

    def make_searcher(self):
        return _PgSearcher(self.op, self.ef)

    def version(self):
        row = self.conn.execute(
            "SELECT version(), (SELECT extversion FROM pg_extension WHERE extname = 'vector')"
        ).fetchone()
        return f"pgvector {row[1]} / {row[0].split(',')[0]}"

    def teardown(self):
        try:
            if self.searcher is not None:
                self.searcher.close()
            self.conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
        finally:
            self.conn.close()
