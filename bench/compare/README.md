# goatdb vs other vector databases

This directory holds a reproducible benchmark of goatdb against **Qdrant,
Weaviate, Milvus (standalone), pgvector and Chroma**. The setup tries to give
every engine the same conditions: the same HNSW parameters, the same resource
limits and the same client placement, with each engine loaded and queried
through its own recommended path.

```
bench/compare/
├── compose.yml          one service per engine (compose profiles) + client
├── client/              client image (python:3.12-slim + official clients, pinned)
├── engines/             one adapter per engine, common interface in base.py
├── run.py               one engine × one dataset → results/<engine>__<dataset>.json
├── run_all.sh           all engines × datasets, sequentially, volumes wiped in between
├── stats.py             merges docker-stats memory samples + volume sizes into the JSON
├── plot.py              recall/QPS curves, resource bars, results/summary.md
└── download.sh          fetches the ann-benchmarks HDF5 files into data/
```

## Methodology

### Datasets

These are the [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) HDF5 files. Each file holds
`train` (the vectors to index), `test` (the queries) and `neighbors` (the exact
top-100 ground truth).

| dataset | vectors | dim | queries | metric | file size |
|---|---:|---:|---:|---|---:|
| `sift-128-euclidean` | 1,000,000 | 128 | 10,000 | L2 | ~501 MB |
| `glove-100-angular` | 1,183,514 | 100 | 10,000 | cosine | ~463 MB |

For glove, every engine uses its **cosine** metric. `run.py` also L2-normalizes
both train and test vectors before inserting them, so engines that normalize
internally and engines that do not receive identical data. The ann-benchmarks
angular ground truth stays valid because cosine ordering does not depend on
vector norms. Vector *i* gets id *i* (goatdb and Chroma use `str(i)`, Weaviate
uses `UUID(int=i)`), and recall is computed against `neighbors` directly.

### Index parameters (identical everywhere)

| | value | goatdb | Qdrant | Weaviate | Milvus | pgvector | Chroma |
|---|---|---|---|---|---|---|---|
| index | HNSW | `index_type: hnsw` | default | `hnsw` | `HNSW` | `USING hnsw` | default |
| M | 16 | `hnsw.m` | `m` | `maxConnections` | `M` | `m` | `max_neighbors` (`hnsw:M`) |
| efConstruction | 200 | `hnsw.ef_construction` | `ef_construct` | `efConstruction` | `efConstruction` | `ef_construction` | `ef_construction` (`hnsw:construction_ef`) |
| search ef | sweep 16, 32, 64, 128, 256, 512 | per request `ef` | per request `hnsw_ef` | **collection** `ef` (reconfigured) | per request `params.ef` | per session `SET hnsw.ef_search` | **collection** `ef_search` (modified) |
| top_k | 10 | | | | | | |

Weaviate and Chroma have no per-query ef. Their adapters change the collection
setting before each step of the sweep and wait until the new value is visible
(Weaviate: `config.update(vector_config=Reconfigure.Vectors.update(...))` and
then poll `config.get()`; Chroma ≥ 1.0: `collection.modify(configuration={"hnsw":
{"ef_search": ef}}`). Chroma's 1.x `configuration` API replaces the legacy
`hnsw:*` metadata keys, and the client maps those keys onto it. Set
`CHROMA_LEGACY_METADATA=1` to use the metadata keys instead. Weaviate's dynamic
ef is disabled implicitly because ef is always set explicitly.

### Load phase

`run.py` inserts vectors in batches of 1,000 through each engine's batch API.
The time it records counts as loading only once the index is **built and
queryable**, so each adapter's `finish_build()` blocks in the way that engine
documents:

| engine | insert path | `finish_build()` |
|---|---|---|
| goatdb | `POST /collections/{name}/vectors/batch` (JSON) | none (graph updated synchronously per batch) |
| Qdrant | `upsert(Batch, wait=True)` over gRPC | poll until collection status is **green** (optimizers idle) for 3 consecutive seconds |
| Weaviate | `data.insert_many` (gRPC batch) | wait until the shard vector queue is 0 and status is `READY` (sync indexing is the default) |
| Milvus | `Collection.insert` (no index yet) | `flush()` → `create_index(HNSW)` → `wait_for_index_building_complete` → `load()` |
| pgvector | binary `COPY` of 1,000 rows per batch into an unindexed table | `CREATE INDEX ... USING hnsw` with `maintenance_work_mem=3GB`, 7 parallel workers + leader = 8 CPUs, then `VACUUM ANALYZE` |
| Chroma | `collection.add` | `count()` + 1 query (forces any pending WAL entries into the HNSW segment) |

For pgvector the index is **built after the load**. pgvector recommends this
for bulk loads, and it is also how Milvus works. Building the index before the
load would make every insert pay graph maintenance and would be much slower.
The result JSON records `insert_s`, `build_s` and `total_load_s`, which is the
fair number to compare across engines. Some engines index during insert and
others index afterwards, so only the total is meaningful.

### Query phase

* **Serial sweep**: one client and one connection. For each ef the client
  runs 100 warm-up queries and then N timed queries (`--queries`, 1,000–10,000).
  It records recall@10, QPS (N / wall time), and p50/p95/p99/mean latency.
  Latency is measured round trip in the client, including serialization.
* **Concurrent**: the client picks the **smallest ef with recall@10 ≥ 0.95**
  (or the best ef if no ef reaches 0.95, flagged as `reached_target: false`).
  8 client threads, each with its own connection, then send queries for 20 s,
  and the client records total QPS.

### Resources and fairness

* Every engine's main container gets `cpus: "8"`, `mem_limit: 8g` and no swap.
  For Go engines (goatdb, Weaviate), `GOMAXPROCS=8` is set because Go < 1.25
  ignores cgroup CPU quotas.
* Milvus also needs etcd and MinIO. These follow Milvus' official compose file
  and have no limits. Their memory **is** included in Milvus' reported peak (the
  sum across the three containers at each sampling instant).
* **Only one engine runs at a time.** `run_all.sh` starts it, benchmarks it,
  measures it, then runs `down -v` (removing its volumes) before the next engine
  starts.
* **The client runs in a container on the same compose network** (`cpus: "4"`,
  `mem_limit: 3g`). On macOS, Docker Desktop's host port forwarding adds
  significant per-request latency, so no ports are published at all.
* **Peak memory** comes from sampling `docker stats` about once per second for
  the whole run (load and queries). On Linux cgroups this figure is
  RSS + kernel memory minus inactive page cache.
* **Disk** is `du -sb` of the engine's data volume(s) after the load, while the
  data is still present (`run.py --keep`).
* **Client protocols**: each engine uses its official Python client on its
  recommended transport: gRPC for Qdrant (`prefer_grpc=True`), Weaviate (v4
  client) and Milvus (pymilvus); the PostgreSQL wire protocol for pgvector
  (psycopg 3, prepared statement, binary COPY); and HTTP/JSON for Chroma.
  **goatdb only has a REST/JSON API**, so it is driven through a
  keep-alive `requests.Session`. JSON encoding and decoding of float vectors is
  part of goatdb's measured cost. That reflects how goatdb is used today, but
  keep it in mind when reading latency numbers.
* Python client threads share the GIL. At high QPS the client can become the
  bottleneck for the fastest engines. Fixed client CPUs (4) make this the same
  for every engine, but absolute concurrent QPS is a lower bound.

### Pinned versions

| component | image / package |
|---|---|
| goatdb | built from this repo (`build: ../..`) |
| Qdrant | `qdrant/qdrant:v1.19.1`, `qdrant-client==1.19.1` |
| Weaviate | `semitechnologies/weaviate:1.39.8`, `weaviate-client==4.23.1` |
| Milvus | `milvusdb/milvus:v2.6.25` (woodpecker MQ), `quay.io/coreos/etcd:v3.5.25`, `milvusdb/minio:RELEASE.2024-12-18T13-15-44Z`, `pymilvus==2.6.17` |
| pgvector | `pgvector/pgvector:0.8.7-pg17`, `psycopg[binary]==3.3.6`, `pgvector==0.5.0` |
| Chroma | `chromadb/chroma:1.5.9`, `chromadb-client==1.5.9` |
| client | `python:3.12-slim`, `numpy==2.5.3`, `h5py==3.16.0`, `matplotlib==3.11.2`, `requests==2.34.2` |

pgvector server settings (`compose.yml`): `shared_buffers=2GB`,
`effective_cache_size=6GB`, `max_parallel_maintenance_workers=7`,
`synchronous_commit=off`, `jit=off`, `shm_size: 2g`. The other engines run with
their defaults.

## How to run

Requirements: Docker Desktop (or Docker Engine) with Compose v2. Give the Docker
VM **at least 12 CPUs and 14 GB of memory**: 8 CPU / 8 GB for the engine, plus
4 CPU / 3 GB for the client, plus etcd and MinIO for Milvus.

```bash
cd bench/compare
./download.sh                      # ~1 GB into data/
./run_all.sh                       # everything; results/ + results/summary.md

# subsets / quicker runs
ENGINES="goatdb qdrant" DATASETS=sift-128-euclidean QUERIES=1000 ./run_all.sh
PRUNE_IMAGES=1 ./run_all.sh        # delete each engine's images after use (low disk)
LIMIT=100000 ./run_all.sh          # first 100k train vectors, ground truth recomputed

# one engine by hand
docker compose --profile qdrant up -d --wait
docker compose run --rm client python run.py --engine qdrant --dataset sift-128-euclidean --queries 1000 --keep
docker compose --profile qdrant down -v
docker compose run --rm --no-deps client python plot.py
```

`run.py --synthetic N [--dim D --metric cosine]` generates a clustered random
dataset with brute-force ground truth. Use it for smoke tests without
downloading anything. To try it against a local goatdb, outside Docker:

```bash
go build -o /tmp/goatdb ../../cmd/goatdb && /tmp/goatdb -addr 127.0.0.1:18080 -dir /tmp/goatdb-data &
python3 -m venv .venv && .venv/bin/pip install numpy h5py requests matplotlib
GOATDB_URL=http://127.0.0.1:18080 .venv/bin/python run.py --engine goatdb --synthetic 5000 --queries 300 \
  --concurrent-seconds 3 --out /tmp/smoke.json
```

Adapters find their servers through environment variables that default to the
compose service names: `GOATDB_URL`, `QDRANT_HOST`, `WEAVIATE_HOST`,
`MILVUS_URI`, `PG_HOST`, `CHROMA_HOST`, and so on (see `engines/*.py`).

### Disk and time budget

* Datasets: ~1 GB in total.
* Images: client ~0.7 GB, Milvus + etcd + MinIO ~1.5–2 GB, pgvector ~0.45 GB,
  Qdrant / Weaviate / Chroma ~0.2–0.3 GB each, and goatdb ~20 MB (plus Go
  build cache).
* Engine data for one 1M-vector run: roughly 0.7–3 GB, depending on the engine.
  pgvector (heap + index + WAL) and Milvus (MinIO binlogs + index) are at the
  high end.
* Only one engine's data exists at a time. With `PRUNE_IMAGES=1`, plan for
  **~8 GB free** at peak (the Milvus run). Without pruning, plan for ~12 GB.
* Time on an 8-CPU engine budget: each engine × dataset run takes about
  15–40 minutes (load and index dominate; 6 ef × 10k serial queries adds a few
  minutes). The full matrix of 12 runs takes **about 4–8 hours**. `QUERIES=1000`
  removes most of the query time.

## Output

* `results/<engine>__<dataset>.json`: parameters, server version, load timings
  and engine-specific build details, the full ef sweep, concurrent QPS, the
  memory peak (per container and total), and volume sizes.
* `results/<dataset>_recall_qps.png`: recall@10 vs serial QPS (log scale), one
  line per engine.
* `results/<dataset>_bars.png`: total load time, peak memory, disk, and
  concurrent QPS at recall ≥ 0.95.
* `results/summary.md`: the same numbers as tables.
* `results/raw/`: raw `docker stats` and `du` samples, plus engine logs on
  failure (gitignored).
