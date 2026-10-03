## goatdb

A vector database built from scratch in Go. Stores high-dimensional vectors with metadata, supports multiple approximate nearest neighbour index algorithms, and exposes a REST API over HTTP.

---

## Architecture

```
cmd/goatdb         entry point, CLI flags, graceful shutdown
api/               HTTP server, routing, middleware
  handlers/        one file per concern (collections, vectors, search)
  middleware/       logging, recovery, request size limit, metrics
core/              shared types and interfaces (no dependencies)
index/             ANN index implementations
  flat             brute-force, exact results
  lsh              locality-sensitive hashing
  ivf              inverted file index with k-means clustering
  hnsw             hierarchical navigable small world graph
db/                collection and database orchestration
storage/           persistence layer
  vector.go        memory-mapped file for embeddings
  meta.go          BoltDB for metadata and slot mapping
  wal.go           write-ahead log for crash recovery
```

Data flow on write: WAL append and sync, then mmap write, then BoltDB update.
Data flow on read: BoltDB slot lookup, then mmap read.
On startup: WAL is replayed into mmap and BoltDB, then truncated.

---

## Install

**Docker (linux/amd64, linux/arm64)**

Images are published to GitHub Container Registry and Docker Hub on every release.

```bash
# GitHub Container Registry
docker run -d --name goatdb -p 8080:8080 -v goatdb_data:/data ghcr.io/fayezzouari/goatdb:latest

# Docker Hub
docker run -d --name goatdb -p 8080:8080 -v goatdb_data:/data fayezzouari/goatdb:latest
```

Pin a version with a tag such as `:0.1.0` or `:0.1`. The image has a built-in healthcheck on `/health`.

**Prebuilt binaries**

Download an archive for Linux or macOS (amd64 or arm64) from the
[Releases page](https://github.com/fayezzouari/goatdb/releases), verify it against
`checksums.txt`, then run it:

```bash
tar -xzf goatdb_<version>_linux_amd64.tar.gz
./goatdb -version
./goatdb -addr :8080 -dir ./data
```

Windows is not supported: the storage layer uses `mmap`/`msync`.

**go install**

`go install` from the GitHub path does not work yet, because the module path
(`github.com/fayez/goatdb`) does not match the repository location. Build from source
instead (see below).

---

## Getting started

**Prerequisites**

- Go 1.22 or later
- Docker and Docker Compose (optional)

**Build and run**

```bash
git clone https://github.com/fayezzouari/goatdb
cd goatdb
go build -o goatdb ./cmd/goatdb
./goatdb -addr :8080 -dir ./data
```

**Run with Docker Compose**

```bash
docker compose up -d
```

The service listens on port 8080 by default. Data is persisted in a named Docker volume.

**Environment overrides**

```bash
GOATDB_PORT=9090 docker compose up -d
```

---

## Configuration

| Flag     | Environment   | Default  | Description                        |
|----------|---------------|----------|------------------------------------|
| -addr    | `GOATDB_ADDR` | :8080    | TCP address the server listens on  |
| -dir     | `GOATDB_DIR`  | ./data   | Directory where data is persisted  |
| -version |               |          | Print the version and exit         |

Flags take precedence over environment variables. In the Docker image the defaults
are `GOATDB_ADDR=:8080` and `GOATDB_DIR=/data`.

---

## API

All request and response bodies are JSON. Errors always include an `error` string and a `code` string.

**Error codes**

| Code           | HTTP status |
|----------------|-------------|
| bad_request    | 400         |
| not_found      | 404         |
| conflict       | 409         |
| internal_error | 500         |

---

### Health

```
GET /health
```

```json
{ "status": "ok", "time": "2026-03-22T10:00:00Z" }
```

---

### Metrics

```
GET /metrics
```

```json
{
  "requests_total": 142,
  "status_2xx": 138,
  "status_4xx": 3,
  "status_5xx": 1
}
```

---

### Collections

**Create**

```
POST /collections
```

```json
{
  "name": "articles",
  "dim": 1536,
  "metric": "cosine",
  "index_type": "hnsw"
}
```

`metric` options: `cosine`, `euclidean`, `dot_product`, `manhattan`. Results are ordered by ascending distance; for `dot_product` the distance is the negated dot product, so the largest dot product comes first.
`index_type` options: `flat`, `lsh`, `ivf`, `hnsw` — defaults to `flat` if omitted

Response `201`:
```json
{ "name": "articles" }
```

---

**List**

```
GET /collections
```

Response `200`:
```json
{ "collections": ["articles", "images"] }
```

---

**Get**

```
GET /collections/{name}
```

Response `200`:
```json
{
  "name": "articles",
  "dim": 1536,
  "metric": "cosine",
  "index_type": "hnsw"
}
```

---

**Drop**

```
DELETE /collections/{name}
```

Response `204` — no body. Permanently deletes all vectors and index data.

---

### Vectors

**Add**

```
POST /collections/{name}/vectors
```

```json
{
  "id": "doc-001",
  "embeddings": [0.1, 0.4, 0.9],
  "metadata": { "title": "Introduction to Go", "year": 2024 }
}
```

Response `201`:
```json
{ "id": "doc-001" }
```

---

**Bulk add**

```
POST /collections/{name}/vectors/batch
```

```json
{
  "vectors": [
    { "id": "doc-001", "embeddings": [0.1, 0.4, 0.9] },
    { "id": "doc-002", "embeddings": [0.3, 0.2, 0.8], "metadata": { "tag": "go" } }
  ]
}
```

Response `201`:
```json
{ "inserted": 2 }
```

All vectors are validated before any are written. If one has a wrong dimension, the entire batch is rejected.

---

**Get**

```
GET /collections/{name}/vectors/{id}
```

Response `200`:
```json
{
  "id": "doc-001",
  "embeddings": [0.1, 0.4, 0.9],
  "metadata": { "title": "Introduction to Go", "year": 2024 }
}
```

---

**Update**

```
PUT /collections/{name}/vectors/{id}
```

```json
{
  "embeddings": [0.2, 0.5, 0.7],
  "metadata": { "title": "Introduction to Go", "updated": true }
}
```

Response `204` — no body.

---

**Delete**

```
DELETE /collections/{name}/vectors/{id}
```

Response `204` — no body.

---

### Search

```
POST /collections/{name}/search
```

```json
{
  "embeddings": [0.1, 0.4, 0.9],
  "top_k": 5
}
```

`top_k` defaults to 10 if omitted or zero.

| Field              | Default | Description                        |
|--------------------|---------|------------------------------------|
| `include_vectors`  | `false` | Return each result's `embeddings`  |
| `include_metadata` | `true`  | Return each result's `metadata`    |

Response `200`:
```json
{
  "results": [
    {
      "id": "doc-001",
      "distance": 0.012,
      "metadata": { "title": "Introduction to Go" }
    }
  ]
}
```

With `"include_vectors": true`, each result also carries `"embeddings": [0.1, 0.4, 0.9]`. Leaving vectors out (and metadata too, with `"include_metadata": false`) skips the storage reads and keeps responses small, which matters at large `top_k` and `dim`.

Results are sorted by distance ascending (nearest first). The distance unit depends on the metric chosen at collection creation time.

---

### Training

Required for `ivf` indexes. Has no effect on other index types.

```
POST /collections/{name}/train
```

No request body.

Response `200`:
```json
{ "status": "trained" }
```

Train after inserting a representative sample of vectors. Searching an untrained IVF index falls back to a single cluster and will have poor recall.

---

## Index types

**flat**

Exact brute-force search. Computes distance to every vector. Guaranteed to return the true top-K. Use for small collections or when recall must be 100%.

**lsh**

Locality-sensitive hashing. Uses random hyperplanes to hash vectors into buckets. Fast insert and search. Recall degrades as the dataset grows. Suitable for high-dimensional data where approximate results are acceptable.

**ivf**

Inverted file index. Partitions vectors into clusters via k-means. At search time, only the nearest clusters are scanned. Requires calling `POST /train` after initial data load. Scales well to large collections.

**hnsw**

Hierarchical navigable small world graph. Builds a multi-layer proximity graph. Offers the best recall-vs-speed trade-off in practice. Higher memory usage than other indexes. Recommended for production workloads.

---

## Storage

Embeddings are stored in a memory-mapped binary file (`vectors.bin`). Each slot is `dim * 4 + 1` bytes — four bytes per float32 plus a tombstone flag for soft deletes.

Metadata and slot assignments are stored in BoltDB (`meta.db`). Deleted slots are returned to a free list and reused on the next insert.

Every write goes through a write-ahead log (`wal.log`) before touching the mmap or BoltDB. On startup the WAL is replayed and then truncated. This ensures no data is lost on an unclean shutdown.

---

## Running tests

```bash
go test ./...
```

```bash
go test -bench=. -benchmem ./...
```

CI runs `go vet`, `go test -race` and `go build` on Linux amd64 and arm64 for every
push and pull request to `master`, and builds the Docker image for both architectures.

---

## Releasing

Releases are driven by git tags. Push a semver tag to publish:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The `Release` workflow then:

- builds Linux and macOS binaries (amd64, arm64) with GoReleaser and attaches them,
  with `checksums.txt`, to a GitHub release;
- builds a multi-arch image (linux/amd64, linux/arm64) and pushes it to
  `ghcr.io/fayezzouari/goatdb` and `docker.io/<DOCKERHUB_USERNAME>/goatdb`, tagged
  `X.Y.Z`, `X.Y`, `X` (from 1.0 on) and `latest`. Pre-release tags such as
  `v0.2.0-rc.1` do not move `latest`.

Required repository secrets (Settings > Secrets and variables > Actions):

| Secret               | Purpose                                                  |
|----------------------|----------------------------------------------------------|
| `DOCKERHUB_USERNAME` | Docker Hub account or organisation the image is pushed to |
| `DOCKERHUB_TOKEN`    | Docker Hub access token with read/write scope            |

GHCR uses the built-in `GITHUB_TOKEN`. If the Docker Hub secrets are not set, the
Docker Hub push is skipped and the image goes to GHCR only.

---

## License

MIT
