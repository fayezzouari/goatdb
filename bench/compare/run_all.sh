#!/usr/bin/env bash
# Run every engine x dataset sequentially: start one engine, benchmark it from
# the client container, sample its memory, measure its data volume, tear it down
# (removing volumes) before the next one starts.
#
# Environment overrides:
#   ENGINES="goatdb qdrant weaviate milvus pgvector chroma"
#   DATASETS="sift-128-euclidean glove-100-angular"
#   QUERIES=10000  EFS=16,32,64,128,256,512  THREADS=8  CONC_SECONDS=20
#   LIMIT=0               use only the first N train vectors (0 = all)
#   PRUNE_IMAGES=0        1 = remove each engine's images after its runs (saves disk)
#   EXTRA_ARGS=""         extra args passed to run.py
#
# Works with macOS' bash 3.2.
set -uo pipefail

cd "$(dirname "$0")"

ENGINES=${ENGINES:-"goatdb qdrant weaviate milvus pgvector chroma"}
DATASETS=${DATASETS:-"sift-128-euclidean glove-100-angular"}
QUERIES=${QUERIES:-10000}
EFS=${EFS:-16,32,64,128,256,512}
THREADS=${THREADS:-8}
CONC_SECONDS=${CONC_SECONDS:-20}
LIMIT=${LIMIT:-0}
PRUNE_IMAGES=${PRUNE_IMAGES:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}
PROJECT=vecbench

DC="docker compose -f compose.yml"

containers_for() {
  case "$1" in
    milvus) echo "vb-milvus vb-milvus-etcd vb-milvus-minio" ;;
    *) echo "vb-$1" ;;
  esac
}

volumes_for() {
  case "$1" in
    milvus) echo "milvus_data milvus_etcd milvus_minio" ;;
    *) echo "$1_data" ;;
  esac
}

images_for() {
  case "$1" in
    goatdb) echo "goatdb:bench" ;;
    qdrant) echo "qdrant/qdrant:v1.19.1" ;;
    weaviate) echo "semitechnologies/weaviate:1.39.8" ;;
    milvus) echo "milvusdb/milvus:v2.6.25 quay.io/coreos/etcd:v3.5.25 milvusdb/minio:RELEASE.2024-12-18T13-15-44Z" ;;
    pgvector) echo "pgvector/pgvector:0.8.7-pg17" ;;
    chroma) echo "chromadb/chroma:1.5.9" ;;
  esac
}

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

sample_mem() {
  # $1 = output file, rest = container names. One `docker stats` snapshot per
  # iteration (~1-2 s each); every line gets the snapshot's timestamp.
  local out=$1; shift
  while :; do
    local ts; ts=$(date +%s)
    docker stats --no-stream --format '{{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}' "$@" 2>/dev/null \
      | sed "s/^/${ts}\t/" >>"$out"
  done
}

mkdir -p results/raw

for ds in $DATASETS; do
  if [ ! -s "data/$ds.hdf5" ]; then
    echo "data/$ds.hdf5 missing; run ./download.sh $ds first" >&2
    exit 1
  fi
done

log "building client image"
$DC build client || exit 1

failed=""
for engine in $ENGINES; do
  for ds in $DATASETS; do
    tag="${engine}__${ds}"
    result="results/${tag}.json"
    memf="results/raw/${tag}.mem.tsv"
    diskf="results/raw/${tag}.disk.tsv"
    rm -f "$memf" "$diskf"

    log "=== $engine / $ds: starting engine"
    $DC --profile "$engine" down -v --remove-orphans >/dev/null 2>&1
    if ! $DC --profile "$engine" up -d --build --wait; then
      echo "failed to start $engine" >&2
      failed="$failed $tag"
      $DC --profile "$engine" down -v >/dev/null 2>&1
      continue
    fi

    # shellcheck disable=SC2046
    sample_mem "$memf" $(containers_for "$engine") &
    sampler=$!

    log "=== $engine / $ds: benchmarking"
    # --keep: leave the data in place so the volume size can be measured below.
    # shellcheck disable=SC2086
    $DC run --rm client python run.py \
      --engine "$engine" --dataset "$ds" --queries "$QUERIES" --efs "$EFS" \
      --threads "$THREADS" --concurrent-seconds "$CONC_SECONDS" --limit "$LIMIT" \
      --keep --out "$result" $EXTRA_ARGS
    rc=$?

    kill "$sampler" 2>/dev/null
    wait "$sampler" 2>/dev/null

    if [ $rc -eq 0 ]; then
      log "=== $engine / $ds: measuring data volume(s)"
      for v in $(volumes_for "$engine"); do
        bytes=$($DC run --rm --no-deps -v "${PROJECT}_${v}:/vol:ro" client du -sb /vol 2>/dev/null | cut -f1)
        printf '%s\t%s\n' "$v" "${bytes:-0}" >>"$diskf"
      done
      $DC run --rm --no-deps client python stats.py merge --result "$result" --mem "$memf" --disk "$diskf"
    else
      echo "run.py failed for $tag (exit $rc)" >&2
      failed="$failed $tag"
      $DC --profile "$engine" logs --tail 100 >"results/raw/${tag}.engine.log" 2>&1
    fi

    log "=== $engine / $ds: tearing down (removing volumes)"
    $DC --profile "$engine" down -v --remove-orphans
  done

  if [ "$PRUNE_IMAGES" = "1" ]; then
    # shellcheck disable=SC2046
    docker image rm $(images_for "$engine") >/dev/null 2>&1 || true
  fi
done

log "plotting"
$DC run --rm --no-deps client python plot.py

if [ -n "$failed" ]; then
  log "FAILED:$failed"
  exit 1
fi
log "done; see results/summary.md and results/*.png"
