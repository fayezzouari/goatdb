#!/usr/bin/env python3
"""Run one engine x dataset benchmark and write a JSON result.

Examples (inside the client container, see README):
    python run.py --engine qdrant --dataset sift-128-euclidean
    python run.py --engine goatdb --synthetic 5000 --dim 32 --queries 200 --out results/smoke.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np

from engines import load

HERE = Path(__file__).resolve().parent
DATASETS = {
    "sift-128-euclidean": "euclidean",
    "glove-100-angular": "cosine",
}


# ── data ──────────────────────────────────────────────────────────────────────


def normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1
    return (x / n).astype(np.float32)


def load_hdf5(name: str, data_dir: Path):
    import h5py

    path = data_dir / f"{name}.hdf5"
    if not path.exists():
        sys.exit(f"{path} not found; run ./download.sh {name}")
    with h5py.File(path, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        gt = np.asarray(f["neighbors"], dtype=np.int64)
    metric = DATASETS[name]
    if metric == "cosine":
        # ann-benchmarks' angular ground truth is computed on cosine distance; we
        # use the cosine metric everywhere and also feed unit vectors so engines
        # that normalize internally and those that don't see identical data.
        train, test = normalize(train), normalize(test)
    return train, test, gt, metric


def brute_force_gt(train: np.ndarray, test: np.ndarray, metric: str, k: int = 100) -> np.ndarray:
    k = min(k, len(train))
    out = np.empty((len(test), k), dtype=np.int64)
    if metric == "cosine":
        tn, qn = normalize(train), normalize(test)
    tsq = (train**2).sum(1)
    for s in range(0, len(test), 256):
        q = test[s : s + 256]
        if metric == "cosine":
            d = -(qn[s : s + 256] @ tn.T)
        else:
            d = tsq[None, :] - 2 * q @ train.T  # + |q|^2, constant per row
        idx = np.argpartition(d, k - 1, axis=1)[:, :k]
        order = np.take_along_axis(d, idx, 1).argsort(1)
        out[s : s + 256] = np.take_along_axis(idx, order, 1)
    return out


def synthetic(n: int, dim: int, nq: int, metric: str, seed: int = 42):
    rng = np.random.default_rng(seed)
    # Clustered data so HNSW recall actually depends on ef.
    centers = rng.normal(size=(max(8, n // 500), dim)).astype(np.float32) * 4
    train = centers[rng.integers(0, len(centers), n)] + rng.normal(size=(n, dim)).astype(np.float32)
    test = centers[rng.integers(0, len(centers), nq)] + rng.normal(size=(nq, dim)).astype(np.float32)
    train, test = train.astype(np.float32), test.astype(np.float32)
    if metric == "cosine":
        train, test = normalize(train), normalize(test)
    return train, test, brute_force_gt(train, test, metric), metric


# ── measurement ───────────────────────────────────────────────────────────────


def recall_at_k(results: list[list[int]], gt: np.ndarray, k: int) -> float:
    hits = sum(len(set(r[:k]) & set(g[:k].tolist())) for r, g in zip(results, gt))
    return hits / (k * len(results))


def serial_queries(searcher, queries: np.ndarray, k: int, warmup: int):
    for q in queries[:warmup]:
        searcher.search(q, k)
    lat = np.empty(len(queries))
    results = []
    t0 = time.perf_counter()
    for i, q in enumerate(queries):
        s = time.perf_counter()
        results.append(list(searcher.search(q, k)))
        lat[i] = time.perf_counter() - s
    wall = time.perf_counter() - t0
    return results, lat, wall


def concurrent_queries(engine, queries: np.ndarray, k: int, threads: int, seconds: float):
    searchers = [engine.make_searcher() for _ in range(threads)]
    counts = [0] * threads
    errors: list[BaseException] = []
    start = threading.Barrier(threads + 1)
    stop_at = [0.0]

    def worker(t: int):
        s = searchers[t]
        n = len(queries)
        i = (t * n) // threads
        start.wait()
        try:
            while time.perf_counter() < stop_at[0]:
                s.search(queries[i % n], k)
                i += 1
                counts[t] += 1
        except BaseException as e:  # surface failures instead of silently low QPS
            errors.append(e)

    ths = [threading.Thread(target=worker, args=(t,), daemon=True) for t in range(threads)]
    for th in ths:
        th.start()
    stop_at[0] = time.perf_counter() + seconds
    t0 = time.perf_counter()
    start.wait()
    for th in ths:
        th.join()
    wall = time.perf_counter() - t0
    for s in searchers:
        if s is not engine:
            s.close()
    if errors:
        raise errors[0]
    return sum(counts), wall


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", choices=sorted(DATASETS))
    g.add_argument("--synthetic", type=int, metavar="N", help="random clustered dataset of N vectors")
    ap.add_argument("--dim", type=int, default=32, help="synthetic only")
    ap.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean", help="synthetic only")
    ap.add_argument("--data-dir", type=Path, default=HERE / "data")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N train vectors (recomputes ground truth)")
    ap.add_argument("--queries", type=int, default=1000, help="number of test queries (max 10000)")
    ap.add_argument("--efs", default="16,32,64,128,256,512")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--concurrent-seconds", type=float, default=20)
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--keep", action="store_true", help="do not drop the collection at the end")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    efs = [int(x) for x in args.efs.split(",") if x]

    # data
    t = time.perf_counter()
    if args.synthetic:
        nq = min(args.queries, 10000)
        train, test, gt, metric = synthetic(args.synthetic, args.dim, nq, args.metric)
        ds_name = f"synthetic-{args.synthetic}-{args.dim}-{args.metric}"
    else:
        train, test, gt, metric = load_hdf5(args.dataset, args.data_dir)
        ds_name = args.dataset
    if args.limit and args.limit < len(train):
        train = train[: args.limit]
        gt = None
    test = test[: args.queries]
    if gt is None:
        gt = brute_force_gt(train, test, metric)
    gt = gt[: len(test)]
    dim = train.shape[1]
    print(f"[{args.engine}] {ds_name}: train={train.shape} queries={len(test)} metric={metric} "
          f"(loaded in {time.perf_counter() - t:.1f}s)", flush=True)

    engine = load(args.engine)()
    result: dict = {
        "engine": args.engine,
        "dataset": ds_name,
        "metric": metric,
        "n_train": int(len(train)),
        "dim": int(dim),
        "n_queries": int(len(test)),
        "params": {"m": args.m, "ef_construction": args.ef_construction, "k": args.k,
                   "batch": args.batch, "threads": args.threads, "efs": efs},
        "client": {"python": platform.python_version(), "host": platform.node()},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    engine.setup(dim, metric, args.m, args.ef_construction)
    result["server_version"] = engine.version()

    # insert
    ids = np.arange(len(train), dtype=np.int64)
    t0 = time.perf_counter()
    last = t0
    for s in range(0, len(train), args.batch):
        engine.insert(ids[s : s + args.batch], train[s : s + args.batch])
        now = time.perf_counter()
        if now - last > 10:
            done = min(s + args.batch, len(train))
            print(f"  inserted {done}/{len(train)} ({done / (now - t0):.0f} vec/s)", flush=True)
            last = now
    insert_s = time.perf_counter() - t0
    print(f"  insert: {insert_s:.1f}s", flush=True)

    t0 = time.perf_counter()
    build_info = engine.finish_build() or {}
    build_s = time.perf_counter() - t0
    print(f"  finish_build: {build_s:.1f}s {build_info}", flush=True)
    result["insert_s"] = insert_s
    result["build_s"] = build_s
    result["total_load_s"] = insert_s + build_s
    result["build_info"] = build_info

    # ef sweep (serial, single client)
    sweep = []
    for ef in efs:
        engine.set_ef(ef)
        res, lat, wall = serial_queries(engine, test, args.k, min(args.warmup, len(test)))
        rec = recall_at_k(res, gt, args.k)
        row = {
            "ef": ef,
            "recall": rec,
            "qps": len(test) / wall,
            "p50_ms": float(np.percentile(lat, 50) * 1e3),
            "p95_ms": float(np.percentile(lat, 95) * 1e3),
            "p99_ms": float(np.percentile(lat, 99) * 1e3),
            "mean_ms": float(lat.mean() * 1e3),
        }
        sweep.append(row)
        print(f"  ef={ef:<4} recall={rec:.4f} qps={row['qps']:.0f} p50={row['p50_ms']:.2f}ms "
              f"p99={row['p99_ms']:.2f}ms", flush=True)
    result["sweep"] = sweep

    # concurrent QPS at the smallest ef reaching the target recall (or the best ef)
    ok = [r for r in sweep if r["recall"] >= args.target_recall]
    chosen = min(ok, key=lambda r: r["ef"]) if ok else max(sweep, key=lambda r: r["recall"])
    engine.set_ef(chosen["ef"])
    n, wall = concurrent_queries(engine, test, args.k, args.threads, args.concurrent_seconds)
    result["concurrent"] = {
        "ef": chosen["ef"],
        "recall": chosen["recall"],
        "reached_target": bool(ok),
        "target_recall": args.target_recall,
        "threads": args.threads,
        "queries": n,
        "seconds": wall,
        "qps": n / wall,
    }
    print(f"  concurrent ({args.threads} threads, ef={chosen['ef']}): {n / wall:.0f} qps", flush=True)

    if not args.keep:
        engine.teardown()
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    out = args.out or HERE / "results" / f"{args.engine}__{ds_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    os.chdir(HERE)
    main()
