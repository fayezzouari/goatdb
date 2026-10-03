#!/usr/bin/env python3
"""Merge host-side resource measurements into a run.py result JSON.

Stdlib only, so it runs on the host or in the client container.

    stats.py merge --result results/x.json --mem raw/x.mem.tsv --disk raw/x.disk.tsv

mem TSV lines (written by run_all.sh from `docker stats --no-stream`):
    <unix_ts>\t<container>\t<MemUsage e.g. "1.2GiB / 8GiB">\t<CPU% e.g. "312.5%">
disk TSV lines:
    <volume>\t<bytes>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

_UNITS = {
    "b": 1, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12,
    "kib": 1024, "mib": 1024**2, "gib": 1024**3, "tib": 1024**4,
}


def parse_size(s: str) -> float:
    m = re.fullmatch(r"\s*([\d.]+)\s*([a-zA-Z]+)\s*", s)
    if not m:
        raise ValueError(f"bad size {s!r}")
    return float(m.group(1)) * _UNITS[m.group(2).lower()]


def mem_summary(path: Path) -> dict:
    per_container: dict[str, float] = defaultdict(float)
    per_ts: dict[str, float] = defaultdict(float)
    cpu_peak: dict[str, float] = defaultdict(float)
    samples = 0
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or "/" not in parts[2]:
            continue
        ts, name, usage = parts[0], parts[1], parts[2]
        try:
            used = parse_size(usage.split("/")[0])
        except (ValueError, KeyError):
            continue
        samples += 1
        per_container[name] = max(per_container[name], used)
        per_ts[ts] += used
        if len(parts) > 3 and parts[3].strip().endswith("%"):
            try:
                cpu_peak[name] = max(cpu_peak[name], float(parts[3].strip().rstrip("%")))
            except ValueError:
                pass
    return {
        "samples": samples,
        "peak_bytes_by_container": dict(per_container),
        # Peak of the summed usage across the engine's containers at one sampling
        # instant (relevant for Milvus = standalone + etcd + minio).
        "peak_bytes_total": max(per_ts.values()) if per_ts else None,
        "peak_cpu_pct_by_container": dict(cpu_peak),
    }


def disk_summary(path: Path) -> dict:
    vols = {}
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[1].strip().isdigit():
            vols[parts[0]] = int(parts[1])
    return {"bytes_by_volume": vols, "bytes_total": sum(vols.values()) if vols else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--result", type=Path, required=True)
    m.add_argument("--mem", type=Path)
    m.add_argument("--disk", type=Path)
    args = ap.parse_args()

    res = json.loads(args.result.read_text())
    if args.mem and args.mem.exists():
        res["memory"] = mem_summary(args.mem)
    if args.disk and args.disk.exists():
        res["disk"] = disk_summary(args.disk)
    args.result.write_text(json.dumps(res, indent=2))
    print(f"merged resource stats into {args.result}: "
          f"mem_peak={res.get('memory', {}).get('peak_bytes_total')} disk={res.get('disk', {}).get('bytes_total')}")


if __name__ == "__main__":
    main()
