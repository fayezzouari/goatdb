#!/usr/bin/env python3
"""Plot results/*.json: recall-vs-QPS per dataset, bars for load time, memory
and disk, plus a markdown summary table (results/summary.md)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ORDER = ["goatdb", "qdrant", "weaviate", "milvus", "pgvector", "chroma"]
# Fixed per-engine colors so an engine looks the same in every chart.
COLORS = {
    "goatdb": "#2a78d6",
    "qdrant": "#e0457b",
    "weaviate": "#1f9e6e",
    "milvus": "#7a5af5",
    "pgvector": "#d97a1e",
    "chroma": "#8a8f98",
}


def load(dir_: Path) -> dict[str, dict[str, dict]]:
    by_ds: dict[str, dict[str, dict]] = defaultdict(dict)
    for p in sorted(dir_.glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if "engine" in r and "sweep" in r:
            by_ds[r["dataset"]][r["engine"]] = r
    return by_ds


def engines_sorted(rs: dict) -> list[str]:
    return sorted(rs, key=lambda e: (ORDER.index(e) if e in ORDER else 99, e))


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def recall_qps(ds: str, rs: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for e in engines_sorted(rs):
        sw = sorted(rs[e]["sweep"], key=lambda r: r["ef"])
        ax.plot([r["recall"] for r in sw], [r["qps"] for r in sw], marker="o", markersize=4,
                linewidth=2, color=COLORS.get(e), label=e)
    ax.set_yscale("log")
    style(ax, f"{ds}: recall@10 vs serial QPS (single client, ef sweep)", "recall@10", "queries / s (log)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def bars(ds: str, rs: dict, out: Path) -> None:
    es = engines_sorted(rs)
    metrics = [
        ("Load time (insert + build), s", lambda r: r.get("total_load_s")),
        ("Peak memory, GiB", lambda r: (r.get("memory") or {}).get("peak_bytes_total") and
         r["memory"]["peak_bytes_total"] / 1024**3),
        ("Data on disk, GiB", lambda r: (r.get("disk") or {}).get("bytes_total") and
         r["disk"]["bytes_total"] / 1024**3),
        ("Concurrent QPS (8 threads, recall ≥ 0.95)", lambda r: (r.get("concurrent") or {}).get("qps")),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4))
    for ax, (title, f) in zip(axes, metrics):
        vals = [f(rs[e]) or 0 for e in es]
        ax.bar(es, vals, color=[COLORS.get(e) for e in es])
        for i, v in enumerate(vals):
            if v:
                ax.text(i, v, f"{v:,.1f}" if v < 100 else f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
        style(ax, title)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(ds, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fmt(v, spec=".1f"):
    return "–" if v is None else format(v, spec)


def summary(by_ds: dict, out: Path) -> None:
    lines = ["# Benchmark summary", ""]
    for ds, rs in sorted(by_ds.items()):
        lines += [f"## {ds}", "",
                  "| engine | version | insert s | build s | total load s | peak mem GiB | disk GiB "
                  "| ef@0.95 | recall | p50 ms | p99 ms | serial QPS | 8-thread QPS |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for e in engines_sorted(rs):
            r = rs[e]
            c = r.get("concurrent") or {}
            row = next((s for s in r["sweep"] if s["ef"] == c.get("ef")), None) or {}
            mem = (r.get("memory") or {}).get("peak_bytes_total")
            disk = (r.get("disk") or {}).get("bytes_total")
            ef = f"{c.get('ef')}" + ("" if c.get("reached_target", True) else " (max)")
            lines.append(
                f"| {e} | {r.get('server_version', '')} | {fmt(r.get('insert_s'))} | {fmt(r.get('build_s'))} "
                f"| {fmt(r.get('total_load_s'))} | {fmt(mem and mem / 1024**3, '.2f')} "
                f"| {fmt(disk and disk / 1024**3, '.2f')} | {ef} | {fmt(row.get('recall'), '.4f')} "
                f"| {fmt(row.get('p50_ms'), '.2f')} | {fmt(row.get('p99_ms'), '.2f')} | {fmt(row.get('qps'), '.0f')} "
                f"| {fmt(c.get('qps'), '.0f')} |"
            )
        lines += ["", "Full ef sweep:", "",
                  "| engine | ef | recall@10 | QPS | p50 ms | p99 ms |", "|---|---:|---:|---:|---:|---:|"]
        for e in engines_sorted(rs):
            for s in sorted(rs[e]["sweep"], key=lambda s: s["ef"]):
                lines.append(f"| {e} | {s['ef']} | {s['recall']:.4f} | {s['qps']:.0f} | {s['p50_ms']:.2f} "
                             f"| {s['p99_ms']:.2f} |")
        lines += [f"", f"![recall vs QPS]({ds}_recall_qps.png)", f"![resources]({ds}_bars.png)", ""]
    out.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results")
    args = ap.parse_args()
    by_ds = load(args.results)
    if not by_ds:
        raise SystemExit(f"no result JSON files in {args.results}")
    for ds, rs in by_ds.items():
        recall_qps(ds, rs, args.results / f"{ds}_recall_qps.png")
        bars(ds, rs, args.results / f"{ds}_bars.png")
    summary(by_ds, args.results / "summary.md")
    print(f"wrote plots and {args.results / 'summary.md'}")


if __name__ == "__main__":
    main()
