#!/usr/bin/env python3
"""Validate and summarize selected published numeric records; standard library only."""
import collections
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent.parent
rows = [json.loads(line) for line in (root / "data/measurements.jsonl").read_text().splitlines() if line.strip()]
assert all("source_file" in r and "source_line" in r for r in rows)
assert all(not ({"text", "content", "api_key", "messages"} & r.keys()) for r in rows)
for filename in ("dspark-compare.jsonl", "dspark-conf-compare.jsonl"):
    groups = collections.defaultdict(list)
    for r in rows:
        if r["source_file"] == filename and r.get("status") == "result":
            groups[r["block"]].append(r)
    for label, group in sorted(groups.items()):
        speeds = [r["timings"]["predicted_per_second"] for r in group]
        drafted = sum(r["timings"]["draft_n"] for r in group)
        accepted = sum(r["timings"]["draft_n_accepted"] for r in group)
        print(f"{filename} {label}: n={len(group)} mean={statistics.mean(speeds):.2f} token/s range={min(speeds):.1f}-{max(speeds):.1f} accepted={accepted}/{drafted}")
print(f"Validated {len(rows)} records from {len({r['source_file'] for r in rows})} source files")
