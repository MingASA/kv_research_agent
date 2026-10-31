#!/usr/bin/env python3
"""Validate the original BurstGPT peak-window lineage without inference."""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path


def audit(source: Path, frozen_directory: Path) -> dict:
    manifest_path = frozen_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    frozen = frozen_directory / "burstgpt_peak10m.csv"
    for name, path in (("BurstGPT_1.csv", source), (frozen.name, frozen)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files_sha256"][name]:
            raise ValueError("Original/frozen source bytes differ: " + name)
    with frozen.open(newline="") as stream:
        frozen_rows = list(csv.DictReader(stream))
    window_seconds = manifest["burst_window_seconds"]
    selected_bin = manifest["burst_peak_window_index"]
    counts = collections.Counter()
    models = collections.Counter()
    excluded = collections.Counter()
    origins, selected = [], []
    first = last = previous = None
    ordered = True
    rows = 0
    with source.open(newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            stamp = int(row["Timestamp"])
            if first is None:
                first = stamp
            if previous is not None and stamp < previous:
                ordered = False
            previous = stamp
            last = stamp
            rows += 1
            models[row["Model"]] += 1
            inputs, outputs = int(row["Request tokens"]), int(row["Response tokens"])
            if inputs <= 0:
                excluded["nonpositive_input"] += 1
                continue
            if outputs <= 0:
                excluded["nonpositive_output"] += 1
                continue
            if inputs + outputs > manifest["input_plus_output_limit"]:
                excluded["above_frozen_context_filter"] += 1
                continue
            bin_index = (stamp - first) // window_seconds
            counts[bin_index] += 1
            if bin_index == selected_bin:
                selected.append(row)
                origins.append(index)
    if not ordered:
        raise ValueError("Original timestamp order changed; window protocol must be revisited")
    if selected != frozen_rows:
        raise ValueError("Frozen peak is not the exact ordered original source window")
    if len(selected) != manifest["burst_peak_requests"] or len(selected) != max(counts.values()):
        raise ValueError("Frozen window is not an eligible highest-count window")
    begin = first + selected_bin * window_seconds
    stamps = [int(row["Timestamp"]) for row in selected]
    return {
        "schema": "kv-burstgpt-source-audit-v1",
        "classification": "confirmed_source_and_frozen_window_lineage_only",
        "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpu_used": False,
        "model_inference_performed": False,
        "source_sha256": manifest["files_sha256"][source.name],
        "frozen_sha256": manifest["files_sha256"][frozen.name],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_rows": rows,
        "source_model_counts": dict(models),
        "source_first_timestamp_seconds": first,
        "source_last_timestamp_seconds": last,
        "source_timestamp_ordered": ordered,
        "eligible_rows": sum(counts.values()),
        "mutually_exclusive_filter_counts_in_order": dict(excluded),
        "filter_order": ["nonpositive_input", "nonpositive_output", "above_frozen_context_filter"],
        "context_filter_input_plus_output_limit": manifest["input_plus_output_limit"],
        "window_origin_seconds": first,
        "window_seconds": window_seconds,
        "window_index": selected_bin,
        "window_begin_inclusive": begin,
        "window_end_exclusive": begin + window_seconds,
        "window_requests": len(selected),
        "window_model_counts": dict(collections.Counter(row["Model"] for row in selected)),
        "first_actual_arrival_seconds": min(stamps),
        "last_actual_arrival_seconds": max(stamps),
        "offered_rps_over_declared_window": len(selected) / window_seconds,
        "one_second_peak_requests": max(collections.Counter(stamps).values()),
        "exact_ordered_source_rows_match": True,
        "source_data_row_indices_zero_based": origins,
        "top_eligible_window_counts": counts.most_common(5),
        "raw_text_available": False,
        "session_ids_available": False,
        "limitations": [
            "Mixed ChatGPT/GPT-4 aggregate demand needs explicit mapping to the deployment instance",
            "Lengths/arrivals only; synthesized prompts are not original semantic conversations",
            "Source zero-output requests do not reveal a desired generation length; exclusions are explicit",
            "Input/output filtering is the historical frozen-window protocol, not full trace coverage",
            "No GPU inference or serving-performance result is established by this audit",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--frozen-directory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source, args.frozen_directory)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("source_rows", "eligible_rows", "window_requests", "exact_ordered_source_rows_match")}))


if __name__ == "__main__":
    main()
