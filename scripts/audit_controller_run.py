#!/usr/bin/env python3
"""Independently audit immutable controller-run mirrors from raw service records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import audit_native_results as evidence
from controller.durable import digest, secure_directory, write_new_json


def audit(root: Path) -> dict:
    ticket = json.loads((root / "ticket.json").read_text())
    request = ticket["request"]
    evidence._require(digest(request) == ticket["request_sha256"], "Immutable request hash differs")
    for name, expected in ticket["assets_sha256"].items():
        evidence._require(evidence._sha(root / name) == expected, "Immutable asset drift: " + name)
    result = json.loads((root / "result.json").read_text())
    complete = json.loads((root / "COMPLETE").read_text())
    evidence._require(complete["result_sha256"] == evidence._sha(root / "result.json"), "Completion/result hash mismatch")
    evidence._require(result["status"] == "COMPLETE" and result["gpu_release_verified"] is True
                      and result["live_owned_containers"] == [], "Run did not finish and release GPU")
    for field in ("request_sha256", "reservation_id"):
        evidence._require(result[field] == ticket[field], "Terminal evidence is for another run")
    evidence._require(request["workload"]["kind"] == "fixed_decode", "This independent audit expects fixed decode")
    evidence._require(evidence._sha(root / "input/canonical_inputs.json") == request["dataset_sha256"], "Input SHA differs")
    inputs = json.loads((root / "input/canonical_inputs.json").read_text())
    source_ids = {row["session_id"] for row in inputs["data"]}
    evidence._require(len(source_ids) == len(inputs["data"]) == 136, "Input uniqueness/count mismatch")
    arms = []
    sessions = {}
    for arm in request["arms"]:
        case = root / "output" / arm
        measurement = json.loads((case / "measurement.json").read_text())
        evidence._require(measurement.get("measurement_success") is True and measurement["request_sha256"] == ticket["request_sha256"]
                          and measurement["arm"] == arm, "Worker completion identity differs")
        for name, expected in measurement["evidence_sha256"].items():
            evidence._require(evidence._sha(case / name) == expected, "Worker evidence content changed: " + name)
        text = (case / "server.log").read_text()
        native = evidence.verify_backend(text, json.loads((case / "native_backend.json").read_text()), arm)
        allocation = evidence.verify_allocation(json.loads((case / "allocation.json").read_text()), arm)
        phases = evidence.read_records(case / "artifacts/profile_export.jsonl", 128, 8)
        for row in phases["profiling"] + phases["warmup"]:
            evidence._require(evidence._value(row["metrics"], "output_token_count") == 512, "Early EOS or variable decode length")
            # The historical synthetic ISL labels omit chat-template tokens.
            # Preserve/report actual server usage instead of imposing the filename label.
        summary = json.loads((case / "artifacts/profile_export_aiperf.json").read_text())
        evidence.verify_summary(summary, phases["profiling"])
        metrics = evidence._metric_output(case, phases["profiling"], summary, 512)
        metrics.update(evidence.telemetry_output(case / "gpu_telemetry.csv", phases["profiling"]))
        sessions[arm] = {phase: {row["metadata"]["conversation_id"]: evidence._value(row["metrics"], "input_sequence_length")
                                 for row in rows} for phase, rows in phases.items()}
        evidence._require(set().union(*(set(rows) for rows in sessions[arm].values())) == source_ids,
                          "Observed sessions differ from frozen source")
        command = json.loads((case / "server_command.json").read_text())
        for required in ("--enable_graph=true", "--python_graph_backend=cudagraphs", "--enable_prefix_cache=false",
                         "--max_memory_utilization=0.93", "--max_tokens_per_batch=4096"):
            evidence._require(required in command, "Unmatched service setting: " + required)
        arms.append({"run_id": request["run_id"], "arm": arm, "concurrency": request["concurrency"],
                     "isl": request["workload"]["isl"], **metrics, **allocation,
                     "graph_captured_batches": native["captured_batches"]})
    evidence._require(sessions["bf16"] == sessions["fp8_e4m3"], "Paired phase/session membership differs")
    bf = next(row for row in arms if row["arm"] == "bf16")
    fp = next(row for row in arms if row["arm"] == "fp8_e4m3")
    differences = {name + "_fp8_delta_pct": 100 * (fp[name] / bf[name] - 1) for name in
                   ("output_tps", "request_rps", "ttft_p50_ms", "ttft_p95_ms", "itl_p50_ms", "itl_p95_ms", "e2e_p50_ms", "e2e_p95_ms")}
    return {"schema": "kv-controller-independent-fixed-audit-v1", "classification": "confirmed_measurement_only",
            "run_id": request["run_id"], "request_sha256": ticket["request_sha256"], "arms": arms,
            "paired": differences, "source_commit": request["server_source_commit"], "binary_sha256": request["binary_sha256"],
            "dataset_sha256": request["dataset_sha256"], "gpu_uuids": request["gpu_uuids"],
            "quality_validated": False, "run_level_replicates": 1,
            "limitation": "One paired fixed-decode run in pinned FA2 config; no matched quality, traffic counters or general production-benefit claim"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    if args.out.exists():
        raise FileExistsError("Audit output must be a new directory")
    secure_directory(args.out)
    write_new_json(args.out / "audit.json", report)
    fields = list(report["arms"][0])
    with (args.out / "cases.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(report["arms"])
    print(json.dumps({"run_id": report["run_id"], "verified": True, "paired": report["paired"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
