#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Describe retained trace errors from verified raw exports, without rerunning GPU work."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ADMISSION_NOTICE = "The number of concurrent requests has reached the limit."


def analyze(audit_path: Path, mirror: Path) -> dict:
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    if audit.get("classification") not in ("confirmed_trace_measurement_integrity",
                                          "confirmed_trace_measurement_integrity_runtime_conflict_retained"):
        raise ValueError("Require a completed independent trace audit")
    root = mirror / "support/agent_work/runs" / audit["run_id"]
    result = {}
    sets = {}
    for arm, data in audit["sealed_arms"].items():
        failed = {row["session_id"]: row for row in data["metrics"]["per_request"] if not row["success"]}
        classified = Counter()
        phases = {name: Counter() for name in ("warm_cache", "measurement")}
        seen = set()
        raw_path = root / "output" / arm / "artifacts/profile_export_raw.jsonl"
        for line in raw_path.open():
            raw = json.loads(line)
            session = raw["metadata"]["conversation_id"]
            if session not in failed:
                continue
            if session in seen:
                raise ValueError("Repeated failed session")
            seen.add(session)
            packets = [packet for response in raw.get("responses", []) for packet in response.get("packets", [])]
            admission = (raw.get("status") == 200 and bool(packets)
                         and all(packet.get("name") == ADMISSION_NOTICE and packet.get("value") is None
                                 for packet in packets))
            label = "HTTP200_plain_text_admission_limit_notice" if admission else "other_retained_failure"
            classified[label] += 1
            phases[failed[session]["logical_phase"]][label] += 1
        if seen != set(failed):
            raise ValueError("Failed attempts are missing from raw exports")
        sets[arm] = {row["session_id"] for row in data["metrics"]["per_request"]
                     if row["success"] and row["logical_phase"] == "measurement"}
        result[arm] = {"all_failed_attempts": len(failed), "classifications": dict(classified),
                       "logical_phases": {key: dict(value) for key, value in phases.items()},
                       "raw_export_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()}
    bf16, fp8 = sets["bf16"], sets["fp8_e4m3"]
    return {"classification": "confirmed_descriptive_trace_failure_analysis",
            "run_id": audit["run_id"], "independent_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "arms": result,
            "measurement_success_pairing": {"both_success": len(bf16 & fp8), "BF16_only": len(bf16 - fp8),
                                             "FP8_only": len(fp8 - bf16)},
            "interpretation": "HTTP200 does not establish success; admission-limit notices here are plain text "
                              "parsed as packet names, not JSON SSE error objects. Success-conditional latency "
                              "compares different completed subsets.",
            "GPU_used": False, "raw_or_official_measurements_modified": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "mirror", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.audit, args.mirror)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"run_id": result["run_id"], "arms": result["arms"],
                      "measurement_success_pairing": result["measurement_success_pairing"]}))


if __name__ == "__main__":
    main()
