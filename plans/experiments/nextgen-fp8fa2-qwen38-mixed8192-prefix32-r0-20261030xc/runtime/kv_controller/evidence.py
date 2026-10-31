#!/usr/bin/env python3
"""Independently audit local mirrors of completed v4.2 native KV experiments.

Reads evidence only; writes a NEW local report directory. Never imports the
historical runner, invokes SSH, modifies a mirror, or launches a model.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


class AuditError(ValueError):
    """Evidence is missing, inconsistent or insufficient for this audit."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _value(metrics: dict, name: str, key: str = "value") -> float:
    item = metrics.get(name, {})
    number = item.get(key) if isinstance(item, dict) else None
    _require(type(number) in (int, float) and math.isfinite(number),
             f"Missing/nonfinite metric {name}.{key}")
    return float(number)


def _close(actual: float, expected: float, message: str) -> None:
    _require(math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-6),
             f"{message}: actual={actual}, expected={expected}")


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    left = math.floor(position)
    right = math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def _markers(text: str, prefix: str) -> list[dict]:
    return [json.loads(line.split(prefix, 1)[1]) for line in text.splitlines() if prefix in line]


def verify_backend(text: str, backend: dict, arm: str) -> dict:
    expected = "FlashInferBackend" if arm == "bf16" else "FlashInferFP8Backend"
    rows = _markers(text, "NATIVE_KV_BACKEND ")
    _require(len(rows) == 1 and rows[0] == backend, "Backend JSON disagrees with actual server log")
    required = {"class": expected, "attention_impl": "flashinfer", "planner_backend": "fa2",
                "paged_prefill": True, "quality_transform": False, "decode_tensor_cores": True,
                "graph_enabled": True, "graph_runner": "DecodeCudaGraphRunner",
                "kv_cache_dtype": "auto" if arm == "bf16" else "fp8_e4m3"}
    for key, value in required.items():
        _require(backend.get(key) == value, f"Backend evidence mismatch: {key}")
    _require(backend.get("flashinfer_version") == "0.6.18.post1", "FlashInfer version mismatch")
    captured = backend.get("graph_captured_batches", [])
    _require(isinstance(captured, list) and bool(captured)
             and all(type(n) is int and n > 0 for n in captured), "Missing captured Graph buckets")
    replays = _markers(text, "NATIVE_KV_GRAPH_REPLAY ")
    _require(bool(replays), "Missing actual Graph replay evidence")
    for row in replays:
        _require(row.get("class") == expected and row.get("captured_batch_size") in captured,
                 "Graph replay did not use a captured native bucket")
        _require(type(row.get("batch_size")) is int
                 and 0 < row["batch_size"] <= row["captured_batch_size"], "Invalid Graph replay batch")
    if arm == "fp8_e4m3":
        _require(backend.get("fp8_views_alias_resident_pages") is True, "FP8 views do not alias resident pages")
        _require(backend.get("kv_data_type") == "torch.float8_e4m3fn", "Not native FP8 views")
        _require(backend.get("k_scale") == 0.125 and backend.get("v_scale") == 0.0625,
                 "Unexpected fixed FP8 scales")
    return {"captured_batches": captured, "replay_markers": replays}


def verify_allocation(allocation: dict, arm: str) -> dict:
    layers = allocation.get("layers", [])
    _require(len(layers) == 48, "Expected Qwen14B 48 resident KV layers")
    expected_format = "bf16" if arm == "bf16" else "fp8_e4m3"
    expected_names = {"key", "value"} if arm == "bf16" else {"key", "value", "key_scale", "value_scale"}
    element_sizes = {"torch.bfloat16": 2, "torch.uint8": 1, "torch.float32": 4}
    capacity = None
    total = 0
    for layer in layers:
        _require(layer.get("key_format") == layer.get("value_format") == expected_format,
                 "Resident KV format mismatch")
        tensors = layer.get("tensors", {})
        _require(set(tensors) == expected_names, "Resident tensor set mismatch")
        shape = tensors["key"]["shape"]
        _require(len(shape) == 4 and shape[1:] == [128, 8, 128]
                 and type(shape[0]) is int and shape[0] > 0, "Invalid Qwen14B paged KV shape")
        token_capacity = shape[0] * shape[1]
        capacity = token_capacity if capacity is None else capacity
        _require(token_capacity == capacity, "KV capacity differs across layers")
        resident = 0
        for name, tensor in tensors.items():
            expected_shape = shape[:3] if name.endswith("_scale") else shape
            dtype = "torch.float32" if name.endswith("_scale") else (
                "torch.bfloat16" if arm == "bf16" else "torch.uint8")
            _require(tensor.get("shape") == expected_shape and tensor.get("dtype") == dtype,
                     "Resident tensor dtype/shape mismatch")
            size = math.prod(expected_shape) * element_sizes[dtype]
            _require(tensor.get("bytes") == size, "Resident tensor byte accounting mismatch")
            resident += size
        _require(layer.get("resident_bytes") == resident, "Layer resident byte total mismatch")
        expected_bpt = 4096 if arm == "bf16" else 2112
        _require(layer.get("bytes_per_token") == resident // token_capacity == expected_bpt,
                 "Resident bytes/token mismatch")
        total += resident
    _require(allocation.get("resident_bytes") == total, "Total resident byte accounting mismatch")
    _require(allocation.get("token_capacity") == capacity, "Total KV token capacity mismatch")
    return {"token_capacity": capacity, "pages_per_layer": layers[0]["tensors"]["key"]["shape"][0],
            "bytes_per_token_per_layer": layers[0]["bytes_per_token"], "resident_bytes": total}


def read_records(path: Path, measured: int, warmup: int) -> dict[str, list[dict]]:
    phases: dict[str, list[dict]] = {"profiling": [], "warmup": []}
    ids = set()
    keys = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metadata = row.get("metadata", {})
        phase = metadata.get("benchmark_phase")
        _require(phase in phases, "Unknown/missing benchmark phase")
        _require(row.get("error") is None and metadata.get("was_cancelled") is False
                 and metadata.get("context_overflow_skip") is False, "Request failed/cancelled/skipped")
        request_id = metadata.get("x_request_id")
        _require(isinstance(request_id, str) and bool(request_id) and request_id not in ids,
                 "Missing/duplicate request ID")
        ids.add(request_id)
        session = metadata.get("conversation_id")
        _require(isinstance(session, str) and bool(session) and metadata.get("turn_index") == 0,
                 "Missing conversation ID or unexpected multi-turn synthetic record")
        key = (phase, session, metadata["turn_index"])
        _require(key not in keys, "Duplicate conversation within phase")
        keys.add(key)
        metrics = row.get("metrics", {})
        output = _value(metrics, "output_token_count")
        prompt = _value(metrics, "input_sequence_length")
        _require(output > 1 and prompt > 0 and output.is_integer() and prompt.is_integer(),
                 "Invalid token counts")
        _close(output, _value(metrics, "usage_completion_tokens"), "Server completion counts differ")
        _close(output, _value(metrics, "output_sequence_length"), "Output length/count differ")
        _close(prompt, _value(metrics, "usage_prompt_tokens"), "Server prompt counts differ")
        _close(prompt + output, _value(metrics, "usage_total_tokens"), "Server total counts differ")
        for name in ("request_latency", "time_to_first_token", "inter_token_latency", "decode_duration"):
            _require(_value(metrics, name) > 0, f"Nonpositive {name}")
        _close(_value(metrics, "decode_duration") / (output - 1),
               _value(metrics, "inter_token_latency"), "ITL/decode duration inconsistency")
        _require(type(metadata.get("request_start_ns")) is int
                 and type(metadata.get("request_end_ns")) is int
                 and metadata["request_end_ns"] > metadata["request_start_ns"], "Invalid request timestamps")
        phases[phase].append(row)
    _require(len(phases["profiling"]) == measured and len(phases["warmup"]) == warmup,
             "Profiling/warmup record count mismatch")
    warm_sessions = {row["metadata"]["conversation_id"] for row in phases["warmup"]}
    profile_sessions = {row["metadata"]["conversation_id"] for row in phases["profiling"]}
    _require(not warm_sessions.intersection(profile_sessions), "Warmup/profiling source overlap")
    return phases


def verify_summary(summary: dict, records: list[dict]) -> None:
    count = len(records)
    _require(summary.get("was_cancelled") is False and summary.get("is_complete") is True
             and not summary.get("error_summary"), "AIPerf summary incomplete/cancelled/errored")
    _require(summary.get("aiperf_version") == "0.13.0", "Unpinned AIPerf version")
    _close(_value(summary, "request_count", "avg"), count, "Summary request count")
    for name in ("request_latency", "time_to_first_token", "inter_token_latency",
                 "input_sequence_length", "output_sequence_length", "output_token_count"):
        values = [_value(row["metrics"], name) for row in records]
        metric = summary.get(name, {})
        _require(metric.get("count") == count, f"Summary {name}.count mismatch")
        _close(_value(summary, name, "avg"), statistics.mean(values), f"Summary {name}.avg")
        _close(_value(summary, name, "sum"), sum(values), f"Summary {name}.sum")
        for label, quantile in (("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99)):
            _close(_value(summary, name, label), _percentile(values, quantile), f"Summary {name}.{label}")


def _options(command: list[str], option: str) -> list[str]:
    return [command[index + 1] for index, item in enumerate(command[:-1]) if item == option]


def telemetry_output(path: Path, records: list[dict]) -> dict:
    """Summarize only profiling-span NVML samples; utilization is not bandwidth."""
    start = min(row["metadata"]["request_start_ns"] for row in records) / 1e9
    end = max(row["metadata"]["request_end_ns"] for row in records) / 1e9
    with path.open(newline="") as stream:
        samples = []
        for row in csv.DictReader(stream):
            timestamp = datetime.datetime.fromisoformat(row["utc"])
            _require(timestamp.tzinfo is not None, "GPU telemetry timestamp lacks timezone")
            seconds = timestamp.timestamp()
            if not start <= seconds <= end:
                continue
            memory = int(row["memory_used_mib"])
            compute = int(row["compute_util_pct"])
            controller = int(row["memory_util_pct"])
            _require(memory > 0 and 0 <= compute <= 100 and 0 <= controller <= 100,
                     "Invalid GPU profiling telemetry")
            samples.append((seconds, memory, compute, controller))
    _require(bool(samples), "No GPU telemetry samples in profiling span")
    samples.sort()
    return {"gpu_profile_samples": len(samples),
            "gpu_memory_used_mib_min": min(row[1] for row in samples),
            "gpu_memory_used_mib_max": max(row[1] for row in samples),
            "gpu_memory_used_mib_p50": statistics.median(row[1] for row in samples),
            "gpu_compute_util_pct_mean": statistics.mean(row[2] for row in samples),
            "gpu_memory_controller_util_pct_mean": statistics.mean(row[3] for row in samples),
            "gpu_telemetry_span_fraction": (samples[-1][0] - samples[0][0]) / (end - start),
            "gpu_telemetry_max_gap_s": max((right[0] - left[0] for left, right in zip(samples, samples[1:])), default=0)}


def _metric_output(case: Path, records: list[dict], summary: dict, cap: int) -> dict:
    values = lambda name: [_value(row["metrics"], name) for row in records]
    tokens = sum(values("output_token_count"))
    # Installed AIPerf 0.13 uses observation_duration() for throughput. Its
    # benchmark_duration metric instead spans first request to last response.
    # Exported phase start/end retain microseconds, so compare at that precision.
    duration = (datetime.datetime.fromisoformat(summary["end_time"])
                - datetime.datetime.fromisoformat(summary["start_time"])).total_seconds()
    _require(duration > 0, "Nonpositive AIPerf observation duration")
    _close(_value(summary, "output_token_throughput", "avg"), tokens / duration, "Throughput from actual tokens")
    _close(_value(summary, "request_throughput", "avg"), len(records) / duration, "RPS from actual requests")
    start = min(row["metadata"]["request_start_ns"] for row in records)
    end = max(row["metadata"]["request_end_ns"] for row in records)
    span = (end - start) / 1e9
    _require(0.95 <= span / duration <= 1.05, "Request timestamp span disagrees with benchmark duration")
    result = {"profiling_records": len(records), "output_tokens": int(tokens),
              "profile_observation_duration_s": duration,
              "benchmark_duration_metric_s": _value(summary, "benchmark_duration", "avg"),
              "raw_request_span_s": span,
              "output_tps": tokens / duration, "raw_span_output_tps": tokens / span,
              "request_rps": len(records) / duration, "isl_mean": statistics.mean(values("input_sequence_length")),
              "osl_cap": cap, "osl_mean": statistics.mean(values("output_sequence_length")),
              "osl_min": min(values("output_sequence_length")), "osl_max": max(values("output_sequence_length")),
              "at_output_cap": sum(value == cap for value in values("output_sequence_length")),
              "osl_mismatch_gt5pct": sum(abs(value - cap) > cap * 0.05 for value in values("output_sequence_length")),
              "token_weighted_decode_ms": sum(values("decode_duration")) / (tokens - len(records)),
              "records_sha256": _sha(case / "artifacts/profile_export.jsonl")}
    for prefix, name in (("ttft", "time_to_first_token"), ("itl", "inter_token_latency"), ("e2e", "request_latency")):
        for label, quantile in (("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99)):
            result[f"{prefix}_{label}_ms"] = _percentile(values(name), quantile)
    return result


def audit_run(root: Path, files_root: Path, provenance: dict) -> dict:
    manifest = _load(root / "manifest.json")
    _require(manifest["schema"] == "cjyu-aiperf-h800-v4.2-native-1", "Unsupported manifest schema")
    _require(manifest["expected_native_commit"] == manifest["repo_heads"]
             == "ceff00bbd38054d2b6b4aa44513d773f2bd174b2", "Wrong ceff source version")
    _require(manifest["binary_sha256"] == "277bd8f391fbef978c86d77a976d488d567b0781bf19c0cc8a9c6763fccb21f3",
             "Unexpected candidate binary")
    _require(manifest["repo_tracked_changes"] == "" and manifest["prefix_cache"] is False
             and manifest["graph_mode"] == "cudagraphs" and manifest["cuda_graph"] is True,
             "Dirty source/prefix-cache/Graph mode mismatch")
    _require(manifest["arms"] == ["bf16", "fp8_e4m3"] and manifest["gpus"] == [2],
             "Not a same-GPU sequential native pair")
    _require((root / "COMPLETE").is_file() and bool((root / "COMPLETE").read_text().strip()), "Missing root COMPLETE")
    _require(manifest["request_count"] == 128 and manifest["warmup_count"] == 8, "Unexpected request protocol")
    origin = manifest["source"]["origin"]
    _require(origin["kind"] == "synthetic" and origin["canonical_sha256"] == manifest["source"]["sha256"],
             "Frozen canonical source mismatch")
    for relative, field in (("aiperf_h800_v4_2_native.py", "script_sha256"),
                            ("aiperf_h800_v2_1.py", "old_script_sha256"),
                            ("aiperf_v3_tokenizer_compat/sitecustomize.py", "tokenizer_shim_sha256")):
        _require(_sha(files_root / "support/inputs/aiperf" / relative) == manifest[field], "Runner/shim hash drift")
    checks = next(row for row in provenance["provenance_checks"] if row["run"] == root.name)
    _require(all(row["matches"] and row["expected"] == row["observed"] for row in checks["checks"]),
             "Current source/input provenance hash mismatch")
    config_hashes = [row["sha256"] for row in provenance["model_metadata"] if row["path"].endswith("/config.json")]
    _require(config_hashes == [manifest["model_config_sha256"]], "Model config hash mismatch")
    rows = []
    jobs = []
    commands = []
    phase_records = {}
    warnings = []
    for job in manifest["jobs"]:
        arm = job["arm"]
        directory = root / "jobs" / f'w{job["wave"]}-g{job["gpu"]}-{job["group"]}-{arm}'
        state = _load(directory / "state.json")
        _require(state["status"] == "complete" and (directory / "COMPLETE").is_file(), "Arm not actually complete")
        backend = _load(directory / "native_backend.json")
        text = (directory / "server.log").read_text(errors="replace")
        graph = verify_backend(text, backend, arm)
        allocation = _load(directory / "allocation.json")
        _require(_markers(text, "KV_ALLOCATION ") == [allocation], "Allocation JSON differs from real server log")
        storage = verify_allocation(allocation, arm)
        docker = _load(directory / "docker_command.json")
        _require(_options(docker, "--gpus") == ["device=GPU-a6169058-34c8-6edc-e4f5-779bf9a10421"], "Wrong GPU UUID/isolation argv")
        _require(_options(docker, "--network") == ["none"] and _options(docker, "--ipc") == ["private"]
                 and _options(docker, "--user") == ["1005:1005"] and "--privileged" not in docker,
                 "Unexpected privileged/network/identity Docker configuration")
        mounts = _options(docker, "-v")
        _require(not any("docker.sock" in mount for mount in mounts), "Docker socket exposed")
        if any(mount.endswith(":rw") and not mount.endswith(":/results:rw") for mount in mounts):
            warnings.append("Historical Docker has shared project caches mounted rw; do not reuse unchanged for new isolated jobs.")
        command = _load(directory / "server_command.json")
        _require("--host=127.0.0.1" in command, "Server did not bind loopback")
        commands.append([arg for arg in command if not arg.startswith("--kv_cache_dtype=")])
        jobs.append({"arm": arm, "graph": graph, "storage": storage, "docker_mounts": mounts})
        for concurrency in manifest["concurrencies"]:
            info = state["cases"][str(concurrency)]
            _require(info["status"] == "success" and info["rc"] == 0 and info["attempt"] == 1,
                     "Missing successful first attempt; retries need a separate audit")
            case = directory / f"case-{concurrency}" / f'attempt-{info["attempt"]:03}'
            recorded = _load(case / "request_audit.json")
            for name, relative in (("record_file_sha256", "profile_export.jsonl"),
                                   ("root_export_sha256", "profile_export_aiperf.json"),
                                   ("profiling_export_sha256", "phases/profiling/profile_export_aiperf.json"),
                                   ("warmup_export_sha256", "phases/warmup/profile_export_aiperf.json")):
                _require(recorded[name] == _sha(case / "artifacts" / relative), "Recorded audit hash mismatch")
            _require(info["summary_sha256"] == _sha(case / "artifacts/profile_export_aiperf.json"), "State summary hash mismatch")
            protocol = _load(case / "protocol.json")
            _require(protocol["source_sha256"] == manifest["source"]["sha256"]
                     and protocol["request_count"] == 128 and protocol["warmup_count"] == 8, "Protocol mismatch")
            phases = read_records(case / "artifacts/profile_export.jsonl", 128, 8)
            summary = _load(case / "artifacts/profile_export_aiperf.json")
            verify_summary(summary, phases["profiling"])
            verify_summary(_load(case / "artifacts/phases/profiling/profile_export_aiperf.json"), phases["profiling"])
            verify_summary(_load(case / "artifacts/phases/warmup/profile_export_aiperf.json"), phases["warmup"])
            output = _metric_output(case, phases["profiling"], summary, origin["osl"])
            output.update(telemetry_output(case / "gpu_telemetry.csv", phases["profiling"]))
            _require(output["output_tokens"] == recorded["observed_output_tokens"], "Recomputed token total differs from old audit")
            _require(output["osl_mismatch_gt5pct"] == recorded["osl_mismatch_gt5pct"], "Recomputed OSL mismatch differs from old audit")
            phase_records[(arm, concurrency)] = phases
            rows.append({"run_id": manifest["run_id"], "run_directory": root.name, "arm": arm,
                         "concurrency": concurrency, "isl_nominal": origin["isl"], **output})
    _require(len(commands) == 2 and commands[0] == commands[1], "Server argv differs beyond KV dtype")
    pairs = []
    for concurrency in manifest["concurrencies"]:
        before = next(row for row in rows if row["arm"] == "bf16" and row["concurrency"] == concurrency)
        after = next(row for row in rows if row["arm"] == "fp8_e4m3" and row["concurrency"] == concurrency)
        for phase in ("warmup", "profiling"):
            identity = lambda row: (row["metadata"]["conversation_id"], row["metadata"]["turn_index"],
                                    _value(row["metrics"], "input_sequence_length"))
            _require(sorted(map(identity, phase_records[("bf16", concurrency)][phase]))
                     == sorted(map(identity, phase_records[("fp8_e4m3", concurrency)][phase])),
                     "Effective phase/session/input-length pairing failed")
        pair = {"run_id": manifest["run_id"], "run_directory": root.name,
                "isl_nominal": origin["isl"], "osl_cap": origin["osl"], "concurrency": concurrency}
        for name in ("output_tps", "request_rps", "osl_mean", "itl_p50_ms", "ttft_p95_ms", "e2e_p50_ms"):
            pair[f"bf16_{name}"] = before[name]
            pair[f"fp8_{name}"] = after[name]
            pair[f"{name}_change_pct"] = 100 * (after[name] / before[name] - 1)
        pair["run_replicates"] = 1
        pair["inference_scope"] = "descriptive_variable_output_service_load; no matched-quality or fixed-decode-speed claim"
        pairs.append(pair)
    return {"run_id": manifest["run_id"], "run_directory": root.name, "manifest_sha256": _sha(root / "manifest.json"),
            "measurement_integrity": "confirmed", "rows": rows, "pairs": pairs, "jobs": jobs,
            "warnings": sorted(set(warnings)), "quality": "unverified",
            "payload_bytes_observed_over_http": False,
            "weight_content_hashes_recorded": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", required=True, type=Path, help="Extracted files directory with SNAPSHOT.json")
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    snapshot = _load(args.mirror / "SNAPSHOT.json")
    for item in snapshot["files"]:
        path = args.mirror / item["relative_path"]
        _require(path.stat().st_size == item["size"] and _sha(path) == item["sha256"], "Mirror changed after collection")
    provenance = _load(args.provenance)
    reports = []
    for directory in sorted(args.mirror.glob("native*")):
        if (directory / "manifest.json").is_file():
            reports.append(audit_run(directory, args.mirror, provenance))
    _require(len(reports) == 6, "Expected six independently observed ceff run manifests")
    args.out.mkdir(parents=True, exist_ok=False)
    output = {"schema": "kv-independent-native-audit-v1", "script_sha256": _sha(Path(__file__)),
              "snapshot_sha256": _sha(args.mirror / "SNAPSHOT.json"),
              "provenance_sha256": _sha(args.provenance), "runs": reports,
              "case_count": sum(len(report["rows"]) for report in reports),
              "limitations": ["One run per load condition; requests share scheduler state and are not independent run replicates.",
                              "Output token limits are not fixed decoded lengths; natural-stop causes are not retained in request JSONL.",
                              "Only first replay is logged per server; no per-timed-step Graph coverage or HBM counters.",
                              "HTTP payload bytes, sampling determinism and matched native quality are not certified.",
                              "Model metadata hashes were collected now; full historical checkpoint content hashes are absent.",
                              "Historical idle checks used 5 seconds and shared rw caches; not sufficient for new policy-controlled scheduling."]}
    (args.out / "audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    for name, key in (("cases.csv", "rows"), ("pairs.csv", "pairs")):
        rows = [row for report in reports for row in report[key]]
        with (args.out / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"runs": len(reports), "cases": output["case_count"],
                      "profiling_requests": sum(row["profiling_records"] for report in reports for row in report["rows"]),
                      "report": str(args.out), "measurement_integrity": "confirmed",
                      "quality": "unverified", "gpu_launches": 0}))


if __name__ == "__main__":
    main()
