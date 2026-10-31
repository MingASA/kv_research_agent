"""Leaf worker for a newly isolated non-root/network-none single-GPU container.

Only the host supervisor issues the admission lease. Graph, resident bytes,
real AIPerf completion and all fixed output lengths are checked before success.
The host deadline covers this worker and all child processes.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from .durable import digest, write_new_json
from .policy import MODEL, validate_execution_authority, validate_request


def file_sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def environment(arm: str) -> dict:
    env = {key: value for key, value in os.environ.items() if not key.startswith("XLLM_")}
    env.update(XLLM_FLASHINFER_NATIVE_KV="1", XLLM_KV_QUALITY_MODE="none", CUDA_VISIBLE_DEVICES="0",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               CUDA_CACHE_PATH="/run_cache/cuda", FLASHINFER_WORKSPACE_BASE="/run_cache/flashinfer",
               FLASHINFER_CACHE_DIR="/run_cache/flashinfer-cache", NO_PROXY="127.0.0.1,localhost",
               no_proxy="127.0.0.1,localhost", MAX_JOBS="4", CMAKE_BUILD_PARALLEL_LEVEL="4")
    # Preserve image CUDA compatibility libraries; replacing this path caused historical driver failures.
    prefix = "/workspace/.venv/lib/python3.12/site-packages/tvm_ffi/lib:/workspace/deps/prefix/lib"
    env["LD_LIBRARY_PATH"] = prefix + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    if arm == "fp8_e4m3":
        env.update(XLLM_QUANTIZED_BACKEND="flashinfer", XLLM_FP8_K_SCALE="0.125", XLLM_FP8_V_SCALE="0.0625")
    elif arm not in ("bf16", "nvfp4"):
        raise ValueError("Unknown paired arm")
    return env


def fixed_server_command(request: dict, arm: str) -> list[str]:
    return [request["server_binary"], "--model=/models/" + MODEL, "--model_id=" + MODEL,
            "--model_impl=python", "--python_model_path=" + request["source_repo"],
            "--host=127.0.0.1", "--port=19001", "--max_cache_size=0", "--enable_prefix_cache=false",
            "--max_memory_utilization=0.93", "--max_tokens_per_batch=4096",
            "--max_seqs_per_batch=" + str(max(8, request["concurrency"])), "--max_concurrent_requests=200",
            "--kv_cache_dtype=" + ("auto" if arm == "bf16" else "fp8_e4m3"),
            "--enable_graph=true", "--enable_prefill_piecewise_graph=false", "--python_graph_backend=cudagraphs"]


def bootstrap() -> dict:
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "Explain a CUDA warp in one sentence."}],
               "max_tokens": 16, "ignore_eos": True, "temperature": 0, "stream": True,
               "stream_options": {"include_usage": True}}
    request = urllib.request.Request("http://127.0.0.1:19001/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    done, tokens, finish = False, 0, None
    with urllib.request.urlopen(request, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            value = json.loads(data)
            if "error" in value:
                raise ValueError("Bootstrap SSE failed")
            if value.get("usage"):
                tokens = value["usage"]["completion_tokens"]
            for choice in value.get("choices", []):
                finish = choice.get("finish_reason") or finish
    if not done or tokens != 16 or finish != "length":
        raise ValueError("Bootstrap did not force real decode/Graph replay")
    return {"done": done, "tokens": tokens, "finish": finish, "payload": payload}


def fixed_lengths(phases: dict) -> None:
    for rows in phases.values():
        for row in rows:
            if row["metrics"]["output_token_count"]["value"] != 512:
                raise ValueError("Fixed-decode request stopped early or changed output length")


def stop_child(process: subprocess.Popen) -> None:
    if process.poll() is None:
        # This process group was created by this worker in a private PID namespace.
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def monitor(process: subprocess.Popen, server: subprocess.Popen | None, path: Path,
            *, deadline_ns: int | None = None) -> None:
    with path.open("x") as stream:
        writer = csv.writer(stream)
        writer.writerow(["utc", "memory_used_mib", "compute_util_pct", "memory_util_pct"])
        while process.poll() is None:
            if deadline_ns is not None and time.monotonic_ns()>=deadline_ns:
                raise TimeoutError('Owned measurement exceeded its declared bounded wall window')
            if server is not None and server.poll() is not None:
                raise ValueError("xLLM server died during actual measurement")
            value = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,utilization.memory",
                                             "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
            fields = [item.strip() for item in value.split(",")]
            if len(fields) != 3 or not all(item.isdigit() for item in fields):
                raise ValueError("Single-GPU telemetry visibility changed")
            writer.writerow([datetime.datetime.now(datetime.timezone.utc).isoformat(), *fields])
            stream.flush()
            time.sleep(1)


def run_fixed(ticket: dict, root: Path, output: Path, arm: str, env: dict) -> dict:
    from . import evidence
    sys.path.insert(0, str(root / "runtime_support"))
    helpers = importlib.import_module("aiperf_h800_v4_2_native")
    request = ticket["request"]
    parent = json.loads(Path(request["dataset_path"]).read_text())
    if len(parent["data"]) != 136 or any(payload.get("ignore_eos") is not True or payload.get("max_tokens") != 512
                                         for session in parent["data"] for payload in session["payloads"]):
        raise ValueError("Frozen fixed-decode controls differ before model load")
    command = fixed_server_command(request, arm)
    write_new_json(output / "server_command.json", command)
    server, client = None, None
    with (output / "server.log").open("x") as log:
        try:
            server = subprocess.Popen(command, cwd=request["source_repo"], env=env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise ValueError("xLLM died before readiness")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:19001/v1/models", timeout=3) as response:
                        models = json.load(response)
                    if any(row.get("id") == MODEL for row in models.get("data", [])):
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Model startup did not complete within 600s")
            write_new_json(output / "bootstrap.json", bootstrap())
            backend = helpers.native_backend_evidence(output / "server.log", arm, True)
            allocation = helpers.native_allocation(output / "server.log", arm)
            evidence.verify_backend((output / "server.log").read_text(), backend, arm)
            evidence.verify_allocation(allocation, arm)
            write_new_json(output / "native_backend.json", backend)
            write_new_json(output / "allocation.json", allocation)
            manifest = {"seed": 20261020, "request_count": 128, "warmup_count": 8,
                        "source": {"kind": "inputs_json", "container_path": request["dataset_path"], "sha256": request["dataset_sha256"]}}
            perf = helpers.perf_command(manifest, request["concurrency"], output / "artifacts")
            aperf_env = env.copy()
            aperf_env.update(AIPERF_DATASET_CONFIGURATION_TIMEOUT="1200", AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="1200",
                             AIPERF_DATASET_MMAP_CACHE_ENABLED="true", AIPERF_DATASET_MMAP_CACHE_DIR="/run_cache/aiperf_mmap",
                             PYTHONPATH=str(root / "runtime_support/tokenizer_compat"), CJYU_AIPERF_LOCAL_TOKENIZER_SHIM="1")
            aperf_env.pop("HF_HUB_OFFLINE", None)
            aperf_env.pop("TRANSFORMERS_OFFLINE", None)
            write_new_json(output / "aiperf_command.json", perf)
            with (output / "aiperf.log").open("x") as perf_log:
                client = subprocess.Popen(perf, cwd=root / "runtime_support", env=aperf_env, stdout=perf_log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                monitor(client, server, output / "gpu_telemetry.csv")
                if client.wait() != 0:
                    raise ValueError("AIPerf actually failed; no measurement success")
            artifacts = output / "artifacts"
            audit = helpers.audit_records_v33(artifacts, manifest)
            phases = evidence.read_records(artifacts / "profile_export.jsonl", 128, 8)
            fixed_lengths(phases)
            evidence.verify_summary(json.loads((artifacts / "profile_export_aiperf.json").read_text()), phases["profiling"])
            evidence.verify_backend((output / "server.log").read_text(), backend, arm)
            write_new_json(output / "request_audit.json", audit)
        finally:
            if client is not None:
                stop_child(client)
            if server is not None:
                stop_child(server)
    return {"protocol": "fixed_decode", "profiling_requests": 128, "warmup_requests": 8,
            "output_tokens_per_request": 512, "graph_capture_and_replay_verified": True,
            "resident_allocation_verified": True, "official_aiperf_complete": True}


def quality_command(ticket: dict, output: Path, arm: str) -> list[str]:
    request = ticket["request"]
    workload = request["workload"]
    data = str(Path(request["dataset_path"]).parent)
    return [sys.executable, "-B", workload["runner_path"], "--model", "/models/" + MODEL,
            "--server-binary", request["server_binary"], "--runtime-source-repo", request["source_repo"],
            "--runner-source-commit", workload["runner_source_commit"],
            "--binary-source-commit", request["server_source_commit"],
            "--attention-backend", "native-flashinfer", "--graph-mode", "cudagraphs",
            "--fp8-k-scale", "0.125", "--fp8-v-scale", "0.0625", "--cohort-scope", "pilot",
            "--arms", "auto" if arm == "bf16" else "fp8_e4m3", "--gpu", "0", "--port", "19001",
            "--concurrency", "1", "--max-seqs-per-batch", "8", "--max-tokens-per-batch", "4096",
            "--max-concurrent-requests", "200", "--max-memory-utilization", "0.93", "--max-cache-size", "0",
            "--gsm8k", data + "/gsm8k.jsonl", "--gsm8k-shots", workload["gsm8k_shots"],
            "--longbench-data", data, "--longbench-repo", workload["longbench_repo"],
            "--approved-data-manifest", workload["approved_data_manifest"], "--output", str(output / "quality")]


def run_quality(ticket: dict, output: Path, arm: str, env: dict) -> dict:
    command = quality_command(ticket, output, arm)
    write_new_json(output / "quality_command.json", command)
    with (output / "quality.log").open("x") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            monitor(process, None, output / "gpu_telemetry.csv")
            if process.wait() != 0:
                raise ValueError("Native quality runner failed; preserve journals and raw logs")
        finally:
            stop_child(process)
    path = output / "quality"
    complete = json.loads((path / "complete.json").read_text())
    if complete["summary_sha256"] != file_sha(path / "summary.json") or complete["csv_sha256"] != file_sha(path / "summary.csv"):
        raise ValueError("Quality completion/summary hash mismatch")
    journal = json.loads((path / "experiment_manifest.json").read_text())
    audit = journal["data_audit"]["datasets"]
    if sum(item["selected_rows"] for item in audit.values()) != 80 or journal["plan"]["scope"] != "protocol_pilot_no_full_quality_claim":
        raise ValueError("Native quality pilot cohort differs")
    return {"protocol": "native_quality_pilot", "samples_per_arm": 80,
            "scope": "protocol_pilot_no_full_quality_claim", "summary_sha256": complete["summary_sha256"],
            "graph_capture_and_replay_verified": True, "resident_allocation_verified": True,
            "paired_quality_claim": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticket", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    ticket = json.loads(args.ticket.read_text())
    request = ticket["request"]
    if digest(request) != ticket["request_sha256"] or args.arm not in request["arms"]:
        raise ValueError("Immutable worker ticket differs")
    validate_execution_authority(ticket["approval"], datetime.datetime.now(datetime.timezone.utc))
    validate_request(request, ticket["approval"])
    output, root = Path("/results"), args.ticket.parent
    # No torch import/model load before durable PID/cgroup watchdog admission.
    until = time.monotonic() + 25
    while not (output / "lease.json").exists() and time.monotonic() < until:
        time.sleep(.05)
    lease = json.loads((output / "lease.json").read_text())
    if lease != {"request_sha256": ticket["request_sha256"], "reservation_id": ticket["reservation_id"]}:
        raise ValueError("No matching watchdog admission lease")
    uuids = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=10).split()
    if uuids != request["gpu_uuids"] or os.environ.get("EXPECTED_GPU_UUID") != uuids[0]:
        raise ValueError("Container driver sees an unapproved GPU set")
    for name, expected in ticket["assets_sha256"].items():
        if file_sha(root / name) != expected:
            raise ValueError("Immutable runtime/input asset changed")
    for name, expected in ((request["server_binary"], request["binary_sha256"]),
                           (request["dataset_path"], request["dataset_sha256"]),
                           ("/models/" + (request["model_id"] if request.get("schema") == "kv-nextgen-run-v1" else MODEL)
                            + "/config.json", request["model_config_sha256"])):
        if file_sha(Path(name)) != expected:
            raise ValueError("Binary/input/model config identity drift")
    env = environment(args.arm)
    write_new_json(output / "runtime_environment.json", {key: value for key, value in env.items()
                   if key.startswith(("XLLM_", "FLASHINFER_")) or key in ("LD_LIBRARY_PATH", "PYTHONPATH")})
    if request["schema"] == "kv-nextgen-run-v1":
        from .nextgen import run_validation
        result = run_validation(ticket, root, output, args.arm, env)
    elif request["schema"] == "kv-production-run-v1":
        if request['workload']['kind'] == 'native_prefill_workspace_operator':
            from .prefill_operator import run_operator
            result = run_operator(ticket, output, args.arm, env)
        elif request['workload']['kind'] == 'native_bf16q_decode_operator':
            from .bf16q_operator import run_operator
            result = run_operator(ticket, output, args.arm, env)
        else:
            from .production import run_service, run_candidate_quality
            result = (run_candidate_quality(ticket, root, output, args.arm, env)
                      if request["workload"]["kind"] in ("native_quality_fa3", "native_quality_xqa", "native_quality_xqa_bf16_reference", "native_quality_xqa_production_batch", "native_quality_bf16q_production_batch") else run_service(ticket, root, output, args.arm, env))
    else:
        result = (run_fixed(ticket, root, output, args.arm, env) if request["workload"]["kind"] == "fixed_decode"
                  else run_quality(ticket, output, args.arm, env))
    for name, expected in request["backend_source_sha256"].items():
        if file_sha(Path(request.get("python_source_repo", request["source_repo"])) / name) != expected:
            raise ValueError("Backend/Graph source changed while the measurement ran")
    result.update(measurement_success=True, arm=args.arm, request_sha256=ticket["request_sha256"],
                  evidence_sha256={str(path.relative_to(output)): file_sha(path)
                                   for path in output.rglob("*") if path.is_file()})
    write_new_json(output / "measurement.json", result)


if __name__ == "__main__":
    main()
