"""Bounded native hybrid-model validation on the existing guarded executor.

This is an initial BF16 correctness/lifecycle contract, not a throughput or
model-quality acceptance. Resource admission, watchdogs and accounting remain
owned by the shared supervisor and persistent ledger.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import hashlib
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
from types import SimpleNamespace

from .approval import ApprovalError
from .durable import write_new_json


SCHEMA = "kv-nextgen-run-v1"
KIND = "native_hybrid_model_validation"
MODEL = "Qwen3.8-27B"
SOURCE_FILES = (
    "xllm/python/attention/backend.py",
    "xllm/python/attention/native_kv.py",
    "xllm/python/attention/flashinfer.py",
    "xllm/python/attention/flashinfer_fp8.py",
    "xllm/python/attention/native_profile.py",
    "xllm/python/model_executor/executor.py",
    "xllm/python/model_executor/runners/decode_cuda_graph.py",
    "xllm/python/models/qwen3_5.py",
    "xllm/python/layers/cuda/qwen3_5/gated_delta_net.py",
    "xllm/core/runtime/worker_impl.cpp",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ApprovalError(message)


def validate_request(request: dict, approval: dict) -> None:
    from .policy import _path

    _require(request.get("schema") == SCHEMA, "Unknown next-generation contract")
    _require(request.get("model_id") == MODEL, "Model must match the verified cached model")
    _require(request.get("model_path") == approval["read_only_model_path"], "Model is outside current read-only scope")
    _require(Path(request["model_path"]).name == MODEL, "Model ID/path mismatch")
    _require(request.get("arms") == ["bf16"], "Initial hybrid validation only admits native BF16")
    _require(len(request.get("gpu_uuids", [])) == 1, "Initial native hybrid contract is single GPU")
    _require(type(request.get("max_wall_seconds")) is int and 600 <= request["max_wall_seconds"] <= 3600,
             "Initial validation needs a bounded startup/cleanup window")
    startup = request.get('startup_seconds', 900)
    _require(type(startup) is int and startup in (900, 1500) and startup <= request['max_wall_seconds'] - 600,
             'Startup window must preserve operator, service and cleanup time')
    recovery = request.get('startup_recovery')
    if startup == 1500:
        _require(request.get('replicate') == 1 and isinstance(recovery, dict)
                 and recovery.get('reason') == 'weight_load_exceeded_initial900s_limit'
                 and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', recovery.get('prior_run_id','')) is not None
                 and recovery['prior_run_id'] != request['run_id']
                 and re.fullmatch(r'[0-9a-f]{64}', recovery.get('prior_result_sha256','')) is not None,
                 'Longer startup requires one explicitly recorded verified startup recovery')
    else:
        _require(recovery is None and request.get('replicate') == 0, 'Initial startup contract cannot hide a repeated run')
    _require(request.get("image_id") == approval["approved_image_id"], "Image differs from existing pinned image")
    for field in ("source_repo", "python_source_repo", "server_binary", "dataset_path"):
        value = _path(request.get(field))
        _require(value.startswith(approval["approved_workspace"] + "/"), "Next-generation asset escaped project BASE")
    _require(request["python_source_repo"] == request["remote_output"] + "/source_repo", "Source checkout escaped run")
    _require(request["dataset_path"] == request["remote_output"] + "/input/validation.json", "Validation input escaped run")
    for field, size in (("server_source_commit", 40), ("python_source_commit", 40), ("binary_sha256", 64),
                        ("dataset_sha256", 64), ("model_config_sha256", 64)):
        _require(re.fullmatch(r"[0-9a-f]{" + str(size) + "}", request.get(field, "")) is not None,
                 "Missing next-generation provenance: " + field)
    hashes = request.get("backend_source_sha256", {})
    _require(set(hashes) == set(SOURCE_FILES) and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes.values()),
             "Missing native hybrid source/Graph identities")
    _require(request.get("graph_mode") == "cudagraphs" and request.get("prefix_cache") is True,
             "Production Graph and prefix lifecycle must be exercised")
    _require(request.get("workload") == {"kind": KIND, "scope": "correctness_cache_lifecycle_not_performance"},
             "Unknown hybrid workload")
    _require(request.get("service") == {"max_batch": 8, "max_tokens_per_batch": 2048,
                                       "linear_state_slots": 16, "max_context": 16384,
                                       "max_memory_utilization": 0.85}, "Unexpected validation service configuration")
    _require(request.get('max_context_scope')=='validation_input_budget_not_server_context_cap',
             'Validation input budget must not be confused with model context capacity')
    _require(request.get("operators") == {"bf16": {"planner": "fa3", "decode": "flashinfer"}},
             "Initial validation requires the explicit native BF16 FA3 path")


def verify_startup_recovery(request: dict, base: Path, uid: int) -> None:
    recovery = request.get('startup_recovery')
    if request.get('schema') != SCHEMA or recovery is None:
        return
    root = base/'agent_work/runs'/recovery['prior_run_id']
    for path in (root, root/'result.json', root/'ticket.json', root/'output/bf16/container.log'):
        _require(path.resolve(strict=True) == path and path.stat().st_uid == uid, 'Unowned startup-recovery evidence')
    raw = (root/'result.json').read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == recovery['prior_result_sha256'], 'Prior terminal result SHA differs')
    result=json.loads(raw)
    _require(result['status'] == 'FAILED' and result['gpu_release_verified'] is True
             and not result['live_owned_containers'] and not result['arm_results'], 'Prior startup is not a released failure')
    prior=json.loads((root/'ticket.json').read_text())['request']
    for field in ('model_id','model_path','model_config_sha256','binary_sha256','server_source_commit',
                  'python_source_commit','backend_source_sha256','dataset_sha256','operators','service',
                  'workload','graph_mode','prefix_cache','experiment_key','gpu_uuids'):
        if field=='service' and request[field] != prior[field]:
            old_service=dict(prior[field]);new_service=dict(request[field])
            old_bound=old_service.pop('max_context');new_bound=new_service.pop('max_context')
            correction=recovery.get('context_budget_correction',{})
            _require(old_service==new_service and old_bound==8192 and new_bound==16384
                     and correction.get('prior_nominal_bound')==8192 and correction.get('new_input_budget')==16384
                     and 8192<correction.get('validated_input_max_prompt_plus_output',0)<=16384
                     and re.fullmatch(r'[0-9a-f]{64}',correction.get('tokenization_evidence_sha256','')) is not None,
                     'Only the independently tokenized input-budget metadata correction is allowed')
            continue
        _require(request[field] == prior[field], 'Startup recovery changed model/workload/runtime: '+field)
    _require('TimeoutError: Native hybrid startup deadline exceeded' in (root/'output/bf16/container.log').read_text(),
             'Prior failure is not the recorded startup timeout')


def server_command(request: dict) -> list[str]:
    service = request["service"]
    return [request["server_binary"], "--model=/models/" + request["model_id"],
            "--model_id=" + request["model_id"], "--backend=llm", "--model_impl=python",
            "--python_model_path=" + request["python_source_repo"], "--host=127.0.0.1", "--port=19001",
            "--max_cache_size=0", "--max_memory_utilization=" + str(service["max_memory_utilization"]),
            "--max_linear_state_cache_slots=" + str(service["linear_state_slots"]),
            "--enable_prefix_cache=true", "--enable_chunked_prefill=true",
            "--max_tokens_per_batch=" + str(service["max_tokens_per_batch"]),
            "--max_seqs_per_batch=" + str(service["max_batch"]), "--max_concurrent_requests=32",
            "--kv_cache_dtype=auto", "--enable_graph=true", "--enable_prefill_piecewise_graph=false",
            "--python_graph_backend=cudagraphs"]


def _stream_case(case: dict, model_id: str) -> dict:
    payload = {"model": model_id, "messages": case["messages"], "temperature": 0,
               "max_tokens": 96, "stream": True, "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request("http://127.0.0.1:19001/v1/chat/completions",
                                     data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    raw, chunks, usage, finish, done, error = [], [], None, None, False, None
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            for line in response:
                text = line.decode("utf-8").strip()
                raw.append(text)
                if not text.startswith("data: "):
                    continue
                content = text[6:]
                if content == "[DONE]":
                    done = True
                    continue
                chunk = json.loads(content)
                chunks.append(chunk)
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    finish = choice.get("finish_reason") or finish
    except (OSError, ValueError) as failure:
        # Client/protocol boundary: preserve failure bytes and reject the run.
        error = type(failure).__name__ + ": " + str(failure)
    answer = "".join(choice.get("delta", {}).get("content", "")
                     for chunk in chunks for choice in chunk.get("choices", []))
    return {"id": case["id"], "request": payload, "raw_SSE": raw, "answer": answer, "usage": usage,
            "finish_reason": finish, "SSE_done": done,
            "error": error, "contract_valid": bool(error is None and done and usage and finish == "stop" and answer),
            "expected_substring_present": case["expected"] in answer}


def _operator_gate(output: Path) -> dict:
    """Exercise actual head_dim=256/GQA=6 FA3 and changed-input Graph replay."""
    import torch
    from tools.benchmark_kv_cache import _attention_metadata
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.flashinfer import FlashInferBackend

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (9, 0):
        raise ValueError("Native hybrid validation requires exactly the admitted SM90 GPU")
    torch.manual_seed(20261027)
    device = torch.device("cuda")
    batch, context, heads, kv_heads, dim, page_size = 2, 129, 24, 4, 256, 64
    key = torch.randn(batch, context, kv_heads, dim, dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)
    shape = (7, page_size, kv_heads, dim)
    cache = LayerCache(torch.zeros(shape, dtype=torch.bfloat16, device=device),
                       torch.zeros(shape, dtype=torch.bfloat16, device=device))
    linear = LayerCache(None, None, conv=torch.ones(2, 8, 3, device=device), ssm=torch.ones(2, 4, 8, 8, device=device))
    backend = FlashInferBackend(heads, kv_heads, dim, dim ** -0.5, -1, device, torch.bfloat16)
    backend.bind_kv_caches([linear, linear, linear, cache])
    pages = torch.tensor([[3, 1, 5], [4, 2, 6]], device=device, dtype=torch.int32)
    positions = torch.arange(context, device=device)
    slots = (pages[:, positions // page_size] * page_size + positions % page_size).flatten()
    cache.key.flatten(0, 1).index_copy_(0, slots.long(), key.flatten(0, 1))
    cache.value.flatten(0, 1).index_copy_(0, slots.long(), value.flatten(0, 1))
    metadata = _attention_metadata(pages, slots.view(batch, context)[:, -1].contiguous(), context, 1, page_size)
    metadata.paged_kv_indices = metadata.paged_kv_indices.clone()
    backend.prepare(metadata, graph_mode=True)
    query = torch.randn(batch, heads, dim, dtype=torch.bfloat16, device=device)
    layer = SimpleNamespace(layer_id=3, num_heads=heads, num_kv_heads=kv_heads, head_dim=dim)
    backend.execute_attention(query, layer)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = backend.execute_attention(query, layer)
    comparisons = []
    for _ in range(2):
        query.copy_(torch.randn_like(query))
        graph.replay()
        k = key.repeat_interleave(heads // kv_heads, dim=2).transpose(1, 2).float()
        v = value.repeat_interleave(heads // kv_heads, dim=2).transpose(1, 2).float()
        scores = query.float().unsqueeze(2) @ k.transpose(-1, -2) * dim ** -0.5
        reference = (torch.softmax(scores, -1) @ v).flatten(1)
        relative = float((actual.float() - reference).norm() / reference.norm())
        comparisons.append(relative)
        if not 0 <= relative <= 0.01:
            raise ValueError("Native hybrid-geometry FA3 decode/Graph reference mismatch")
    if not torch.all(linear.ssm == 1) or not torch.all(linear.conv == 1):
        raise ValueError("Binding or attention changed unrelated recurrent state")
    result = {"classification": "SM90_native_BF16_FA3_hybrid_geometry_operator_only",
              "heads": heads, "kv_heads": kv_heads, "head_dim": dim, "page_size": page_size,
              "real_capture_and_changed_input_replay": True, "relative_L2": comparisons,
              "linear_state_unchanged": True, "model_loaded": False}
    write_new_json(output / "operator_acceptance.json", result)
    del backend, cache, linear, graph, query, actual, key, value, k, v, scores, reference
    torch.cuda.empty_cache()
    return result


def run_validation(ticket: dict, root: Path, output: Path, arm: str, env: dict) -> dict:
    from .worker import stop_child

    request = ticket["request"]
    env = dict(env, XLLM_FLASHINFER_NATIVE_PLANNER="fa3", XLLM_FLASHINFER_NATIVE_DECODE="flashinfer")
    command = server_command(request)
    write_new_json(output / "server_command.json", command)
    write_new_json(output / "nextgen_environment.json", {k: v for k, v in env.items() if k.startswith("XLLM_")})
    with (output / "operator_acceptance.log").open("x") as log:
        subprocess.run([sys.executable, "-B", "-m", __name__, "--operator-output", str(output)],
                       env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
    operator = json.loads((output / "operator_acceptance.json").read_text())
    dataset = json.loads(Path(request["dataset_path"]).read_text())
    if dataset.get("schema") != "native-hybrid-validation-input-v1" or len(dataset.get("cases", [])) != 16:
        raise ValueError("Frozen hybrid validation input differs")
    server = None
    rows = []
    with (output / "server.log").open("x") as log:
        try:
            server = subprocess.Popen(command, cwd=request["python_source_repo"], env=env,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + request.get('startup_seconds', 900)
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise ValueError("Native hybrid server exited during startup")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:19001/v1/models", timeout=2) as response:
                        if any(row.get("id") == request["model_id"] for row in json.load(response).get("data", [])):
                            break
                except (OSError, ValueError):
                    time.sleep(1)
            else:
                raise TimeoutError("Native hybrid startup deadline exceeded")
            # First cold calls and exact repeats surround concurrent requests.
            # Raw responses are durable before any output/correctness assertion.
            journal = output / "validation_raw.jsonl"
            with journal.open("x") as stream:
                for cases, parallel in ((dataset["cases"][:4], False), (dataset["cases"][4:], True),
                                        (dataset["cases"][:4], False)):
                    if parallel:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                            batch = list(pool.map(lambda case: _stream_case(case, request["model_id"]), cases))
                    else:
                        batch = [_stream_case(case, request["model_id"]) for case in cases]
                    for row in batch:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        rows.append(row)
            text = (output / "server.log").read_text()
            allocations = [json.loads(line.split("KV_ALLOCATION ", 1)[1])
                           for line in text.splitlines() if "KV_ALLOCATION " in line]
            backends = [json.loads(line.split("NATIVE_KV_BACKEND ", 1)[1])
                        for line in text.splitlines() if "NATIVE_KV_BACKEND " in line]
            allocation = allocations[-1] if allocations else {}
            backend = backends[-1] if backends else {}
            repeated = all(a["answer"] == b["answer"] for a, b in zip(rows[:4], rows[-4:]))
            result = {"classification": "native_hybrid_BF16_service_validation",
                      "model_loaded": True, "requests": len(rows), "raw_contracts_valid": all(r["contract_valid"] for r in rows),
                      "known_answer_checks_passed": sum(r["expected_substring_present"] for r in rows),
                      "deterministic_repeated_requests_match": repeated,
                      "native_allocation": allocation, "native_backend": backend,
                      "operator": operator,
                      "real_Graph_replay_logged": "NATIVE_KV_GRAPH_REPLAY" in text,
                      "model_quality_validated": False, "service_performance_measured": False,
                      "official_aiperf_complete": False}
            write_new_json(output / "validation_result.json", result)
            if (not result["raw_contracts_valid"] or not repeated or result["known_answer_checks_passed"] != len(rows)
                    or not result["real_Graph_replay_logged"] or not backend.get("graph_enabled")
                    or allocation.get("full_attention_layer_ids") != list(range(3, 64, 4))
                    or len(allocation.get("linear_state_layers", [])) != 48
                    or allocation.get("linear_state_resident_bytes", 0) <= 0):
                raise ValueError("Hybrid model correctness, storage, recurrent lifecycle or real Graph gate failed")
            return result
        finally:
            if server is not None:
                stop_child(server)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-output", type=Path, required=True)
    _operator_gate(parser.parse_args().operator_output)
