"""Real conversational serving, explicit modern FA3, natural stopping and raw wire audit."""

from __future__ import annotations

import collections
import datetime
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request
from typing import Iterator

from .durable import digest, write_new_json, write_new_bytes
from . import evidence
from .worker import MODEL, file_sha, monitor, stop_child, bootstrap


def verify_backend(log: str, arm: str, operator: dict | None = None,
                   fp8_scales: dict | None = None) -> dict:
    scales = {'key':.125, 'value':.0625} if fp8_scales is None else fp8_scales
    if scales not in ({'key':.125,'value':.0625}, {'key':.125,'value':.125}):
        raise ValueError('Unsupported explicitly declared FP8 scale contract')
    rows = evidence._markers(log, "NATIVE_KV_BACKEND ")
    expected = "FlashInferBackend" if arm == "bf16" else "FlashInferFP8Backend"
    if len(rows) != 1:
        raise ValueError("Missing/duplicate real backend construction marker")
    row = rows[0]
    xqa = operator is not None and operator.get("decode") == "xqa"
    bf16q = operator is not None and operator.get('profile_scope')=='production_bf16q_decode'
    planner = operator["planner"] if operator is not None else "fa3"
    for field, value in {"class": expected, "attention_impl": "flashinfer", "planner_backend": planner,
                         "paged_prefill": True, "graph_enabled": True, "profile_scope": "production_bf16q_decode" if bf16q else "production_xqa" if xqa else "production_fa3",
                         "quality_transform": False, "kv_cache_dtype": "auto" if arm == "bf16" else "fp8_e4m3",
                         "q_data_type": "torch.bfloat16" if arm == "bf16" or xqa or bf16q else "torch.float8_e4m3fn",
                         "kv_data_type": "torch.bfloat16" if arm == "bf16" else "torch.float8_e4m3fn"}.items():
        if row.get(field) != value:
            raise ValueError("Production backend did not select the declared path: " + field)
    if xqa and any(row.get(key) != value for key, value in {
            "decode_impl": "xqa", "decode_internal_q_data_type": "torch.bfloat16" if arm == "bf16" else "torch.float8_e4m3fn",
            "xqa_enable_pdl": True,
            "xqa_unused_kv_policy": "finite_full_initialization",
            "xqa_max_context": 32768, "xqa_graph_padding": "reserved_page0_one_token_discard"}.items()):
        raise ValueError("XQA actual internal precision/padding/context differs")
    if bf16q and any(row.get(k)!=v for k,v in {'decode_impl':'flashinfer','decode_internal_q_data_type':'torch.bfloat16',
            'resident_initialization_policy':'finite_full_initialization','graph_padding_policy':'empty_indptr_rows_discarded',
            'full_BF16_KV_shadow':False}.items()):
        raise ValueError('BF16-Q actual precision/padding/resident storage differs')
    if arm != "bf16" and (row.get("q_scale") != (None if xqa or bf16q else 1.0) or row.get("k_scale") != scales['key']
                          or row.get("v_scale") != scales['value'] or row.get("fp8_views_alias_resident_pages") is not True):
        raise ValueError("FP8 query/cache scales or actual resident alias differ")
    captures = row.get("graph_captured_batches", [])
    replays = evidence._markers(log, "NATIVE_KV_GRAPH_REPLAY ")
    if not captures or not replays or any(item.get("class") != expected or item.get("captured_batch_size") not in captures for item in replays):
        raise ValueError("No real native FA3 Graph capture/replay")
    return row


def _jsonl_rows(path: Path) -> Iterator[dict]:
    """JSONL is framed by physical LF; Unicode separators belong to strings."""
    with path.open('rb') as stream:
        for line in stream:
            yield json.loads(line.decode('utf-8'))


def _verify_observation_rates(summary: dict, output_tokens: int, requests: int) -> dict:
    """Verify rates against the resolution actually retained by ISO exports.

    AIPerf uses integer-nanosecond window bounds for rates, but exports each
    bound through datetime.fromtimestamp(ns / 1e9), retaining microseconds.
    Keep count/usage checks exact and bound only this timestamp conversion.
    """
    start=datetime.datetime.fromisoformat(summary['start_time'])
    end=datetime.datetime.fromisoformat(summary['end_time'])
    duration=(end-start).total_seconds()
    if duration<=0:raise ValueError('Nonpositive exported observation interval')
    bound_ns=math.ceil(1000+1e9*.5*(math.ulp(start.timestamp())+math.ulp(end.timestamp())))+2
    implied={}
    for name,count in (('output_token_throughput',output_tokens),('request_throughput',requests)):
        rate=evidence._value(summary,name,'avg')
        if rate<=0:raise ValueError('Nonpositive actual throughput')
        observed=count/rate
        if abs(observed-duration)>bound_ns/1e9:
            raise ValueError('Raw counts/rate exceed the exported clock resolution: '+name)
        implied[name]=observed
    evidence._close(evidence._value(summary,'output_token_throughput','avg') /
        evidence._value(summary,'request_throughput','avg'), output_tokens/requests,
        'Reported rates do not share the same actual token/request counts and window')
    return {'ISO_observation_seconds':duration,'export_conversion_bound_ns':bound_ns,
            'rate_implied_observation_seconds':implied,
            'difference_from_ISO_ns':{k:(v-duration)*1e9 for k,v in implied.items()}}


def audit_requests(artifacts: Path, source: Path, measured: int, warmup: int) -> dict:
    frozen = json.loads(source.read_text())
    payloads = {row["session_id"]: row["payloads"][0] for row in frozen["data"]}
    phases = {"profiling": [], "warmup": []}
    metric_by_id = {}
    session_phases = {}
    for row in _jsonl_rows(artifacts / 'profile_export.jsonl'):
        meta, metrics = row["metadata"], row["metrics"]
        phase, identity = meta["benchmark_phase"], meta["x_request_id"]
        session = meta["conversation_id"]
        if phase not in phases or identity in metric_by_id or session in session_phases:
            raise ValueError("Duplicated/missing actual request/session/phase")
        if (row.get("error") is not None or meta.get("was_cancelled") is not False
                or meta.get("context_overflow_skip") is not False or meta.get("turn_index") != 0 or session not in payloads):
            raise ValueError("Failed/cancelled/skipped/unknown production request")
        output = evidence._value(metrics, "output_token_count")
        prompt = evidence._value(metrics, "input_sequence_length")
        if output < 1 or not output.is_integer() or prompt < 1 or not prompt.is_integer():
            raise ValueError("Production request has invalid actual usage")
        evidence._close(output, evidence._value(metrics, "usage_completion_tokens"), "Completion usage mismatch")
        evidence._close(prompt, evidence._value(metrics, "usage_prompt_tokens"), "Prompt usage mismatch")
        evidence._close(prompt + output, evidence._value(metrics, "usage_total_tokens"), "Total usage mismatch")
        if output > payloads[session]["max_tokens"]:
            raise ValueError("Server exceeded natural generation cap")
        if output > 1:
            evidence._close(evidence._value(metrics, "decode_duration") / (output - 1),
                            evidence._value(metrics, "inter_token_latency"), "ITL/actual decode mismatch")
        # A legitimate one-token answer has no inter-token interval; do not reject it.
        if evidence._value(metrics, "request_latency") <= 0 or evidence._value(metrics, "time_to_first_token") <= 0:
            raise ValueError("Invalid actual request/first-token latency")
        metric_by_id[identity] = row
        session_phases[session] = phase
        phases[phase].append(row)
    if len(phases["profiling"]) != measured or len(phases["warmup"]) != warmup:
        raise ValueError("Actual AIPerf profiling/warmup counts differ")
    seen = set()
    stops = collections.Counter()
    for raw in _jsonl_rows(artifacts / 'profile_export_raw.jsonl'):
        meta = raw["metadata"]
        identity = meta["x_request_id"]
        session = meta["conversation_id"]
        if identity in seen or identity not in metric_by_id or raw.get("status") != 200 or raw.get("error") is not None:
            raise ValueError("Raw transport failed or differs from metric records")
        expected_meta = metric_by_id[identity]["metadata"]
        if any(meta.get(key) != expected_meta.get(key) for key in ("conversation_id", "benchmark_phase", "turn_index")):
            raise ValueError("Raw transport session/phase differs from its metric record")
        if raw.get("payload") != payloads.get(session) or raw["payload"].get("ignore_eos", False) is not False:
            raise ValueError("Actual HTTP payload differs from the genuine natural-stop input")
        done, finish, usage = False, None, None
        for response in raw["responses"]:
            for packet in response.get("packets", []):
                if packet.get("name") != "data":
                    continue
                value = packet.get("value", "")
                if value == "[DONE]":
                    done = True
                    continue
                item = json.loads(value)
                if item.get("error"):
                    raise ValueError("Raw server SSE error")
                if item.get("usage"):
                    usage = item["usage"]
                for choice in item.get("choices", []):
                    finish = choice.get("finish_reason") or finish
        if not done or finish not in ("stop", "length") or usage is None:
            raise ValueError("Actual streaming request did not terminate cleanly with real usage")
        evidence._close(usage["completion_tokens"], evidence._value(metric_by_id[identity]["metrics"], "output_token_count"),
                        "Raw SSE and metric usage differ")
        evidence._close(usage["prompt_tokens"], evidence._value(metric_by_id[identity]["metrics"], "input_sequence_length"),
                        "Raw SSE prompt usage differs")
        evidence._close(usage["prompt_tokens"] + usage["completion_tokens"],
                        evidence._value(metric_by_id[identity]["metrics"], "usage_total_tokens"), "Raw SSE total usage differs")
        if meta["benchmark_phase"] == "profiling":
            stops[finish] += 1
        seen.add(identity)
    if seen != set(metric_by_id):
        raise ValueError("Raw wire export is incomplete")
    summary = json.loads((artifacts / "profile_export_aiperf.json").read_text())
    if summary.get("is_complete") is not True or summary.get("was_cancelled") is not False or summary.get("error_summary"):
        raise ValueError("AIPerf did not actually complete")
    if summary.get("aiperf_version") != "0.13.0":
        raise ValueError("AIPerf version changed")
    rows = phases["profiling"]
    duration = (datetime.datetime.fromisoformat(summary["end_time"]) - datetime.datetime.fromisoformat(summary["start_time"])).total_seconds()
    outputs = [evidence._value(row["metrics"], "output_token_count") for row in rows]
    inputs = [evidence._value(row["metrics"], "input_sequence_length") for row in rows]
    _verify_observation_rates(summary,int(sum(outputs)),len(rows))
    result = {"profiling_records": measured, "warmup_records": warmup, "actual_output_tokens": int(sum(outputs)),
              "output_tps": sum(outputs) / duration, "request_rps": len(rows) / duration, "observation_seconds": duration,
              "isl_mean": statistics.mean(inputs), "osl_mean": statistics.mean(outputs), "osl_min": min(outputs),
              "osl_max": max(outputs), "finish_reason_counts": dict(stops), "one_token_answers": outputs.count(1),
              "http_payload_and_sse_verified": True, "profile_session_ids_sha256": digest(sorted(row["metadata"]["conversation_id"] for row in rows)),
              "profile_prompt_counts_sha256": digest(sorted((row["metadata"]["conversation_id"], evidence._value(row["metrics"], "input_sequence_length")) for row in rows))}
    for name, label in (("time_to_first_token", "ttft"), ("inter_token_latency", "itl"), ("request_latency", "e2e")):
        values = [evidence._value(row["metrics"], name) for row in rows
                  if name != "inter_token_latency" or evidence._value(row["metrics"], "output_token_count") > 1]
        for percentile, quantile in (("p50", .5), ("p95", .95), ("p99", .99)):
            result[label + "_" + percentile + "_ms"] = evidence._percentile(values, quantile) if values else None
    result.update(evidence.telemetry_output(artifacts.parent / "gpu_telemetry.csv", rows))
    return result


def candidate_quality_command(ticket: dict, output: Path, arm: str) -> list[str]:
    request = ticket["request"]
    workload = request["workload"]
    source = request["python_source_repo"]
    data = str(Path(request["dataset_path"]).parent)
    bf16q = workload['kind']=='native_quality_bf16q_production_batch'
    full_cohort = bf16q or workload["kind"] in ("native_quality_xqa_bf16_reference", "native_quality_xqa_production_batch")
    bf16_reference = full_cohort and arm == "bf16"
    xqa = workload["kind"] in ("native_quality_xqa", "native_quality_xqa_production_batch") and arm != "bf16"
    profile = (["--native-planner", "fa2", "--native-decode", "flashinfer", "--fp8-bf16q-decode"] if bf16q else
               ["--native-planner", "fa3", "--native-decode", "xqa"] if bf16_reference else
               ["--native-planner", "fa2", "--native-decode", "xqa"] if xqa else
               ["--native-planner", "fa3", "--fp8-q-scale", "1.0"])
    return [sys.executable, "-B", source + "/tools/run_qwen25_joint_kv_quality.py", "--model", "/models/" + MODEL,
            "--server-binary", request["server_binary"], "--runtime-source-repo", source,
            "--runtime-source-commit", request["python_source_commit"], "--runner-source-commit", request["python_source_commit"],
            "--binary-source-commit", request["server_source_commit"], "--attention-backend", "native-flashinfer",
            *profile, "--graph-mode", "cudagraphs",
            "--fp8-k-scale", str(request['fp8_scales']['key']), "--fp8-v-scale", str(request['fp8_scales']['value']), "--cohort-scope",
            "full" if full_cohort else "fragment" if workload["cohort"] == "completion_fragment" else "pilot",
            "--arms", "auto" if arm == "bf16" else "fp8_e4m3", "--gpu", "0", "--port", "19001",
            "--concurrency", str(request['concurrency']), "--max-seqs-per-batch", "128", "--max-tokens-per-batch", "10240",
            "--max-concurrent-requests", "200", "--max-memory-utilization", "0.8", "--max-cache-size", "0",
            "--gsm8k", data + "/gsm8k.jsonl", "--gsm8k-shots", workload["gsm8k_shots"],
            "--longbench-data", data, "--longbench-repo", workload["longbench_repo"],
            "--approved-data-manifest", workload["approved_data_manifest"], "--output", str(output / "quality")]


def run_candidate_quality(ticket: dict, root: Path, output: Path, arm: str, env: dict) -> dict:
    """Import/exercise the official scorer with CUDA hidden before any model startup."""
    request = ticket["request"]
    scale_coverage = request['workload'].get('scale_calibration') is not None
    if scale_coverage:
        from .scale_coverage import validate_prerequisites
        validate_prerequisites(request, output)
    bf16q = request['workload']['kind']=='native_quality_bf16q_production_batch'
    if bf16q:
        prior=request['workload']['operator_validation'];path=Path(prior['audit_path'])
        audit=json.loads(path.read_text())
        if (file_sha(path)!=prior['audit_sha256']
                or audit.get('classification')!='confirmed_operator_evidence_integrity_and_BF16Q_Graph128_contract_only'
                or any(audit.get(k)!=prior[k] for k in ('run_id','source_commit','terminal_result_sha256'))
                or audit.get('actual_numerical_Graph_reported_and_sealed') is not True
                or audit.get('model_loaded') is not False
                or audit.get('graph128',{}).get('production_max_batch')!=128
                or audit.get('backend_source_sha256')!=prior['source_sha256']
                or audit.get('binary_sha256')!=request['binary_sha256'] or audit.get('image_id')!=request['image_id']
                or audit.get('operators')!=request['operators']
                or any(request['backend_source_sha256'].get(k)!=v for k,v in prior['source_sha256'].items())):
            raise ValueError('BF16-Q same-source audited numerical/Graph prerequisite differs before model load')
        write_new_json(output/'operator_prerequisite_reused.json',{'prior':prior,'no_duplicate_GPU_operator_run':True})
    env["PYTHONPATH"] = request["workload"]["scorer_dependencies"] + ":" + env["PYTHONPATH"]
    cpu_env = dict(env, CUDA_VISIBLE_DEVICES="")
    preflight = ("import json,sys; from pathlib import Path; "
                 "from tools import eval_kv_cache_quality as e; "
                 "m=e._longbench_assets(Path(sys.argv[1]))[2]; "
                 "r={'qa_f1':m.qa_f1_score('blue','blue'),'code_sim':m.code_sim_score('value = 1','value = 1')}; "
                 "assert r=={'qa_f1':1.0,'code_sim':1.0}; print(json.dumps(r))")
    with (output / "scorer_cpu_preflight.log").open("x") as log:
        subprocess.run([sys.executable, "-B", "-c", preflight, request["workload"]["longbench_repo"]],
                       env=cpu_env, cwd=request["python_source_repo"], stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True)
    full_cohort = bf16q or request["workload"]["kind"] in ("native_quality_xqa_bf16_reference", "native_quality_xqa_production_batch")
    bf16_reference = full_cohort and arm == "bf16"
    xqa_candidate = request["workload"]["kind"] in ("native_quality_xqa", "native_quality_xqa_bf16_reference", "native_quality_xqa_production_batch")
    if xqa_candidate:
        probe_env = dict(env, XLLM_FLASHINFER_NATIVE_PLANNER="fa3" if arm == "bf16" else "fa2")
        if arm != "bf16" or bf16_reference:
            probe_env["XLLM_FLASHINFER_NATIVE_DECODE"] = "xqa"
        probe = [sys.executable, "-B", request["python_source_repo"] + "/tools/check_native_fa3_operator.py",
                 "--arm", arm, "--out", str(output / "operator_acceptance.json")]
        if scale_coverage:
            probe += ['--key-scale',str(request['fp8_scales']['key']),
                      '--value-scale',str(request['fp8_scales']['value'])]
        with (output / "operator_acceptance.log").open("x") as log:
            subprocess.run(probe, env=probe_env, cwd=request["python_source_repo"], stdout=log,
                           stderr=subprocess.STDOUT, timeout=900, check=True)
        accepted = json.loads((output / "operator_acceptance.json").read_text())
        if scale_coverage and any(accepted.get('backend',{}).get(k+'_scale') != v
                                  for k,v in (('k',request['fp8_scales']['key']),('v',request['fp8_scales']['value']))):
            raise ValueError('Actual operator gate did not exercise the declared calibrated scales')
        required = {"real_capture_and_replay": True, "partial_page_and_replan": True,
                    "noncontiguous_q_checked": True, "model_loaded": False}
        if arm != "bf16" or bf16_reference:
            required.update(padded_graph_bucket_checked=True, fresh_dirty_cache_initialized_to_zero=True,
                            unused_padding_reference="maximum_finite_FP8_or_large_finite_BF16")
        if any(accepted.get(key) != value for key, value in required.items()):
            raise ValueError("Operator exit0 without the required actual reference/Graph/initialization evidence")
    command = candidate_quality_command(ticket, output, arm)
    write_new_json(output / "quality_command.json", command)
    with (output / "quality.log").open("x") as log:
        process = subprocess.Popen(command, env=env, cwd=request["python_source_repo"], stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            monitor(process, None, output / "gpu_telemetry.csv")
            if process.wait() != 0:
                raise ValueError("FA3 candidate quality failed; retain original logs and journals")
        finally:
            stop_child(process)
    directory = output / "quality"
    complete = json.loads((directory / "complete.json").read_text())
    if (complete["summary_sha256"] != file_sha(directory / "summary.json")
            or complete["csv_sha256"] != file_sha(directory / "summary.csv")):
        raise ValueError("FA3 quality completion/summary hashes differ")
    manifest = json.loads((directory / "experiment_manifest.json").read_text())
    expected_scope = ("one_unique_dataset_round" if full_cohort else
                      "evaluation_fragment_no_full_quality_claim" if request["workload"]["cohort"] == "completion_fragment"
                      else "protocol_pilot_no_full_quality_claim")
    if (sum(row["selected_rows"] for row in manifest["data_audit"]["datasets"].values()) != request["workload"]["samples_per_arm"]
            or manifest["plan"]["scope"] != expected_scope
            or manifest["source_commit"] != request["python_source_commit"]):
        raise ValueError("FA3 quality cohort or runtime identity differs")
    expected_arm = "auto" if arm == "bf16" else arm
    attempts = list((directory / "arms" / expected_arm).glob("attempt-*"))
    if len(attempts) != 1:
        raise ValueError("Require one immutable quality server attempt per supervised arm")
    backend = verify_backend((attempts[0] / "server.log").read_text(), arm, request["operators"][arm], request['fp8_scales'])
    return {"protocol": "production_bf16q_quality" if bf16q else "production_xqa_quality" if xqa_candidate else "production_fa3_quality", "samples_per_arm": request["workload"]["samples_per_arm"],
            "scope": expected_scope, "paired_quality_claim": False,
            "model_quality_validated": False, "actual_graph_and_resident_kv_verified": True,
            "summary_sha256": complete["summary_sha256"]}


def aiperf_service_command(request: dict, output: Path) -> list[str]:
    kind=request['workload']['kind']
    trace=kind in ('anonymous_trace_xqa','anonymous_trace_xqa_bf16_reference','anonymous_trace_component_diagnostic')
    perf=['/workspace/.venv-aiperf/bin/aiperf','profile','--model',MODEL,'--tokenizer','/models/'+MODEL,
        '--url','http://127.0.0.1:19001','--endpoint-type','chat','--streaming',
        '--use-legacy-max-tokens','--use-server-token-count','--random-seed','20261026','--export-level','raw',
        '--output-artifact-dir',str(output/'artifacts'),'--request-count',str(request['workload']['requests']),
        '--concurrency',str(request['concurrency']),'--dataset-sampling-strategy','sequential',
        '--request-timeout-seconds','1200','--input-file',request['dataset_path'],
        '--custom-dataset-type','mooncake_trace' if trace else 'inputs_json']
    if kind=='native_tools_xqa_bf16_reference' and request['workload'].get('response_mode')=='buffered':
        perf.remove('--streaming')
    if trace:
        perf+=['--fixed-schedule','--fixed-schedule-auto-offset','--workers-max','1','--no-gpu-telemetry','--ui','none']
    elif request['workload']['warmup']>0:
        perf+=['--warmup-request-count',str(request['workload']['warmup'])]
    return perf


def run_service(ticket: dict, root: Path, output: Path, arm: str, env: dict) -> dict:
    request = ticket["request"]
    range_diagnostic = request['workload']['kind'] == 'heldout_kv_range_probe'
    long_qa_service = request['workload']['kind'] == 'long_document_QA_natural_xqa_bf16_reference'
    from .range_probe import LONG_QA_SCOPE, validate_longcontext_inputs
    long_qa_diagnostic = range_diagnostic and request['workload'].get('content_scope') == LONG_QA_SCOPE
    range_coordinator, range_result = None, None
    native_tools = request['workload']['kind'] == 'native_tools_xqa_bf16_reference'
    buffered_tools = native_tools and request['workload'].get('response_mode', 'streaming') == 'buffered'
    diagnostic = request["workload"]["kind"] == "anonymous_trace_component_diagnostic"
    trace = request["workload"]["kind"] in ("anonymous_trace_xqa", "anonymous_trace_xqa_bf16_reference", "anonymous_trace_component_diagnostic")
    operator = request["operators"][arm]
    xqa = operator.get("decode") == "xqa"
    env.update(XLLM_FLASHINFER_NATIVE_PLANNER=operator["planner"])
    if xqa:
        env.update(XLLM_FLASHINFER_NATIVE_DECODE="xqa")
    elif arm != "bf16":
        env.update(XLLM_FLASHINFER_FP8_QUERY="1", XLLM_FLASHINFER_FP8_Q_SCALE="1.0")
    source_repo = request["python_source_repo"]
    source = Path(request["dataset_path"])
    if trace:
        from .trace_serving import validate_inputs
        trace_rows, trace_manifest = validate_inputs(source, Path(request["workload"]["trace_manifest"]))
        if (len(trace_rows) != request["workload"]["requests"]
                or trace_manifest["arrival_scaling"] != request["workload"]["arrival_scaling"]):
            raise ValueError("Actual trace count/clock differs from the approved request")
        first_payload = trace_rows[0]["payload"]
    else:
        data = json.loads(source.read_text())
        required = request["workload"]["requests"] + request["workload"]["warmup"]
        if len(data["data"]) < required or len({row["session_id"] for row in data["data"]}) != len(data["data"]):
            raise ValueError("Insufficient unique genuine production input")
        if long_qa_diagnostic or long_qa_service:
            validate_longcontext_inputs(data, json.loads(Path(request['workload']['input_manifest']).read_text()))
        elif any(payload.get("ignore_eos", False) is not False or payload.get("max_tokens") != 1024
               for row in data["data"] for payload in row["payloads"]):
            raise ValueError("Production requests must retain natural stopping and declared cap")
        first_payload = data["data"][0]["payloads"][0]
        if native_tools:
            from .native_tools import validate_native_inputs
            validate_native_inputs(source, Path(request['workload']['input_manifest']))
            if json.loads(Path(request['workload']['input_manifest']).read_text()).get('response_mode', 'streaming') != request['workload'].get('response_mode', 'streaming'):
                raise ValueError('Declared native tool response mode differs from frozen input')
    diagnostics, component_result = None, None
    if range_diagnostic:
        if arm != 'bf16':
            raise ValueError('The heldout distribution diagnostic only observes the BF16 reference')
        manifest=json.loads(Path(request['workload']['input_manifest']).read_text())
        if (manifest.get('canonical_sha256') != request['dataset_sha256']
                or manifest.get('requests') != 128
                or not long_qa_diagnostic and manifest.get('quality_labels_loaded') is not False
                or manifest.get('messages_and_natural_stop_controls_unchanged') is not True):
            raise ValueError('Heldout original natural-stop calibration input differs')
        from .range_probe import RangeProbeCoordinator
        range_coordinator=RangeProbeCoordinator(output,ticket)
        env['XLLM_KV_RANGE_PROBE_CONFIG']=str(range_coordinator.root/'config.json')
    if diagnostic:
        from .component_profile import ComponentProfileCoordinator
        diagnostics = ComponentProfileCoordinator(output, ticket, arm, trace_manifest)
        env["XLLM_COMPONENT_DIAGNOSTIC_CONFIG"] = str(diagnostics.root / "config.json")
    write_new_json(output / "production_environment.json", {key: value for key, value in env.items()
                   if key.startswith(("XLLM_", "FLASHINFER_")) or key in ("LD_LIBRARY_PATH", "PYTHONPATH", "MAX_JOBS")})
    if range_diagnostic or long_qa_service:
        from .aiperf_cli_admission import validate_service_cli
        validate_service_cli(aiperf_service_command(request,output),root,output,env,
                             request['workload']['aiperf_cli_config_sha256'])
    # Exercise CLI and the real native operators before loading the 14B checkpoint.
    help_result = subprocess.run(["/workspace/.venv-aiperf/bin/aiperf", "profile", "--help"], env=env,
                                 text=True, capture_output=True, timeout=60, check=True)
    for flag in ("--export-level", "--custom-dataset-type", "--use-server-token-count"):
        if flag not in help_result.stdout:
            raise ValueError("Actual production AIPerf CLI is incompatible")
    probe = [sys.executable, "-B", source_repo + "/tools/check_native_fa3_operator.py", "--arm", arm,
             "--out", str(output / "operator_acceptance.json")]
    with (output / "operator_acceptance.log").open("x") as log:
        probe_env = env.copy()
        probe_env.pop("XLLM_COMPONENT_DIAGNOSTIC_CONFIG", None)
        probe_env.pop('XLLM_KV_RANGE_PROBE_CONFIG',None)
        subprocess.run(probe, env=probe_env, cwd=source_repo, stdout=log, stderr=subprocess.STDOUT, timeout=900, check=True)
    accepted = json.loads((output / "operator_acceptance.json").read_text())
    if (accepted.get("real_capture_and_replay") is not True or accepted.get("partial_page_and_replan") is not True
            or xqa and (accepted.get("fresh_dirty_cache_initialized_to_zero") is not True
                        or accepted.get("padded_graph_bucket_checked") is not True)):
        raise ValueError("Actual operator evidence is incomplete; model must not start")
    if diagnostic:
        probe_root = output / 'diagnostic_probe'
        probe_root.mkdir(mode=0o700)
        with (output / 'component_profiler_acceptance.log').open('x') as log:
            subprocess.run([sys.executable, '-B', source_repo + '/tools/check_component_profiler.py',
                            '--out', str(probe_root / 'probe.json')], env=probe_env, cwd=source_repo,
                           stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True)
        if json.loads((probe_root / 'probe.json').read_text()).get('real_cuda_graph_kernel_activities') is not True:
            raise ValueError('Actual process-local profiler/Graph acceptance failed before model')
    if range_diagnostic:
        probe_root=output/'range_operator';probe_root.mkdir(mode=0o700)
        with (probe_root/'probe.log').open('x') as log:
            subprocess.run([sys.executable,'-B',source_repo+'/tools/check_kv_range_probe.py',
                '--out',str(probe_root/'acceptance.json'),'--request-sha256',ticket['request_sha256']],
                env=probe_env,cwd=source_repo,stdout=log,stderr=subprocess.STDOUT,timeout=300,check=True)
        accepted_range=json.loads((probe_root/'acceptance.json').read_text())
        if any(accepted_range.get(k) is not True for k in (
            'exact_integer_counts_match_Torch_QDQ','FP32_moments_match_Torch_QDQ',
            'nonpositive_padding_NaN_Inf_excluded','valid_nonfinite_values_counted',
            'actual_capture_and_changed_input_replay','reset_excludes_warmup_and_preserves_Graph_buffers',
            'actual_quiescent_reset_export_thread_handshake','layers_and_phases_separate')):
            raise ValueError('Actual range reduction/Graph/control gate failed; model must not start')
    command = [request["server_binary"], "--model=/models/" + MODEL, "--model_id=" + MODEL, "--model_impl=python",
               "--python_model_path=" + source_repo, "--host=127.0.0.1", "--port=19001", "--max_cache_size=0",
               "--max_memory_utilization=0.8", "--enable_prefix_cache=true", "--enable_chunked_prefill=true",
               "--max_tokens_per_batch=10240", "--max_seqs_per_batch=128", "--max_concurrent_requests=200",
               "--kv_cache_dtype=" + ("auto" if arm == "bf16" else "fp8_e4m3"), "--enable_graph=true",
               "--enable_prefill_piecewise_graph=false", "--python_graph_backend=cudagraphs"]
    if native_tools:
        command += ['--tool_call_parser=qwen25']
    write_new_json(output / "server_command.json", command)
    server, client, observations = None, None, None
    with (output / "server.log").open("x") as log:
        try:
            server = subprocess.Popen(command, cwd=source_repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            until = time.monotonic() + 900
            while time.monotonic() < until:
                if server.poll() is not None:
                    raise ValueError("Production xLLM exited before readiness")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:19001/v1/models", timeout=3) as response:
                        if any(row.get("id") == MODEL for row in json.load(response).get("data", [])):
                            break
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Production model startup/Graph capture did not finish")
            # Use a real source request as the admission probe; natural stop remains enabled.
            if native_tools:
                write_new_json(output / 'bootstrap.json', bootstrap())
            payload = dict(first_payload)
            if not native_tools:
                payload["max_tokens"] = 1 if trace else 32
            write_new_json(output / "admission_probe.json", {"payload": payload, "included_in_measurement": False,
                "cache_effect": "First source prompt prefix remains cached before the continuous warm-cache stream"})
            req = urllib.request.Request("http://127.0.0.1:19001/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as response:
                stream = response.read()
            if not buffered_tools and b"[DONE]" not in stream:
                raise ValueError("Production admission request did not complete SSE")
            if native_tools:
                from .native_tools import parse_sse_packets, parse_buffered_chat_response
                write_new_bytes(output / 'native_tool_admission_raw.txt', stream)
                packets = [{'name': 'data', 'value': line[5:].strip().decode()}
                           for line in stream.splitlines() if line.startswith(b'data:')]
                parsed = parse_buffered_chat_response(json.loads(stream)) if buffered_tools else parse_sse_packets(packets)
                write_new_json(output / 'native_tool_admission.json', parsed)
                if not parsed['tool_calls']:
                    raise ValueError('First independent warm request did not demonstrate native tool-call API; retain evidence')
            backend = verify_backend((output / "server.log").read_text(), arm, operator)
            allocations = evidence._markers((output / "server.log").read_text(), "KV_ALLOCATION ")
            if len(allocations) != 1:
                raise ValueError("Resident KV evidence missing/duplicated")
            allocation = evidence.verify_allocation(allocations[0], arm)
            write_new_json(output / "native_backend.json", backend)
            write_new_json(output / "allocation.json", allocations[0])
            if range_coordinator is not None:
                range_coordinator.reset_after_bootstrap()
            perf = aiperf_service_command(request,output)
            if trace or long_qa_service:
                from .service_metrics import ServiceMetricCollector
                observations = ServiceMetricCollector(output / "service_metrics.jsonl", request["run_id"],
                                                       ticket["request_sha256"], allocation["pages_per_layer"],
                                                       on_sample=diagnostics.on_sample if diagnostics is not None else None)
                observations.start()
            aperf_env = env.copy()
            aperf_env.update(PYTHONPATH=str(root / "runtime_support/tokenizer_compat"), CJYU_AIPERF_LOCAL_TOKENIZER_SHIM="1",
                             AIPERF_DATASET_MMAP_CACHE_ENABLED="true", AIPERF_DATASET_MMAP_CACHE_DIR="/run_cache/aiperf_mmap")
            aperf_env.pop("HF_HUB_OFFLINE", None); aperf_env.pop("TRANSFORMERS_OFFLINE", None)
            write_new_json(output / "aiperf_command.json", perf)
            with (output / "aiperf.log").open("x") as perf_log:
                client = subprocess.Popen(perf, cwd=root, env=aperf_env, stdout=perf_log, stderr=subprocess.STDOUT, start_new_session=True)
                if diagnostics is not None:
                    diagnostics.client_started()
                monitor(client, server, output / "gpu_telemetry.csv")
                client_code = client.wait()
                if client_code != 0 and (not (trace or native_tools) or client_code != 1):
                    raise ValueError("Actual production AIPerf failed")
            if observations is not None:
                observations.stop(); observations = None
            # Persist the actual device statistics before parsing any result
            # export. A postprocessing error must not discard the observations.
            if range_coordinator is not None:
                range_result=range_coordinator.export_after_requests()
            if trace:
                from .trace_serving import audit_trace
                audit = audit_trace(output / "artifacts", source, Path(request["workload"]["trace_manifest"]))
                audit["aiperf_exit_code"] = client_code
                if diagnostics is not None:
                    component_result = diagnostics.verify(audit)
            elif native_tools:
                from .native_tools import audit_native_tools
                audit = audit_native_tools(output / 'artifacts', source, Path(request['workload']['input_manifest']))
                audit['aiperf_exit_code'] = client_code
            else:
                audit = audit_requests(output / "artifacts", source, request["workload"]["requests"], request["workload"]["warmup"])
            verify_backend((output / "server.log").read_text(), arm, operator)
            write_new_json(output / "request_audit.json", audit)
        finally:
            try:
                if observations is not None: observations.stop()
            finally:
                try:
                    if client is not None: stop_child(client)
                finally:
                    if server is not None: stop_child(server)
    return {"protocol": "long_document_QA_natural_context_service_v1" if long_qa_service else "long_document_QA_natural_context_diagnostic_v1" if long_qa_diagnostic else "native_tool_name_arguments_natural_API_v1" if native_tools else "anonymous_arrival_length_prefix_replay_v1" if trace else "sharegpt_context_natural_production_v1", "scope": "production_KV_distribution_diagnostic_not_headline_performance" if range_diagnostic else "production_component_observer_diagnostic_not_headline_performance" if diagnostic else "production_serving_candidate_quality_pending",
            "model_quality_validated": False, "metrics": audit, "allocation": allocation,
            **({"component_diagnostic": component_result} if diagnostic else {}),
            **({'KV_distribution_diagnostic':range_result} if range_diagnostic else {}),
            "actual_declared_graph_and_resident_kv_verified": True, "official_aiperf_complete": True}
