"""Pure execution contracts and Docker argv construction; no remote effects."""

from __future__ import annotations

import datetime
from pathlib import PurePosixPath
import re

from .approval import ApprovalError, validate_approval


CONTRACT = "kv-controller-v1"
MODEL = "Qwen2.5-14B-Instruct"
SOURCE_FILES = ("xllm/python/model_executor/executor.py", "xllm/python/attention/flashinfer.py",
                "xllm/python/attention/flashinfer_fp8.py", "xllm/python/attention/quantized.py",
                "xllm/python/attention/quantized_triton.py", "xllm/python/attention/native_kv.py",
                "xllm/python/model_executor/runners/decode_cuda_graph.py", "xllm/core/runtime/worker_impl.cpp")
PRODUCTION_SOURCES = SOURCE_FILES + ("xllm/python/attention/native_profile.py", "xllm/python/attention/query_quantization.py",
                                    "tools/check_native_fa3_operator.py")
QUALITY_SOURCES = ("tools/run_qwen25_joint_kv_quality.py", "tools/eval_kv_cache_quality.py",
                   "tools/native_kv_quality_contract.py", "tools/qwen25_experiment_model.py", "scripts/logger.py")
XQA_SOURCES = PRODUCTION_SOURCES + ("xllm/python/attention/xqa_decode.py",)
COMPONENT_SOURCES = XQA_SOURCES + ("xllm/python/model_executor/component_diagnostics.py",
                                   "tools/check_component_profiler.py")
NATIVE_TOOL_SOURCES = XQA_SOURCES + ("xllm/api_service/chat_service_impl.cpp",
    "xllm/function_call/qwen25_detector.cpp", "xllm/function_call/function_call_parser.cpp")
PREFILL_WORKSPACE_SOURCES = XQA_SOURCES + ("xllm/python/attention/fp8_prefill_workspace.py",
                                         "tools/check_fp8_prefill_workspace.py")
KV_RANGE_SOURCES = XQA_SOURCES + ("xllm/python/attention/kv_range_probe.py", "tools/check_kv_range_probe.py")
BF16Q_DECODE_SOURCES = XQA_SOURCES + ("tools/check_fp8_bf16q_decode.py",)
FA3_OPERATORS = {"bf16": {"planner": "fa3", "q": "bf16"},
                 "fp8_e4m3": {"planner": "fa3", "q": "fp8_e4m3", "q_scale": 1.0}}
XQA_OPERATORS = {"bf16": {"planner": "fa3", "q": "bf16"},
                 "fp8_e4m3": {"planner": "fa2", "q": "bf16", "decode": "xqa",
                               "decode_internal_q": "fp8_e4m3", "max_context": 32768, "enable_pdl": True,
                               "kv_padding_policy": "finite_full_initialization"}}
BF16_XQA_OPERATORS = {
    "bf16": {"planner": "fa3", "q": "bf16", "decode": "xqa", "decode_internal_q": "bf16",
             "max_context": 32768, "enable_pdl": True, "kv_padding_policy": "finite_full_initialization"},
    "fp8_e4m3": dict(XQA_OPERATORS["fp8_e4m3"]),
}
PREFILL_WORKSPACE_OPERATORS = {"fp8_e4m3": {
    **XQA_OPERATORS["fp8_e4m3"], "planner": "fa3", "prefill_kv": "bf16",
    "prefill_workspace_mib": 1536}}
BF16Q_DECODE_OPERATORS = {"fp8_e4m3": {
    "planner":"fa2", "q":"bf16", "decode":"flashinfer", "decode_internal_q":"bf16",
    "profile_scope":"production_bf16q_decode", "kv_padding_policy":"finite_full_initialization"}}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ApprovalError(message)


def _path(value: object) -> str:
    _require(isinstance(value, str) and value.startswith("/") and "\x00" not in value,
             "Expected an absolute path")
    _require(str(PurePosixPath(value)) == value and ".." not in PurePosixPath(value).parts,
             "Path must be canonical and contain no traversal")
    _require("," not in value and not any(ord(char) < 32 for char in value), "Unsafe Docker mount path encoding")
    return value


def validate_execution_authority(approval: dict, now: datetime.datetime) -> None:
    validate_approval(approval, now)
    _require(approval.get("execution_contract") == CONTRACT,
             "User must explicitly approve the concrete kv-controller-v1 contract")
    identity = approval.get("approved_remote_identity", {})
    _require(type(identity.get("uid")) is int and identity["uid"] > 0
             and type(identity.get("gid")) is int and identity["gid"] > 0
             and isinstance(identity.get("hostname"), str) and bool(identity["hostname"]),
             "Approval must bind the discovered non-root UID/GID/hostname")
    _require(re.fullmatch(r"sha256:[0-9a-f]{64}", approval.get("approved_image_id", "")) is not None,
             "Approval must pin the existing Docker image ID")
    _path(approval.get("read_only_model_path"))
    policy = approval.get("shared_gpu_policy", {})
    _require(policy.get("kind") in ("reservation_file", "opportunistic_no_scheduler", "idle_only_standing_user"),
             "Shared reservation policy is unknown")
    if policy["kind"] == "reservation_file":
        _path(policy.get("path"))
    elif policy["kind"] == "opportunistic_no_scheduler":
        _require(policy.get("user_confirms_no_external_reservations") is True,
                 "No-scheduler opportunistic use requires the user's explicit reservation confirmation")
    else:
        _require(approval.get("authorization_origin", {}).get("kind") == "explicit_user_message"
                 and policy.get("no_foreign_gpu_use") is True and policy.get("never_interrupt_others") is True,
                 "Idle-only standing authority must preserve the user's explicit shared-server conditions")
    limits = approval.get("container_limits", {})
    _require(type(limits.get("cpus")) is int and 1 <= limits["cpus"] <= 8,
             "Explicit bounded CPU allocation required")
    _require(type(limits.get("memory_gib")) is int and 32 <= limits["memory_gib"] <= 128,
             "Explicit bounded container RAM allocation required")
    _require(type(limits.get("shm_gib")) is int and 1 <= limits["shm_gib"] <= limits["memory_gib"] // 2,
             "Explicit bounded private shared memory required")


def validate_request(request: dict, approval: dict) -> None:
    if request.get("schema") == "kv-nextgen-run-v1":
        from .nextgen import validate_request as validate_nextgen_request
        validate_nextgen_request(request, approval)
        return
    production = request.get("schema") == "kv-production-run-v1"
    workload = request.get('workload', {})
    recovery = workload.get('recovery')
    prefill_operator = production and workload.get('kind') == 'native_prefill_workspace_operator'
    range_probe = production and workload.get('kind') == 'heldout_kv_range_probe'
    long_qa_service = production and workload.get('kind') == 'long_document_QA_natural_xqa_bf16_reference'
    bf16q_operator = production and workload.get('kind') == 'native_bf16q_decode_operator'
    bf16q_quality = production and workload.get('kind') == 'native_quality_bf16q_production_batch'
    from .scale_coverage import validate_contract as validate_scale_contract
    scale_coverage = validate_scale_contract(request)
    _require(production or request.get("schema") == "kv-native-run-v1", "Unsupported job schema")
    _require(len(request.get("gpu_uuids", [])) == 1, "v1 runs one isolated single-GPU instance per job")
    _require(type(request.get("max_wall_seconds")) is int and 600 <= request["max_wall_seconds"] <= 86400,
             "Wall time must include startup, two stable-idle windows and cleanup")
    missing_bf16 = (production and workload.get('kind') == 'native_quality_xqa'
                   and workload.get('cohort') == 'completion_fragment' and recovery is not None
                   and request.get('arms') == ['bf16'])
    full_bf16_reference = (production and workload.get('kind') == 'native_quality_xqa_bf16_reference'
                          and workload.get('cohort') == 'full_reference' and recovery is None
                          and request.get('arms') == ['bf16'])
    _require(request.get("arms") in (["bf16", "fp8_e4m3"], ["fp8_e4m3", "bf16"]) or missing_bf16 or full_bf16_reference
             or prefill_operator and request.get('arms') == ['fp8_e4m3']
             or bf16q_operator and request.get('arms') == ['fp8_e4m3']
             or bf16q_quality and request.get('arms') == ['fp8_e4m3']
             or scale_coverage and request.get('arms') == ['fp8_e4m3']
             or range_probe and request.get('arms') == ['bf16'],
             "Explicit paired native BF16/FP8 arm order required")
    if recovery is not None:
        _require(missing_bf16 and recovery.get('verified_arm') == 'fp8_e4m3'
                 and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', recovery.get('prior_run_id','')) is not None,
                 'Recovery is only the exact missing BF16 counterpart of a verified FP8 fragment')
        for field in ('prior_terminal_result_sha256', 'recovered_audit_sha256', 'arm_measurement_sha256'):
            _require(re.fullmatch(r'[0-9a-f]{64}', recovery.get(field,'')) is not None,
                     'Missing immutable quality recovery prerequisite: '+field)
    for field in ("source_repo", "server_binary", "dataset_path"):
        path = _path(request.get(field))
        _require(path.startswith(approval["approved_workspace"] + "/"), "Source/binary/data escaped approved BASE")
    _require(request.get("model_path") == approval["read_only_model_path"], "Model path not approved")
    _require(request.get("image_id") == approval["approved_image_id"], "Docker image identity differs")
    for field, size in (("server_source_commit", 40), ("binary_sha256", 64), ("dataset_sha256", 64),
                        ("model_config_sha256", 64)):
        _require(re.fullmatch(r"[0-9a-f]{" + str(size) + "}", request.get(field, "")) is not None,
                 "Missing pinned provenance: " + field)
    _require(request.get("graph_mode") == "cudagraphs" and request.get("prefix_cache") is production,
             "Explicit matched Graph/prefix configuration required")
    _require(request.get("fp8_scales") == ({"key": 0.125, "value": 0.125} if scale_coverage
                                          else {"key": 0.125, "value": 0.0625}),
             "v1 must preserve audited fixed scales; new scales need a distinct contract")
    hashes = request.get("backend_source_sha256", {})
    workload = request.get("workload", {})
    diagnostic = production and workload.get("kind") == "anonymous_trace_component_diagnostic"
    native_tools = production and workload.get('kind') == 'native_tools_xqa_bf16_reference'
    trace = production and workload.get("kind") in ("anonymous_trace_xqa", "anonymous_trace_xqa_bf16_reference", "anonymous_trace_component_diagnostic")
    production_batch = production and workload.get("kind") == "native_quality_xqa_production_batch"
    bf16_xqa_reference = native_tools or long_qa_service or production and workload.get("kind") in ("sharegpt_natural_xqa_bf16_reference", "anonymous_trace_xqa_bf16_reference", "native_quality_xqa_bf16_reference", "native_quality_xqa_production_batch", "anonymous_trace_component_diagnostic")
    xqa = native_tools or long_qa_service or production and workload.get("kind") in ("native_quality_xqa", "native_quality_xqa_bf16_reference", "native_quality_xqa_production_batch", "anonymous_trace_xqa", "anonymous_trace_xqa_bf16_reference", "sharegpt_natural_xqa", "sharegpt_natural_xqa_bf16_reference", "anonymous_trace_component_diagnostic")
    quality = bf16q_quality or production and workload.get("kind") in ("native_quality_fa3", "native_quality_xqa", "native_quality_xqa_bf16_reference", "native_quality_xqa_production_batch")
    sources = (XQA_SOURCES if xqa else PRODUCTION_SOURCES) + QUALITY_SOURCES if quality else (XQA_SOURCES if xqa else PRODUCTION_SOURCES if production else SOURCE_FILES)
    if diagnostic:
        sources = COMPONENT_SOURCES
        _require(request.get('diagnostic') == {'method': 'owned_process_torch_profiler', 'max_windows': 3,
            'max_forwards_per_window': 16, 'max_seconds_per_window': 6, 'max_trace_mib': 32,
            'window_source_seconds': 90, 'semantic_quality_claim': False, 'headline_performance_claim': False}
            and request['max_wall_seconds'] <= 5400
            and type(request.get('max_output_gib')) in (int, float) and 0 < request['max_output_gib'] <= 3
            and workload.get('family') == 'toolagent' and workload.get('arrival_scaling') == .125,
            'Component diagnostic requires exact bounded ownprocess/Graph/trace controls')
    else:
        _require('diagnostic' not in request, 'Do not inject a profiler into an ordinary production run')
    if native_tools:
        sources = NATIVE_TOOL_SOURCES
    if prefill_operator:
        sources = PREFILL_WORKSPACE_SOURCES
    if range_probe:
        sources = KV_RANGE_SOURCES
    if bf16q_operator:
        sources = BF16Q_DECODE_SOURCES
    if bf16q_quality:
        sources = BF16Q_DECODE_SOURCES + QUALITY_SOURCES
        prior=workload.get('operator_validation',{})
        _require(request['arms']==['fp8_e4m3'] and request['concurrency']==128
                 and request['max_wall_seconds']<=5400 and 0<request.get('max_output_gib',0)<=5
                 and prior.get('run_id')=='prod-bf16q-decode-op-r0-20261027a'
                 and prior.get('source_commit')=='78004505e2db44e88994899e8ea3e25c109978a7'
                 and prior.get('audit_path')==request['remote_output']+'/input/bf16q_operator_audit.json'
                 and all(re.fullmatch(r'[0-9a-f]{64}',prior.get(k,'')) is not None for k in
                         ('terminal_result_sha256','audit_sha256'))
                 and set(prior.get('source_sha256',{}))==set(BF16Q_DECODE_SOURCES)
                 and all(prior['source_sha256'][k]==hashes.get(k) for k in BF16Q_DECODE_SOURCES),
                 'BF16-Q quality requires the independently audited same-inference-source Graph128 prerequisite')
    _require(set(hashes) == set(sources)
             and all(re.fullmatch(r"[0-9a-f]{64}", value or "") for value in hashes.values()),
             "All native backend/Graph/C++ admission sources must be pinned")
    _require(type(request.get("concurrency")) is int and 1 <= request["concurrency"] <= (1024 if trace else 128 if production else 16),
             "Concurrency outside bounded v1 range")
    if production:
        _require(workload.get("kind") in ("sharegpt_natural", "sharegpt_natural_xqa", "sharegpt_natural_xqa_bf16_reference", "long_document_QA_natural_xqa_bf16_reference", "native_quality_fa3", "native_quality_xqa", "native_quality_xqa_bf16_reference", "native_quality_xqa_production_batch", "anonymous_trace_xqa", "anonymous_trace_xqa_bf16_reference", "anonymous_trace_component_diagnostic", "native_tools_xqa_bf16_reference", "native_prefill_workspace_operator", "heldout_kv_range_probe", "native_bf16q_decode_operator", "native_quality_bf16q_production_batch")
                 and workload.get("ignore_eos") is trace,
                 "Natural semantic requests and explicitly anonymous per-request length replay must remain distinct")
        if bf16q_operator:
            _require(request['arms']==['fp8_e4m3'] and request['concurrency']==128
                     and request['max_wall_seconds']<=1800 and 0<request.get('max_output_gib',0)<=3
                     and workload.get('method')=='resident_FP8_BF16Q_production_Graph128_gate_v1'
                     and workload.get('model_loaded') is False and workload.get('AIPerf_ran') is False
                     and workload.get('geometry')=={'batch':128,'heads':40,'kv_heads':8,'head_dim':128,'page_size':128}
                     and workload.get('relative_l2_limit')==.01
                     and set(workload.get('primary_sources_sha256',{}))=={
                         'include/flashinfer/attention/prefill.cuh','include/flashinfer/mma.cuh','flashinfer/decode.py'}
                     and all(re.fullmatch(r'[0-9a-f]{64}',v or '') for v in workload['primary_sources_sha256'].values()),
                     'BF16-Q candidate requires a bounded actual-production Graph128/numerical operator gate')
        elif long_qa_service:
            from .range_probe import LONG_QA_SCOPE
            _require(request['concurrency'] == 128 and workload.get('requests') == 128
                     and workload.get('warmup') == 0 and request['max_wall_seconds'] <= 1800
                     and 0 < request.get('max_output_gib', 0) <= 5
                     and workload.get('content_scope') == LONG_QA_SCOPE
                     and workload.get('arrival_mode') == 'closed_loop_concurrency_finite_cohort'
                     and workload.get('official_generation_caps_preserved') is True
                     and workload.get('aiperf_zero_warmup_policy') == 'omit_explicit_option'
                     and workload.get('activation_diagnostic_enabled') is False
                     and workload.get('input_manifest') == request['remote_output']+'/input/production_manifest.json'
                     and workload.get('aiperf_cli_config_sha256') ==
                         '2be937b185ddfe3a19e8090aa4411f1015a2e00899d29db3055d3f7e19bdd42b'
                     and request['dataset_sha256'] ==
                         '1a36f83244b09b3edd85895586ec324624640ade004eb3ece672e94ee5dcfe5a'
                     and workload.get('source_content_sha256') ==
                         'cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64',
                     'Natural long-document QA service requires the exact128 genuine requests, original caps, no activation observer and bounded paired deployment')
        elif range_probe:
            from .range_probe import CHAT_SCOPE, LONG_QA_SCOPE
            scope = workload.get('content_scope')
            label_separation = (scope == CHAT_SCOPE and workload.get('quality_labels_loaded') is False
                or scope == LONG_QA_SCOPE and workload.get('quality_labels_used_in_selection') is False
                and workload.get('answers_exported_to_probe') is False
                and workload.get('official_generation_caps_preserved') is True
                and workload.get('evaluation_excluded_count') == 400)
            _require(request['arms']==['bf16'] and request['concurrency']==128
                     and request['max_wall_seconds']<=1800 and 0<request.get('max_output_gib',0)<=3
                     and workload.get('requests')==128 and workload.get('warmup')==0
                     and workload.get('method')=='heldout_BF16_KV_distribution_GPU_aggregate_v1'
                     and workload.get('aiperf_zero_warmup_policy')=='omit_explicit_option'
                     and workload.get('jsonl_framing')=='physical_LF_UTF8'
                     and workload.get('export_before_postprocessing') is True
                     and re.fullmatch(r'[0-9a-f]{64}',workload.get('aiperf_cli_config_sha256','')) is not None
                     and label_separation
                     and workload.get('input_manifest')==request['remote_output']+'/input/production_manifest.json',
                     'KV range diagnostic requires the bounded heldout128 single-BF16 natural Graph scope and explicit input separation')
        elif prefill_operator:
            _require(request['arms'] == ['fp8_e4m3'] and request['concurrency'] == 1
                     and request['max_wall_seconds'] <= 1800 and 0 < request.get('max_output_gib', 0) <= 3
                     and workload.get('method') == 'recorded_production_prefill_repair_operator_gate'
                     and workload.get('model_loaded') is False and workload.get('AIPerf_ran') is False
                     and workload.get('geometry') == {'sequences': 1, 'query_tokens': 510, 'kv_tokens': 1022, 'layers': 48},
                     'Prefill operator gate requires bounded single-GPU/no-model/no-AIPerf recorded geometry')
        elif quality:
            cohort = workload.get("cohort")
            _require((bf16q_quality and request['concurrency']==128 and cohort=='full_production_batch'
                      and workload.get('samples_per_arm')==1719) or
                     (production_batch and request["concurrency"] == 128
                      and cohort == "full_production_batch" and workload.get("samples_per_arm") == 1719
                      and (scale_coverage or request['arms'] in (['bf16', 'fp8_e4m3'], ['fp8_e4m3', 'bf16']))) or
                     (not production_batch and request["concurrency"] == 8 and
                     (full_bf16_reference and workload.get("samples_per_arm") == 1719 or
                      workload.get("kind") != "native_quality_xqa_bf16_reference"
                      and cohort in ("pilot", "completion_fragment")
                      and workload.get("samples_per_arm") == (80 if cohort == "pilot" else 1639))),
                     "Quality requires an exact C8 scope or the explicit paired full C128 deployment check")
            if cohort == "completion_fragment":
                _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", workload.get("reuse_run_id", "")) is not None,
                         "Quality completion requires an exact previously completed parent run")
                for field in ("parent_result_sha256", "parent_audit_sha256"):
                    _require(re.fullmatch(r"[0-9a-f]{64}", workload.get(field, "")) is not None,
                             "Quality completion requires pinned parent result/audit: " + field)
            _require(workload.get("scorer_dependencies") == approval["approved_workspace"] + "/experiments/joint-kv-14b/longbench-deps-py312",
                     "Quality must use the previously discovered read-only isolated scorer dependencies")
            for field, suffix in (("longbench_repo", "/input/scorer"), ("gsm8k_shots", "/input/five_train_examples.json"),
                                  ("approved_data_manifest", "/input/approved_data_manifest.json")):
                _require(workload.get(field) == request["remote_output"] + suffix, "Quality input must be frozen in this run: " + field)
        elif native_tools:
            _require(request['concurrency'] == 128 and workload.get('requests') == 258
                     and workload.get('response_mode', 'streaming') in ('streaming', 'buffered')
                     and workload.get('warmup') == 16 and workload.get('tool_call_parser') == 'qwen25'
                     and workload.get('source_commit') == '6ea57973c7a6097fd7c5915698c54c17c5b1b6c8'
                     and workload.get('content_scope') == 'original_BFCL_live_simple_messages_native_tools'
                     and workload.get('arrival_mode') == 'closed_loop_concurrency'
                     and workload.get('input_manifest') == request['remote_output'] + '/input/production_manifest.json'
                     and request['max_wall_seconds'] <= 3600 and 0 < request.get('max_output_gib', 0) <= 3,
                     'Native tools require the exact full frozen natural-API cohort and bounded job')
        elif trace:
            _require(type(workload.get("requests")) is int and 128 <= workload["requests"] <= 10000
                     and workload.get("warmup") == 0 and workload.get("arrival_mode") == "fixed_schedule_continuous_warm_cache"
                     and workload.get("content_scope") == "anonymous_arrival_length_prefix_emulation"
                     and workload.get("family") in ("conversation", "toolagent", "synthetic", "burstgpt")
                     and type(workload.get("arrival_scaling")) in (int, float) and .1 <= workload["arrival_scaling"] <= 4
                     and workload.get("trace_manifest") == request["remote_output"] + "/input/trace_manifest.json",
                     "Trace replay requires bounded counts, immutable clock/manifest and explicit anonymous semantics")
            archive = request.get('dataset_archive', {})
            codec = archive.get('codec', 'gzip')
            _require(codec in ('gzip','xz')
                     and archive.get('relative_path') == 'input/payload_trace.jsonl.' + ('xz' if codec=='xz' else 'gz')
                     and type(archive.get('uncompressed_size_bytes')) is int
                     and 0 < archive['uncompressed_size_bytes'] <= 64*2**20
                     and request['dataset_path'] == request['remote_output']+'/input/payload_trace.jsonl',
                     "Trace requires a bounded byte-identical decompressed payload in this run")
        else:
            _require(type(workload.get("requests")) is int and 128 <= workload["requests"] <= 2048
                     and type(workload.get("warmup")) is int and 8 <= workload["warmup"] <= 64,
                     "Bounded, explicit real request counts required")
        _require(request.get("service") == {"max_seqs_per_batch": 128, "max_tokens_per_batch": 10240,
                                            "max_concurrent_requests": 200, "memory_utilization": .8},
                 "Production v1 uses real scheduler defaults with a supported Graph batch envelope")
        expected_operators = BF16Q_DECODE_OPERATORS if bf16q_operator or bf16q_quality else {'bf16':BF16_XQA_OPERATORS['bf16']} if range_probe else PREFILL_WORKSPACE_OPERATORS if prefill_operator else BF16_XQA_OPERATORS if bf16_xqa_reference else XQA_OPERATORS if xqa else FA3_OPERATORS
        _require(request.get("operators") == expected_operators, "Production operator profile differs")
        _require(request.get("python_source_repo") == request["remote_output"] + "/source_repo"
                 and re.fullmatch(r"[0-9a-f]{40}", request.get("python_source_commit", "")) is not None,
                 "Production tools/runtime must use a new isolated pinned checkout")
        _require(workload.get("source_content_sha256") is not None
                 and re.fullmatch(r"[0-9a-f]{64}", workload["source_content_sha256"]) is not None,
                 "Real source dataset provenance required")
    elif workload.get("kind") == "fixed_decode":
        _require(workload.get("requests") == 128 and workload.get("warmup") == 8
                 and workload.get("output_tokens") == 512 and workload.get("ignore_eos") is True,
                 "Fixed-decode control protocol differs")
        _require(workload.get("isl") in (16384, 24576), "v1 only targets the planned causal comparison")
    else:
        _require(workload.get("kind") == "native_quality", "Unsupported workload")
        _require(workload.get("cohort") == "pilot" and workload.get("samples_per_arm") == 80
                 and request["concurrency"] == 1, "v1 quality is the 80-sample sequential pilot")
        _require(workload.get("ignore_eos") is False, "Quality must use natural stopping")
        for field in ("runner_path", "longbench_repo", "gsm8k_shots", "approved_data_manifest"):
            path = _path(workload.get(field))
            _require(path.startswith(approval["approved_workspace"] + "/"), "Quality assets escaped BASE")
        _require(re.fullmatch(r"[0-9a-f]{40}", workload.get("runner_source_commit", "")) is not None,
                 "Quality tools require their independent source commit")
    _require(isinstance(request.get("research_reason"), str) and len(request["research_reason"]) >= 16,
             "Explain the changed control variable/statistical need; no blind reruns")


def docker_command(ticket: dict, arm: str, device_index: int) -> list[str]:
    request, approval = ticket["request"], ticket["approval"]
    _require(arm in request["arms"], "Arm not in reserved manifest")
    _require(type(device_index) is int and 0 <= device_index < 8, "Invalid live GPU index mapping")
    base = approval["approved_workspace"]
    run = request["remote_output"]
    identity = approval["approved_remote_identity"]
    limits = approval["container_limits"]
    model_id = request["model_id"] if request.get("schema") == "kv-nextgen-run-v1" else MODEL
    name = "kvra-" + ticket["reservation_id"][:20] + "-" + arm
    label = {"kvra.run_id": request["run_id"], "kvra.reservation_id": ticket["reservation_id"],
             "kvra.request_sha256": ticket["request_sha256"], "kvra.arm": arm}
    command = ["docker", "create", "--name", name]
    for key, value in sorted(label.items()):
        command += ["--label", key + "=" + value]
    command += ["--runtime", "nvidia", "--gpus", "device=" + request["gpu_uuids"][0],
                "--device", "/dev/nvidia" + str(device_index), "--device", "/dev/nvidiactl",
                "--device", "/dev/nvidia-uvm", "--device", "/dev/nvidia-uvm-tools",
                "--user", str(identity["uid"]) + ":" + str(identity["gid"]),
                "--network", "none", "--ipc", "private", "--shm-size", str(limits["shm_gib"]) + "g",
                "--cpus", str(limits["cpus"]), "--memory", str(limits["memory_gib"]) + "g",
                "--memory-swap", str(limits["memory_gib"]) + "g", "--pids-limit", "1024",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--security-opt", "seccomp=unconfined", "--read-only",
                "--mount", "type=bind,src=" + base + ",dst=/workspace,readonly",
                "--mount", "type=bind,src=" + base + ",dst=" + base + ",readonly",
                "--mount", "type=bind,src=" + request["model_path"] + ",dst=/models/" + model_id + ",readonly",
                "--mount", "type=bind,src=" + run + "/output/" + arm + ",dst=/results",
                "--mount", "type=bind,src=" + run + "/cache/" + arm + ",dst=/run_cache",
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=4g,mode=1777"]
    runtime = "/workspace/" + str(PurePosixPath(run).relative_to(base)) + "/runtime"
    environment = {"EXPECTED_GPU_UUID": request["gpu_uuids"][0], "HOME": "/tmp", "USER": "kv-research-worker",
                   "LOGNAME": "kv-research-worker", "PYTHONDONTWRITEBYTECODE": "1",
                   "PYTHONPATH": runtime + ":/workspace/.venv/lib/python3.12/site-packages:" + request.get("python_source_repo", request["source_repo"]),
                   "HF_HOME": "/run_cache/huggingface", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "XDG_CACHE_HOME": "/run_cache/xdg", "TRITON_CACHE_DIR": "/run_cache/triton",
                   "TORCHINDUCTOR_CACHE_DIR": "/run_cache/torchinductor",
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "NUMEXPR_NUM_THREADS": "1"}
    for key, value in sorted(environment.items()):
        command += ["-e", key + "=" + value]
    command += ["--entrypoint", "/workspace/.venv/bin/python", request["image_id"], "-B", "-m",
                "kv_controller.worker", "--ticket", run + "/ticket.json", "--arm", arm]
    return command
