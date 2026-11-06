"""Qualification of frozen upstream with one documented common CUDA build fix through the guarded controller.

This contract runs the pinned server and observes forwarded CUDA Graph APIs.
The observer is confined to untimed qualification and excluded from service.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request

from .approval import ApprovalError
from .durable import write_new_json

KIND = 'upstream_official_A_model_qualification'
OFFICIAL_COMMIT = '261274e5f45449e98d7740144c3a958a29f1fd3e'
BASELINE_COMMIT = '721afdabc1de1af8e594b4da3e6fa926def50821'
MODEL = 'Qwen3.8-27B'
CAMPAIGN_GPU = 'GPU-0c628a64-5f31-3501-fa3c-723f5b69254a'
SOURCE_FILES = (
    'xllm/python/attention/backend.py',
    'xllm/python/attention/flashinfer.py',
    'xllm/python/model_executor/executor.py',
    'xllm/python/model_executor/runners/decode_cuda_graph.py',
    'xllm/python/models/qwen3_5.py',
    'xllm/python/layers/cuda/qwen3_5/gated_delta_net.py',
    'xllm/core/runtime/worker_impl.cpp',
    'xllm/core/runtime/mtp_worker_impl.cpp',
    'xllm/core/runtime/py_executor_impl.cpp',
    'xllm/core/kv_cache/layout/kv_cache_estimation.cpp',
    'xllm/core/kv_cache/layout/kv_cache_shape.cpp',
    'xllm/core/kv_cache/storage/kv_cache_utils.cpp',
)
SERVICE = {'max_batch': 32, 'max_tokens_per_batch': 8192,
           'max_memory_utilization': 0.90, 'block_size': 128}


def validate_request(request: dict, approval: dict) -> None:
    from .policy import _path

    def require(value: bool, reason: str) -> None:
        if not value:
            raise ApprovalError(reason)

    require(request.get('schema') == 'kv-nextgen-run-v1', 'Unknown upstream controller schema')
    require(request.get('workload') == {'kind': KIND, 'scope': 'official_model_prefix_graph_qualification'},
            'Unknown official A qualification workload')
    require(request.get('arms') == ['bf16'] and request.get('model_id') == MODEL,
            'Official A requires the cached Qwen3.8 model and BF16 only')
    require(request.get('model_path') == approval['read_only_model_path']
            and Path(request['model_path']).name == MODEL, 'Model escaped the approved read-only path')
    require(request.get('image_id') == approval['approved_image_id'], 'Unpinned runtime image')
    require(request.get('upstream_commit') == OFFICIAL_COMMIT
            and request.get('server_source_commit') == BASELINE_COMMIT
            and request.get('python_source_commit') == BASELINE_COMMIT,
            'Official A must use the exact official base plus the isolated NPU-only build guard in C++ and Python')
    require(len(request.get('gpu_uuids', [])) == 1, 'Qualification uses one admitted GPU')
    require(type(request.get('max_wall_seconds')) is int and 1800 <= request['max_wall_seconds'] <= 3600,
            'Qualification must retain a bounded startup/cleanup window')
    require(request.get('startup_seconds') == 1500, 'Frozen official startup window differs')
    for field in ('source_repo', 'python_source_repo', 'server_binary', 'dataset_path'):
        require(_path(request.get(field)).startswith(approval['approved_workspace'] + '/'),
                'Qualification source/binary/input escaped the owned project')
    require(request['python_source_repo'] == request['remote_output'] + '/source_repo'
            and request['dataset_path'] == request['remote_output'] + '/input/validation.json',
            'Qualification input/Python checkout escaped its new run')
    for field in ('binary_sha256', 'dataset_sha256', 'model_config_sha256'):
        require(re.fullmatch('[0-9a-f]{64}', request.get(field, '')) is not None, 'Missing SHA: ' + field)
    hashes = request.get('backend_source_sha256', {})
    require(set(hashes) == set(SOURCE_FILES)
            and all(re.fullmatch('[0-9a-f]{64}', h) is not None for h in hashes.values()),
            'Missing official executor/attention/cache provenance')
    require(request.get('service') == SERVICE and request.get('graph_mode') == 'cudagraphs'
            and request.get('prefix_cache') is True, 'Official production mechanisms/configuration differ')
    require(request.get('operators') == {'bf16': {'planner': 'official_auto', 'decode': 'official_auto'}},
            'Official A cannot select the historical FA3/XQA experimental backend')
    dependency_id = request.get('runtime_dependency_build_id', '')
    require(re.fullmatch('upstream-ac-261274e5-20261102[a-z]', dependency_id) is not None,
            'Private runtime dependency lacks a frozen official build identity')
    require(_path(request.get('runtime_library_directory')) ==
            approval['approved_workspace'] + '/agent_work/builds/' + dependency_id + '/zeromq-prefix/lib',
            'Private runtime dependency escaped its frozen build')
    require(re.fullmatch('[0-9a-f]{64}', request.get('zeromq_library_sha256', '')) is not None,
            'Missing private runtime library identity')
    require(request.get('native_configuration') is None, 'Official A cannot accept native KV extension configuration')
    require(request.get('cache_allocator') is None, 'Formal A keeps the official default allocator')
    require(request.get('server_binary_execution') == 'required', 'Qualification must execute the official binary')
    require(request.get('startup_recovery') is None and request.get('environment_recovery') is None,
            'A new official qualification cannot hide an old experimental retry')


def server_command(request: dict, output: Path) -> list[str]:
    service = request['service']
    command = [request['server_binary'], '--model=/models/' + MODEL, '--model_id=' + MODEL,
            '--backend=llm', '--model_impl=python', '--python_model_path=' + request['python_source_repo'],
            '--host=127.0.0.1', '--port=19001', '--enable_graph=true',
            '--python_graph_backend=cudagraphs', '--enable_prefix_cache=true', '--enable_chunked_prefill=true',
            '--max_memory_utilization=' + str(service['max_memory_utilization']),
            '--block_size=' + str(service['block_size']),
            '--max_tokens_per_batch=' + str(service['max_tokens_per_batch']),
            '--max_seqs_per_batch=' + str(service['max_batch']), '--max_concurrent_requests=200',
            '--enable_online_profile=false']
    native = request.get('native_configuration')
    if request.get('cache_allocator') == 'elastic_vmm':
        command += ['--enable_vmm_kvcache=true', '--vmm_elastic=true']
    if native is not None:
        command += ['--kv_cache_dtype=' + native['dtype'], '--kv_cache_k_scale=' + str(native['key_scale']),
                    '--kv_cache_v_scale=' + str(native['value_scale']),
                    '--kv_cache_prefill_backend=' + native['prefill'], '--kv_cache_decode_backend=' + native['decode']]
    return command


def verify_packaging_recovery(request: dict, root: Path):
    recovery = request.get('packaging_recovery')
    if recovery is None:
        return
    import hashlib
    from .durable import digest
    if request.get('replicate') != 1 or recovery.get('reason') != 'standalone_evidence_module_freeze':
        raise ValueError('Only the one explicit standalone-module packaging repair is supported')
    prior = Path(request['approved_workspace'])/'agent_work/runs'/recovery['prior_run_id']
    raw = (prior/'result.json').read_bytes()
    result = json.loads(raw)
    old = json.loads((prior/'ticket.json').read_bytes())['request']
    if (hashlib.sha256(raw).hexdigest() != recovery['prior_result_sha256']
            or result.get('status') != 'FAILED' or result.get('gpu_release_verified') is not True
            or result.get('live_owned_containers') != [] or result.get('arm_results') != []
            or result.get('request_sha256') != digest(old)
            or "ModuleNotFoundError: No module named 'scripts.audit_native_results'" not in
               (prior/'output/bf16/container.log').read_text()):
        raise ValueError('Packaging repair predecessor is not the exact preserved released failure')
    for field in ('model_id', 'model_config_sha256', 'server_source_commit', 'python_source_commit',
                  'binary_sha256', 'service', 'operators', 'gpu_uuids', 'dataset_sha256', 'experiment_key'):
        if request[field] != old[field]:
            raise ValueError('Packaging recovery changed scientific deployment: '+field)
    proof_raw = (root/'input/packaging_import_CPU.json').read_bytes()
    if (hashlib.sha256(proof_raw).hexdigest() != recovery['CPU_import_evidence_sha256']
            or json.loads(proof_raw).get('frozen_entry_imports_verified') is not True):
        raise ValueError('Packaging repair lacks its actual frozen-entry CPU import proof')


def trace_summary(directory: Path) -> dict:
    traces = sorted(directory.glob('*.pt.trace.json'))
    if not traces:
        raise ValueError('Official profiler did not produce a trace')
    launches = 0
    names: dict[str, int] = {}
    for path in traces:
        if path.stat().st_size > 256 * 2**20:
            raise ValueError('Qualification trace exceeded its bounded diagnostic size')
        for event in json.loads(path.read_bytes()).get('traceEvents', []):
            name = event.get('name', '')
            if event.get('cat') in ('cuda_runtime', 'cuda_driver') and name in ('cudaGraphLaunch', 'cudaGraphLaunch_ptsz'):
                launches += 1
            if event.get('cat') == 'kernel':
                names[name] = names.get(name, 0) + 1
    return {'trace_files': [str(p) for p in traces], 'actual_cuda_graph_launches': launches,
            'kernel_names_and_counts': names, 'all_requests_in_graph_claim': False}


def _application_metrics() -> dict:
    from .service_metrics import parse_metric

    values = {}
    for name in ('prefix_cache_prompt_tokens_total', 'prefix_cache_hit_tokens_total',
                 'num_used_blocks', 'num_blocks_in_prefix_cache',
                 'num_running_requests', 'num_waiting_requests'):
        with urllib.request.urlopen('http://127.0.0.1:19001/vars/' + name, timeout=5) as response:
            body = response.read(128 * 1024 + 1)
        if len(body) > 128 * 1024:
            raise ValueError('Official application metric response exceeded its bound')
        values[name] = parse_metric(name, body.decode())
    return values


def run(ticket: dict, root: Path, output: Path, arm: str, env: dict) -> dict:
    from .nextgen import _stream_case, native_python_environment
    from .worker import stop_child, file_sha
    from .nextgen_multiturn import owned_loopback_generate, _packets
    from .native_tools import parse_sse_packets

    request = ticket['request']
    verify_packaging_recovery(request, root)
    from .upstream_graph_observer import configure, snapshot, reuse_model_proof
    reused_proof = reuse_model_proof(request, root)
    if arm not in ('bf16', 'fp8_e4m3', 'nvfp4', 'mxfp4'):
        raise ValueError('Unknown frozen qualification format')
    # The legacy worker sets experimental variables; the official process must
    # receive none of them. Preserve the image's CUDA compatibility libraries.
    env = native_python_environment(request, {k: v for k, v in env.items() if not k.startswith('XLLM_')})
    env['PYTHONPATH'] = request['python_source_repo'] + ':' + env.get('PYTHONPATH', '')
    dependency = request['runtime_library_directory']
    if file_sha(Path(dependency) / 'libzmq.so.5.2.5') != request['zeromq_library_sha256']:
        raise ValueError('Private ZeroMQ runtime library changed')
    env['LD_LIBRARY_PATH'] = dependency + ':' + env.get('LD_LIBRARY_PATH', '')
    if request.get('GPU_context_test_binary') and reused_proof is None:
        test = Path(request['GPU_context_test_binary'])
        if (not str(test).startswith(request['approved_workspace']+'/agent_work/builds/')
                or file_sha(test) != request['GPU_context_test_sha256']):
            raise ValueError('Deferred GPU-context cache test executable changed')
        with (output/'GPU_context_cache_tests.log').open('x') as log:
            subprocess.run([str(test), '--gtest_filter=HostKVCacheTest.*'], env=env,
                           stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True)
    write_new_json(output / 'official_environment.json', {k: v for k, v in env.items()
                   if k.startswith('XLLM_') or k in ('LD_LIBRARY_PATH', 'PYTHONPATH')})
    env = configure(request, root, output, env)
    write_new_json(output/'graph_observer_environment.json', {k: env[k] for k in ('LD_PRELOAD', 'KVRA_GRAPH_COUNTER_FILE')})
    inputs = json.loads(Path(request['dataset_path']).read_bytes())
    if len(inputs.get('cases', [])) != 16 or len({c['id'] for c in inputs['cases']}) != 16:
        raise ValueError('Official qualification must reuse the 16 frozen known-answer cases')
    command = server_command(request, output)
    write_new_json(output / 'server_command.json', command)
    responses = []
    before = after = None
    with (output / 'server.log').open('x') as log:
        server = subprocess.Popen(command, env=env, cwd=output, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + request['startup_seconds']
            while True:
                if server.poll() is not None:
                    raise RuntimeError('Official xLLM server exited during startup')
                if time.monotonic() > deadline:
                    raise TimeoutError('Official xLLM startup deadline exceeded')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:19001/v1/models', timeout=2) as response:
                        models = json.load(response)
                    if models.get('data'):
                        break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(1)
            if reused_proof is not None:
                prefix = reused_proof['cross_request_prefix_probe']
                write_new_json(output/'reused_model_prefix_proof.json', reused_proof)
            else:
                for case in inputs['cases']:
                    result = _stream_case(case, MODEL)
                    write_new_json(output / ('case-' + case['id'] + '.json'), result)
                    responses.append(result)
                    if not result['contract_valid'] or not result['expected_substring_present']:
                        raise ValueError('Official model known-answer/natural-stop check failed: ' + case['id'])
                probe = inputs['prefix_probe']
                payload = probe['payload']
                if (payload.get('model') != MODEL or payload.get('temperature') != 0
                        or payload.get('max_tokens') != 512 or payload.get('ignore_eos', False)
                        or not 8192 <= probe['prompt_tokens_cpu'] <= 20480):
                    raise ValueError('Prefix probe must use the frozen, full real GovReport request')
                prefix_results = []
                snapshots = [_application_metrics()]
                for repeat in range(2):
                    raw = owned_loopback_generate(payload)
                    write_new_json(output / ('prefix-raw-' + str(repeat) + '.json'), raw)
                    if raw['status'] != 200 or raw.get('error') is not None:
                        raise ValueError('Official cross-request prefix HTTP failed')
                    parsed = parse_sse_packets(_packets(raw['raw_SSE']))
                    write_new_json(output / ('prefix-parsed-' + str(repeat) + '.json'), parsed)
                    if (parsed['usage']['prompt_tokens'] != probe['prompt_tokens_cpu']
                            or not 0 < parsed['usage']['completion_tokens'] <= payload['max_tokens']):
                        raise ValueError('Official prefix probe usage differs from the frozen tokenizer/cap')
                    prefix_results.append(parsed)
                    snapshots.append(_application_metrics())
                    if repeat == 0:
                        unrelated = _stream_case(inputs['cases'][0], MODEL)
                        write_new_json(output / 'prefix-intervening-independent-request.json', unrelated)
                        if not unrelated['contract_valid'] or not unrelated['expected_substring_present']:
                            raise ValueError('Intervening independent request failed')
                cold_hits = snapshots[1]['prefix_cache_hit_tokens_total'] - snapshots[0]['prefix_cache_hit_tokens_total']
                repeated_hits = snapshots[2]['prefix_cache_hit_tokens_total'] - snapshots[1]['prefix_cache_hit_tokens_total']
                prefix = {'snapshots': snapshots, 'cold_hit_tokens': cold_hits, 'repeat_hit_tokens': repeated_hits,
                          'same_answer_after_independent_request': prefix_results[0]['content'] == prefix_results[1]['content'],
                          'API_cached_tokens_used_as_hit_evidence': False}
                write_new_json(output / 'cross_request_prefix_probe.json', prefix)
                if repeated_hits <= 0 or not prefix['same_answer_after_independent_request']:
                    raise ValueError('Official cross-request prefix/GDN restore qualification failed')
            # Warm the exact serving bucket before the observation window so
            # capture/warmup calls cannot count as serving replay evidence.
            warmed = _stream_case(inputs['cases'][0], MODEL)
            write_new_json(output/'graph-window-warmup.json', warmed)
            if not warmed['contract_valid'] or not warmed['expected_substring_present']:
                raise ValueError('Graph-window model warmup failed')
            before = snapshot(output/'cuda_graph_calls.bin', server.pid)
            for number in range(2):
                result = _stream_case(inputs['cases'][0], MODEL)
                write_new_json(output/('profile-repeat-'+str(number)+'.json'), result)
                if not result['contract_valid'] or not result['expected_substring_present']:
                    raise ValueError('Graph-window actual serving request failed')
            after = snapshot(output/'cuda_graph_calls.bin', server.pid)
            counts = {name: after[name]-before[name] for name in
                      ('attempts', 'cudaGraphLaunch_success', 'cudaGraphLaunch_ptsz_success', 'failures')}
            trace = {'method': 'transparent_forwarding_of_actual_CUDA_Runtime_API_calls',
                     'before': before, 'after': after, 'window_counts': counts,
                     'actual_cuda_graph_launches': counts['cudaGraphLaunch_success']+counts['cudaGraphLaunch_ptsz_success'],
                     'all_requests_in_graph_claim': False, 'formal_performance_measurement': False}
            write_new_json(output/'trace_summary.json', trace)
            if trace['actual_cuda_graph_launches'] <= 0 or counts['failures'] != 0:
                raise ValueError('No successful serving Graph API launch in the measured diagnostic window')
        finally:
            stop_child(server)
    classification = ('untimed_actual_VMM_profile_model_qualification' if request['workload']['kind'] == 'upstream_native_VMM_compatibility'
                      else 'shared_path_BF16_minimal_regression' if request['workload']['kind'] == 'upstream_common_path_BF16_regression'
                      else 'official_A_model_qualification' if arm == 'bf16' else 'native_C_model_qualification')
    result = {'classification': classification, 'model_loaded': True,
              'known_answers_checked': 16 if reused_proof else len(responses),
              'known_answers_reused_from_original_run': request.get('model_proof_reuse'),
              'fresh_graph_window_known_requests': 3, 'official_source_commit': OFFICIAL_COMMIT, 'baseline_buildfix_commit': BASELINE_COMMIT, 'actual_model_source_commit': request['python_source_commit'],
              'xLLM_server_binary_executed': True, 'actual_model_graph_replay': True,
              'performance_claim': False, 'quality_benchmark_claim': False,
              'prefix_reuse_claim': True, 'cross_request_prefix_probe': prefix, 'trace': trace}
    write_new_json(output / 'validation_result.json', result)
    return result
