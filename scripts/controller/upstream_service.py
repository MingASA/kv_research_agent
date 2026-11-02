"""Finite A/C production cases using the existing AIPerf raw-wire auditor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

from .durable import write_new_json
from .upstream_qualification import KIND as A_KIND, MODEL, server_command, _application_metrics
from .upstream_native import MODEL_KIND as C_KIND

KIND = 'upstream_AC_fixed_production_service'
CASES = (('sharegpt-c32', 'sharegpt', 'measured512', 'warmup16', 32),
         ('sharegpt-c64', 'sharegpt', 'measured512', 'warmup16', 64),
         ('govreport-c32', 'govreport', 'measured64', 'warmup4', 32))


def validate_request(request: dict, approval: dict):
    from .approval import ApprovalError
    from .upstream_qualification import validate_request as validate_A
    from .upstream_native import validate_request as validate_C

    if (request.get('workload') != {'kind': KIND, 'scope': 'fixed512_sharegpt_and64_govreport_AC_v1'}
            or request.get('benchmark_cases') != [c[0] for c in CASES]
            or request.get('graph_observer') is not None or request.get('model_proof_reuse') is not None
            or request.get('max_wall_seconds') != 7200):
        raise ApprovalError('Production service requires the frozen finite three-case campaign')
    qualification = dict(request)
    official = request.get('arms') == ['bf16']
    qualification['workload'] = {'kind': A_KIND if official else C_KIND,
        'scope': 'official_model_prefix_graph_qualification' if official else 'native_model_prefix_graph_qualification'}
    qualification['max_wall_seconds'] = 3600
    (validate_A if official else validate_C)(qualification, approval)
    config = request.get('service_benchmark', {})
    if (set(config) != {'input_files_sha256', 'aiperf_cli_config_sha256', 'qualification_run_id',
                        'qualification_result_sha256', 'full_CLI_protocol_CPU_audit_sha256'}
            or set(config['input_files_sha256']) != {'sharegpt/manifest.json', 'sharegpt/measured512.json',
                'sharegpt/retokenized_Qwen38_CPU-a1.json',
                'sharegpt/warmup16.json', 'govreport/manifest.json', 'govreport/measured64.json', 'govreport/warmup4.json'}):
        raise ApprovalError('Production inputs or qualification identities are incomplete')
    import re
    hashes = list(config['input_files_sha256'].values()) + [config[k] for k in
        ('aiperf_cli_config_sha256', 'qualification_result_sha256', 'full_CLI_protocol_CPU_audit_sha256')]
    if any(re.fullmatch('[0-9a-f]{64}', value) is None for value in hashes):
        raise ApprovalError('Production provenance requires exact SHA256 values')
    if re.fullmatch('[A-Za-z0-9_-]{1,128}', config['qualification_run_id']) is None:
        raise ApprovalError('Invalid source qualification run identity')


def verify_prerequisites(request: dict, root: Path):
    from .nextgen_performance import validate_payloads, validate_cli_protocol_proof

    config = request['service_benchmark']
    for name, expected in config['input_files_sha256'].items():
        raw = (root/'input/production'/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Production cohort changed: '+name)
        if Path(name).name.startswith(('measured', 'warmup')):
            observed = validate_payloads(raw, model_id=MODEL, latest_upstream=True)
            expected_count = {'measured512.json': 512, 'warmup16.json': 16,
                              'measured64.json': 64, 'warmup4.json': 4}[Path(name).name]
            if observed['requests'] != expected_count or observed['sessions'] != expected_count:
                raise ValueError('Fixed single-turn production cardinality changed')
    proof = root/'input/full_CLI_protocol_CPU_audit.json'
    if hashlib.sha256(proof.read_bytes()).hexdigest() != config['full_CLI_protocol_CPU_audit_sha256']:
        raise ValueError('Existing full CLI protocol proof changed')
    validate_cli_protocol_proof(json.loads(proof.read_bytes()), require_session_limit=True)
    prior = Path(request['approved_workspace'])/'agent_work/runs'/config['qualification_run_id']
    raw = (prior/'result.json').read_bytes()
    result = json.loads(raw)
    old = json.loads((prior/'ticket.json').read_bytes())['request']
    if (hashlib.sha256(raw).hexdigest() != config['qualification_result_sha256']
            or result.get('status') != 'COMPLETE' or result.get('gpu_release_verified') is not True
            or result.get('live_owned_containers') != []):
        raise ValueError('Production requires the actual released successful qualification')
    for key in ('arms', 'model_id', 'model_config_sha256', 'image_id', 'gpu_uuids', 'service',
                'binary_sha256', 'server_source_commit', 'python_source_commit', 'native_configuration'):
        if old.get(key) != request.get(key):
            raise ValueError('Production and qualified deployment differ: '+key)


def run(ticket: dict, root: Path, output: Path, arm: str, env: dict):
    from .nextgen import native_python_environment
    from .nextgen_performance import aiperf_command
    from .aiperf_cli_admission import validate_service_cli
    from .production import audit_requests
    from .worker import file_sha, monitor, stop_child
    from .service_metrics import ServiceMetricCollector

    request = ticket['request']
    verify_prerequisites(request, root)
    env = native_python_environment(request, {k: v for k, v in env.items() if not k.startswith('XLLM_')})
    env['PYTHONPATH'] = request['python_source_repo']+':'+env.get('PYTHONPATH', '')
    library = request['runtime_library_directory']
    if file_sha(Path(library)/'libzmq.so.5.2.5') != request['zeromq_library_sha256']:
        raise ValueError('Shared private runtime library changed')
    env['LD_LIBRARY_PATH'] = library+':'+env.get('LD_LIBRARY_PATH', '')
    if env.get('KVRA_GRAPH_COUNTER_FILE') or 'libkvra_graph_counter' in env.get('LD_PRELOAD', ''):
        raise ValueError('Diagnostic observer must not reach formal service')
    client_env = dict(env, PYTHONPATH=str(root/'runtime_support/tokenizer_compat'),
        CJYU_AIPERF_LOCAL_TOKENIZER_SHIM='1', CUDA_VISIBLE_DEVICES='',
        AIPERF_DATASET_MMAP_CACHE_ENABLED='true', AIPERF_DATASET_MMAP_CACHE_DIR='/run_cache/aiperf_mmap')
    cases = {}
    deadline = time.monotonic_ns() + (request['max_wall_seconds']-180)*10**9
    for name, dataset, measured, warmup, concurrency in CASES:
        directory = output/name
        directory.mkdir()
        command = server_command(request, directory)
        # A fresh process resets cache state for each declared pressure point.
        write_new_json(directory/'server_command.json', command)
        with (directory/'server.log').open('x') as log:
            server = subprocess.Popen(command, env=env, cwd=directory, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            try:
                ready_deadline = min(deadline, time.monotonic_ns()+request['startup_seconds']*10**9)
                while True:
                    if server.poll() is not None or time.monotonic_ns() > ready_deadline:
                        raise RuntimeError('Production server failed or exceeded its startup window')
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:19001/v1/models', timeout=2) as response:
                            if json.load(response).get('data'):
                                break
                    except (OSError, ValueError):
                        pass
                    time.sleep(1)
                import re
                capacities = re.findall(r'kv cache capacity: [^\n]*?blocks: (\d+), slot_size:',
                                        (directory/'server.log').read_text())
                if len(capacities) != 1:
                    raise ValueError('Missing unique actual single-worker physical KV capacity')
                pages = int(capacities[0])
                phases = {}
                for phase, source_name in (('warmup', warmup), ('measured', measured)):
                    phase_dir = directory/phase
                    phase_dir.mkdir()
                    source = root/'input/production'/dataset/(source_name+'.json')
                    count = len(json.loads(source.read_bytes())['data'])
                    demand = min(count, concurrency)
                    argv = aiperf_command(MODEL, source, phase_dir/'artifacts', requests=count,
                        concurrency=demand, sessions=count, latest_upstream=True)
                    write_new_json(phase_dir/'command.json', argv)
                    validate_service_cli(argv, root, phase_dir, client_env,
                                         request['service_benchmark']['aiperf_cli_config_sha256'])
                    before = _application_metrics()
                    collector = ServiceMetricCollector(phase_dir/'application_metrics.jsonl',
                        request['run_id'], ticket['request_sha256'], pages,
                        accounting_version='upstream261_effective_usage_excludes_padding')
                    collector.start()
                    started = time.monotonic_ns()
                    try:
                        with (phase_dir/'aiperf.log').open('x') as client_log:
                            client = subprocess.Popen(argv, env=client_env, cwd=root, stdout=client_log,
                                stderr=subprocess.STDOUT, start_new_session=True)
                            try:
                                monitor(client, server, phase_dir/'gpu_telemetry.csv', deadline_ns=deadline)
                                if client.wait() != 0:
                                    raise RuntimeError('AIPerf failed; retain full/partial raw records')
                            finally:
                                stop_child(client)
                    finally:
                        collector.stop()
                    after = _application_metrics()
                    write_new_json(phase_dir/'prefix_counter_window.json', {'before': before, 'after': after,
                        'hit_tokens_delta': after['prefix_cache_hit_tokens_total']-before['prefix_cache_hit_tokens_total'],
                        'prompt_tokens_delta': after['prefix_cache_prompt_tokens_total']-before['prefix_cache_prompt_tokens_total'],
                        'API_cached_tokens_used_as_hit_evidence': False})
                    audit = audit_requests(phase_dir/'artifacts', source, count, 0, allow_multi_turn=True)
                    if dataset == 'sharegpt':
                        expected_counts = {row['session_id']: row['prompt_tokens'] for row in json.loads(
                            (root/'input/production/sharegpt/retokenized_Qwen38_CPU-a1.json').read_bytes())['requests']
                            if row['phase'] == source_name}
                    else:
                        entry = json.loads((root/'input/production/govreport/manifest.json').read_bytes())['files'][source_name]
                        expected_counts = {row['session_id']: row['prompt_tokens_cpu'] for row in entry['lineage']}
                    observed_counts = {}
                    for raw_row in (phase_dir/'artifacts/profile_export.jsonl').read_bytes().split(b'\n'):
                        if raw_row:
                            row = json.loads(raw_row)
                            observed_counts[row['metadata']['conversation_id']] = int(row['metrics']['usage_prompt_tokens']['value'])
                    if observed_counts != expected_counts:
                        raise ValueError('Actual server prompt token counts differ from frozen official27B tokenizer')
                    audit['all_official_tokenizer_and_server_prompt_counts_match'] = True
                    audit.update(worker_observation_started_monotonic_ns=started,
                                 worker_observation_ended_monotonic_ns=time.monotonic_ns())
                    write_new_json(phase_dir/'independent_worker_audit.json', audit)
                    phases[phase] = audit
                cases[name] = phases
            finally:
                stop_child(server)
    result = {'classification': 'latest_upstream_real_fixed_A_C_service', 'cases': cases,
              'official_AIPerf_complete': True, 'warmup_external_and_unmeasured': True,
              'fresh_server_per_case': True, 'quality_acceptance_claim': False,
              'all_requests_Graph_coverage_claim': False}
    write_new_json(output/'service_result.json', result)
    return result
