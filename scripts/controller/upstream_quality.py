"""Same-version frozen1720 real-text quality generation on normal A/C serving."""

import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

from .approval import ApprovalError
from .durable import write_new_json
from .upstream_qualification import KIND as A_KIND, server_command
from .upstream_native import MODEL_KIND as C_KIND

KIND = 'upstream_AC_frozen_core_quality1720'
LIMITS = {'requests': 1720, 'client_concurrency': 16, 'client_token_budget': 131072,
          'max_seconds': 5400, 'request_timeout_seconds': 600}


def validate_request(request: dict, approval: dict):
    from .upstream_qualification import validate_request as validate_A
    from .upstream_native import validate_request as validate_C
    if (request.get('workload') != {'kind': KIND, 'scope': 'same_version_GSM8K1319_LongBench401_full_text'}
            or request.get('max_wall_seconds') != 7200 or request.get('graph_observer') is not None):
        raise ApprovalError('Quality requires the exact finite cohort and an unobserved normal serving process')
    q = dict(request)
    official = request['arms'] == ['bf16']
    q['workload'] = {'kind': A_KIND if official else C_KIND,
        'scope': 'official_model_prefix_graph_qualification' if official else 'native_model_prefix_graph_qualification'}
    q['max_wall_seconds'] = 3600
    if request.get('next_stage_profile') is not None:
        from .next_stage_deployment import validate_deployment
        validate_deployment(q, approval)
    else:
        (validate_A if official else validate_C)(q, approval)
    config = request.get('quality_evaluation', {})
    if any(config.get(k) != v for k, v in LIMITS.items()):
        raise ApprovalError('Quality source count or bounded client configuration changed')
    import re
    if any(re.fullmatch('[0-9a-f]{64}', config.get(k, '')) is None
           for k in ('requests_sha256', 'source_manifest_sha256')):
        raise ApprovalError('Quality needs actual frozen source identities')


def run(ticket: dict, root: Path, output: Path, arm: str, env: dict):
    from .nextgen import native_python_environment
    from .nextgen_quality import run_core_generations
    from .worker import file_sha, stop_child
    request = ticket['request']
    reference = request['qualification_reference']
    prior = Path(request['approved_workspace'])/'agent_work/runs'/reference['run_id']
    raw = (prior/'result.json').read_bytes()
    result = json.loads(raw)
    old = json.loads((prior/'ticket.json').read_bytes())['request']
    if (hashlib.sha256(raw).hexdigest() != reference['result_sha256'] or result.get('status') != 'COMPLETE'
            or result.get('gpu_release_verified') is not True or result.get('live_owned_containers') != []):
        raise ValueError('Quality requires an actual completed released same-version model/Graph qualification')
    for key in ('arms', 'model_id', 'model_config_sha256', 'image_id', 'gpu_uuids', 'service',
                'binary_sha256', 'server_source_commit', 'python_source_commit', 'native_configuration'):
        if old.get(key) != request.get(key):
            raise ValueError('Quality and qualified model deployment differ: '+key)
    env = native_python_environment(request, {k:v for k,v in env.items() if not k.startswith('XLLM_')})
    env['PYTHONPATH'] = request['python_source_repo']+':'+env.get('PYTHONPATH', '')
    library = request['runtime_library_directory']
    if file_sha(Path(library)/'libzmq.so.5.2.5') != request['zeromq_library_sha256']:
        raise ValueError('Pinned runtime library changed')
    env['LD_LIBRARY_PATH'] = library+':'+env.get('LD_LIBRARY_PATH', '')
    if env.get('KVRA_GRAPH_COUNTER_FILE') or 'libkvra_graph_counter' in env.get('LD_PRELOAD', ''):
        raise ValueError('Diagnostic observer must not reach quality or formal service')
    command = server_command(request, output)
    write_new_json(output/'server_command.json', command)
    started = time.monotonic()
    with (output/'server.log').open('x') as log:
        server = subprocess.Popen(command, env=env, cwd=output, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            ready_deadline = started+request['startup_seconds']
            while True:
                if server.poll() is not None or time.monotonic() > ready_deadline:
                    raise RuntimeError('Quality server failed or exceeded startup window')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:19001/v1/models', timeout=2) as response:
                        if json.load(response).get('data'):
                            break
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            result = run_core_generations(request, root, output,
                available_seconds=request['max_wall_seconds']-(time.monotonic()-started)-120)
        finally:
            stop_child(server)
    return {'classification': 'same_version_real_1720_quality_generations_pending_independent_scores',
            'quality': result, 'performance_claim': False, 'accuracy_acceptance_claim': False}
