"""Reuse the real32 native-tool conversations on the fixed upstream A/C model."""

import json
from pathlib import Path
import subprocess
import time
import urllib.request

from .approval import ApprovalError
from .durable import write_new_json
from .nextgen_tools_service import validate_tools_contract, run_tools
from .upstream_qualification import KIND as A_KIND, server_command
from .upstream_native import MODEL_KIND as C_KIND

KIND = 'upstream_AC_real_native_tools32'
TOOL_GPU = 'GPU-47374a5f-689b-fa8c-49c4-4aa6840ebfb7'


def validate_request(request: dict, approval: dict):
    from .upstream_qualification import validate_request as validate_A
    from .upstream_native import validate_request as validate_C
    if (request.get('workload') != {'kind': KIND, 'scope': 'same_version_original32_112turns_native_tools'}
            or request.get('max_wall_seconds') != 5400 or request.get('graph_observer') is not None
            or request.get('gpu_uuids') != [TOOL_GPU]):
        raise ApprovalError('Tool evaluation requires the observer-free finite original conversation cohort')
    q = dict(request)
    official = request['arms'] == ['bf16']
    q['workload'] = {'kind': A_KIND if official else C_KIND,
        'scope': 'official_model_prefix_graph_qualification' if official else 'native_model_prefix_graph_qualification'}
    q['max_wall_seconds'] = 3600
    (validate_A if official else validate_C)(q, approval)
    validate_tools_contract(request)


def run(ticket: dict, root: Path, output: Path, arm: str, env: dict):
    from .nextgen import native_python_environment
    from .deployment_quality import verify_quality_reference
    from .worker import file_sha, stop_child
    request = ticket['request']
    verify_quality_reference(request, root, root.stat().st_uid, 'input/tool_quality_scores.json',
                             request['tool_evaluation']['quality_scores_sha256'])
    env = native_python_environment(request, {k:v for k,v in env.items() if not k.startswith('XLLM_')})
    env['PYTHONPATH'] = request['python_source_repo']+':'+env.get('PYTHONPATH', '')
    library = request['runtime_library_directory']
    if file_sha(Path(library)/'libzmq.so.5.2.5') != request['zeromq_library_sha256']:
        raise ValueError('Pinned runtime library changed')
    env['LD_LIBRARY_PATH'] = library+':'+env.get('LD_LIBRARY_PATH', '')
    if env.get('KVRA_GRAPH_COUNTER_FILE') or 'libkvra_graph_counter' in env.get('LD_PRELOAD', ''):
        raise ValueError('Diagnostic observer must not reach tool quality')
    command = server_command(request, output)+['--tool_call_parser=qwen3_coder']
    write_new_json(output/'server_command.json', command)
    with (output/'server.log').open('x') as log:
        server = subprocess.Popen(command, env=env, cwd=output, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic()+request['startup_seconds']
            while True:
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Tool server failed or exceeded startup window')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:19001/v1/models', timeout=2) as response:
                        if json.load(response).get('data'):
                            break
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            result = run_tools(request, root, output)
        finally:
            stop_child(server)
    return {'classification': 'same_version_real_native32_tool_generations_pending_independent_scores',
            'tool_generations': result, 'performance_claim': False, 'accuracy_acceptance_claim': False}
