"""Untimed observation of successful serving CUDA Runtime Graph API calls."""

import hashlib
import json
from pathlib import Path
import struct


def snapshot(path: Path, pid: int):
    if not path.exists():
        return {'pid': pid, 'attempts': 0, 'cudaGraphLaunch_success': 0,
                'cudaGraphLaunch_ptsz_success': 0, 'failures': 0, 'initialized': False}
    if path.is_symlink() or path.stat().st_size != 64:
        raise ValueError('Invalid diagnostic CUDA API counter file')
    words = struct.unpack('=8Q', path.read_bytes())
    if words[:3] != (0x4b56524147524150, 1, pid):
        raise ValueError('CUDA API counter does not belong to the actual serving process')
    return dict(zip(('pid', 'attempts', 'cudaGraphLaunch_success', 'cudaGraphLaunch_ptsz_success', 'failures'),
                    words[2:7]), initialized=True)


def configure(request: dict, root: Path, output: Path, env: dict):
    config = request['graph_observer']
    library = root/'input/libkvra_graph_counter.so'
    source = root/'input/cuda_graph_counter.c'
    if (config['kind'] != 'transparent_cuda_runtime_graph_counter_v1'
            or hashlib.sha256(library.read_bytes()).hexdigest() != config['library_sha256']
            or hashlib.sha256(source.read_bytes()).hexdigest() != config['source_sha256']):
        raise ValueError('Diagnostic observer binary or source changed')
    return dict(env, LD_PRELOAD=str(library)+(':'+env['LD_PRELOAD'] if env.get('LD_PRELOAD') else ''),
                KVRA_GRAPH_COUNTER_FILE=str(output/'cuda_graph_calls.bin'))


def reuse_model_proof(request: dict, root: Path):
    reference = request.get('model_proof_reuse')
    if reference is None:
        return None
    certificate_raw = (root/'input/partial_model_proof.json').read_bytes()
    certificate = json.loads(certificate_raw)
    if (hashlib.sha256(certificate_raw).hexdigest() != reference['certificate_sha256']
            or certificate.get('actual_model_and_prefix_verified') is not True
            or certificate.get('known_answers_checked') != 16
            or certificate.get('original_run_status') != 'FAILED'
            or certificate.get('original_failure_scope') != 'unsupported_CUDA_torch_profiler_after_model_prefix_pass'):
        raise ValueError('Cannot reuse an unverified or relabeled model/prefix proof')
    prior_id = reference['prior_run_id']
    if Path(prior_id).name != prior_id or prior_id in ('.', '..'):
        raise ValueError('Unsafe original model proof identity')
    prior = Path(request['approved_workspace'])/'agent_work/runs'/prior_id
    raw = (prior/'result.json').read_bytes()
    result = json.loads(raw)
    old = json.loads((prior/'ticket.json').read_bytes())['request']
    if (hashlib.sha256(raw).hexdigest() != reference['prior_result_sha256']
            or result.get('status') != 'FAILED' or result.get('gpu_release_verified') is not True
            or result.get('live_owned_containers') != []):
        raise ValueError('Original profiling failure is not released and preserved')
    for field in ('model_id', 'model_config_sha256', 'server_source_commit', 'python_source_commit',
                  'binary_sha256', 'service', 'operators', 'gpu_uuids', 'dataset_sha256', 'experiment_key'):
        if request[field] != old[field]:
            raise ValueError('Model proof reuse changed the scientific deployment: '+field)
    for relative, expected in certificate['original_files_sha256'].items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or not str(path).startswith(('input/', 'output/bf16/')):
            raise ValueError('Unsafe original evidence path')
        if hashlib.sha256((prior/path).read_bytes()).hexdigest() != expected:
            raise ValueError('Original model/prefix evidence changed')
    return certificate
