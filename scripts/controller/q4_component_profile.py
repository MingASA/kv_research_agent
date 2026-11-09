"""Bounded diagnostic admission; candidate implementations remain read-only."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .approval import ApprovalError
from .durable import write_new_json

KIND = 'upstream_readonly_Q4_component_profile'
SCOPE = 'SM90_D256_Q24_KV4_supported_components_with_prefill_graph_qualification_BLOCKED'
SOURCE_COMMIT = 'ba5eaec0a1b5b941da4b8ded0343d43f2c2b4158'
PROBE_DRIVER_SHA256 = '71bf6439d29f0558c0a261d514019e65116a8af876afc2c0a5fd384c4ef38965'
GRAPH_FAILURE_REFERENCE = {
    'run_id': 'upstream-fast-q4audit-20261106b-pilot-0001',
    'result_sha256': '0f19e9e2bc0991b8e13585558b97a9c25386a66f3a638ed6774b1f8872b613f0',
    'failure_sha256': '8e65da70156ea0383f6d5e5d809edc4cd34676f6935720e501088beb694bc124',
    'driver_sha256': '7e8736bdfd17392bb3906c256bf3f9d7d98bbceb3ed3c1e4c047a66ecb37e541',
}
BLOCKED_GRAPH_ROUTES = ['mx_prefill_wrapper', 'nv_prefill_wrapper', 'matched_nv_prefill_wrapper']
BLOCKED_GRAPH_REASON = 'pinned_FA2_graph_ready_non_split_padding_mask_defect'
PROTOCOL = {
    'hardware': 'SM90', 'Q_heads': 24, 'KV_heads': 4, 'head_dim': 256,
    'query_dtype': 'bfloat16', 'page_size': 128,
    'main_cases': [['prefill', 1, 8192], ['decode', 32, 2048], ['decode', 32, 16384]],
    'formats': ['mxfp4', 'nvfp4'],
    'NV_decode_routes': ['guarded_paged_prefill', 'native_batch_decode'],
    'SGLang_target_execution': 'unsupported_static_guard_no_GPU_execution',
    'candidate_runtime_source_mutations': False,
    'scale_unpack_inside_fused_Attention_is_not_independent_wall_time': True,
    'full_model_service_speedup_claim': False,
    'execution_scope': 'supported_Eager_components_not_full_production_Graph_qualification',
    'blocked_graph_routes': BLOCKED_GRAPH_ROUTES,
    'blocked_graph_reason': BLOCKED_GRAPH_REASON,
    'production_graph_qualification': False,
    'all_cases_require_CUPTI_kernel_events': True,
    'conformance': {
        'independent_reference_atol_rtol': [0.02, 0.02],
        'same_fused_MX_matched_NV_strict_atol_rtol': [0.001, 0.001],
        'same_fused_MX_matched_NV_is_hard_gate': True,
        'cross_prefill_native_decode_strict_atol_rtol': [0.001, 0.001],
        'cross_prefill_native_decode_equivalence_is_required_report_not_hard_gate': True,
        'original_failed_pilot_and_strict_failures_preserved': True,
    },
}


def validate_acceptance(acceptance: dict, result: dict, phase: str) -> None:
    expected_cases = (['pilot_prefill_b1_s128', 'pilot_decode_b2_s129'] if phase == 'pilot'
                      else ['prefill_b1_s8192', 'decode_b32_s2048', 'decode_b32_s16384'])
    hard = ('completed', 'pass', 'source_checked', 'all_independent_reference_passed',
            'same_fused_MX_NV_control_passed', 'all_writer_graph_checks_passed',
            'all_native_decode_graph_checks_passed', 'all_component_gpu_activity_observed')
    gates = hard[3:] + ('strict_prefill_native_decode_equivalence_all_pass',)
    if (acceptance.get('schema') != 'q4_supported_component_acceptance_v3'
            or result.get('schema') != 'q4_supported_component_result_v3'
            or acceptance.get('phase') != phase or result.get('phase') != phase
            or any(acceptance.get(key) is not True for key in hard)
            or type(acceptance.get('strict_prefill_native_decode_equivalence_all_pass')) is not bool
            or any(key in acceptance or key in result for key in ('all_route_conformance_passed', 'all_graph_checks_passed'))
            or any(type(result.get(key)) is not bool or result.get(key) != acceptance.get(key) for key in gates)
            or any(value.get('production_graph_qualification') is not False
                   or value.get('all_requested_graph_paths_supported') is not False
                   or value.get('blocked_graph_routes') != BLOCKED_GRAPH_ROUTES
                   or value.get('blocked_graph_reason') != BLOCKED_GRAPH_REASON
                   for value in (acceptance, result))
            or acceptance.get('model_loaded') is not False
            or acceptance.get('service_benchmark') is not False
            or acceptance.get('full_vllm_engine_tested') is not False
            or acceptance.get('sglang_target') != 'unsupported_NA'
            or type(acceptance.get('artifact_bytes_before_acceptance')) is not int
            or not 0 < acceptance['artifact_bytes_before_acceptance'] <= 64 * 2**20
            or result.get('completed') is not True or result.get('source_checked') is not True
            or result.get('service_or_model_quality_claim') is not False
            or [case.get('case', {}).get('name') for case in result.get('cases', [])] != expected_cases):
        raise ValueError('Incomplete or mismatched Q4 correctness/Graph diagnostic evidence')


def validate_request(request: dict, approval: dict) -> None:
    from .native_next_stage import KIND as ORIGINAL_KIND, validate_request as validate_original
    driver = Path(__file__).with_name('q4_component_probe.py')
    if (request.get('workload') != {'kind': KIND, 'scope': SCOPE}
            or request.get('profile_protocol') != PROTOCOL
            or request.get('profile_phase') not in ('pilot', 'main')
            or request.get('graph_qualification_failure_reference') != GRAPH_FAILURE_REFERENCE
            or request.get('python_source_commit') != SOURCE_COMMIT
            or request.get('next_stage_profile') != 'ocp_mxfp4_20261105'
            or request.get('server_binary_execution') != 'none'
            or type(request.get('max_wall_seconds')) is not int
            or not 1800 <= request['max_wall_seconds'] <= 3600
            or not driver.is_file()
            or request.get('probe_driver_sha256') != PROBE_DRIVER_SHA256
            or request.get('probe_driver_sha256') != hashlib.sha256(driver.read_bytes()).hexdigest()):
        raise ApprovalError('Q4 profiling needs its exact bounded source/geometry/driver contract')
    pilot = request.get('profile_pilot_reference')
    if request['profile_phase'] == 'main':
        if (not isinstance(pilot, dict) or set(pilot) != {'run_id', 'result_sha256'}
                or re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,127}', pilot.get('run_id', '')) is None
                or pilot['run_id'] == request.get('run_id')
                or re.fullmatch('[0-9a-f]{64}', pilot.get('result_sha256', '')) is None):
            raise ApprovalError('Main component profiling requires its immutable successful pilot reference')
    elif pilot is not None:
        raise ApprovalError('Pilot cannot adopt another diagnostic as its own prerequisite')
    # Preserve all established authority, identity, quota, source, runtime,
    # storage and single-GPU admission constraints. This projection authorizes
    # no model server, extra device, unsupported SGLang kernel or arbitrary CLI.
    common = dict(request, workload={'kind': ORIGINAL_KIND,
        'scope': 'OCP_MXFP4_SM90_D256_bitwise_paged_Graph_gate'})
    validate_original(common, approval)


def verify_original_graph_rejection(request: dict, output: Path) -> None:
    """Keep failed full-Graph qualification distinct from restricted measurement."""
    from .durable import digest
    reference = request['graph_qualification_failure_reference']
    if reference != GRAPH_FAILURE_REFERENCE:
        raise ValueError('Supported-component scope requires the exact preserved original Graph failure')
    prior = Path(request['approved_workspace'])/'agent_work/runs'/reference['run_id']
    paths = [prior/'result.json', prior/'ticket.json', prior/'output/mxfp4/profile/failure.json']
    for path in paths:
        if path.resolve(strict=True) != path or not path.is_file() or path.stat().st_uid != os.getuid():
            raise ValueError('Original Graph rejection is not an owned canonical artifact')
    raw, ticket_raw, failure_raw = (path.read_bytes() for path in paths)
    result, ticket, failure = (json.loads(value) for value in (raw, ticket_raw, failure_raw))
    old = ticket['request']
    if (hashlib.sha256(raw).hexdigest() != reference['result_sha256']
            or hashlib.sha256(failure_raw).hexdigest() != reference['failure_sha256']
            or result.get('status') != 'FAILED' or result.get('gpu_release_verified') is not True
            or result.get('live_owned_containers') != []
            or result.get('run_id') != reference['run_id'] or result.get('arm_results') != []
            or result.get('request_sha256') != ticket.get('request_sha256')
            or digest(old) != ticket.get('request_sha256')
            or old.get('python_source_commit') != request.get('python_source_commit')
            or old.get('image_id') != request.get('image_id')
            or old.get('backend_source_sha256') != request.get('backend_source_sha256')
            or old.get('workload', {}).get('kind') != KIND or old.get('profile_phase') != 'pilot'
            or old.get('probe_driver_sha256') != reference['driver_sha256']
            or failure.get('type') != 'AcceleratorError'
            or 'illegal memory access' not in failure.get('message', '')):
        raise ValueError('Original failed Graph qualification evidence changed')
    write_new_json(output/'original_graph_rejection.json', dict(reference,
        original_status='FAILED', original_run_released=True, original_run_successful=False,
        production_graph_qualification=False, scope='supported components only'))


def verify_pilot(request: dict, output: Path) -> None:
    if request['profile_phase'] != 'main':
        return
    from .next_stage_deployment import _completed_reference
    prior, old = _completed_reference(request, request['profile_pilot_reference'])
    if any(old.get(key) != request.get(key) for key in (
            'workload', 'profile_protocol', 'python_source_commit', 'probe_driver_sha256',
            'image_id', 'backend_source_sha256', 'gpu_uuids',
            'graph_qualification_failure_reference')) or old.get('profile_phase') != 'pilot':
        raise ValueError('Main diagnostic must reuse its source/driver/protocol/GPU-qualified pilot')
    profile = prior/'output/mxfp4/profile'
    validate_acceptance(json.loads((profile/'acceptance.json').read_bytes()),
                        json.loads((profile/'result.json').read_bytes()), 'pilot')
    write_new_json(output/'pilot_reuse.json', dict(request['profile_pilot_reference'],
        source_driver_protocol_GPU_checked=True, actual_completed_released_pilot=True))


def run(ticket: dict, root: Path, output: Path, arm: str, env: dict) -> dict:
    from .nextgen import native_python_environment
    request = ticket['request']
    verify_original_graph_rejection(request, output)
    verify_pilot(request, output)
    env = native_python_environment(request, {k: v for k, v in env.items() if not k.startswith('XLLM_')})
    env['PYTHONPATH'] = request['python_source_repo'] + ':' + env.get('PYTHONPATH', '')
    command = [sys.executable, '-B', '-m', __package__+'.q4_component_probe',
               '--output', str(output/'profile'), '--phase', request['profile_phase']]
    write_new_json(output/'profile_command.json', command)
    with (output/'profile.log').open('x') as log:
        subprocess.run(command, env=env, cwd=request['python_source_repo'], stdout=log,
                       stderr=subprocess.STDOUT, check=True,
                       timeout=min(2400, request['max_wall_seconds']-120))
    acceptance = json.loads((output/'profile/acceptance.json').read_bytes())
    result = json.loads((output/'profile/result.json').read_bytes())
    validate_acceptance(acceptance, result, request['profile_phase'])
    return {'classification': 'readonly_Q4_supported_components_full_prefill_graph_qualification_BLOCKED',
            'profile_phase': request['profile_phase'], 'acceptance': acceptance,
            'model_loaded': False, 'candidate_runtime_source_mutations': False,
            'full_model_service_speedup_claim': False, 'quality_claim': False}
