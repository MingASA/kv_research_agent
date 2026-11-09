"""Bounded P0/P1 qualification on the frozen runtime, distinct from old audits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from .approval import ApprovalError
from .durable import write_new_json
from .upstream_native import OPERATOR_KIND, SOURCE_FILES


KIND = 'upstream_KV_P0_P1_operator'
SCOPE = 'fixed_SM90_D256_Graph_mask_host_planning_and_native_NV_writer'
EXTENSION = {'schema': 'kv_p0_p1_deployment_v1', 'profile': 'kv_p0_p1_20261109'}
PROTOCOL = {
    'revision': '20261109-a1', 'hardware': 'SM90',
    'Q_heads': 24, 'KV_heads': 4, 'head_dim': 256, 'page_size': 128,
    'query_dtype': 'bfloat16', 'seed': 20261109,
    'main_cases': [['prefill', 1, 8192], ['decode', 32, 2048], ['decode', 32, 16384]],
    'event_warmup': 10, 'event_repeats': 40, 'event_inner': 5,
    'plan_warmup': 3, 'plan_repeats': 20,
    'independent_reference_atol_rtol': [0.02, 0.02],
    'P1_min_targeted_median_improvement': 0.10,
    'P1_max_GPU_component_regression': 0.05,
    'reverse_order_replication': False, 'full_model_service_claim': False,
}
TOOLS = {
    'mxfp4': 'tools/qualify_mxfp4_graph.py',
    'nvfp4': 'tools/qualify_native_kv_p1.py',
}
EXTRAS = {
    'mxfp4': (
        'xllm/python/attention/mxfp4_reference.py',
        'xllm/python/attention/mxfp4_writer.py',
        'xllm/python/attention/mxfp4_flashinfer.py',
        'xllm/python/attention/mxfp4_graph_sources.py',
        'xllm/python/attention/mxfp4_decode_plan.py',
    ),
    'nvfp4': (
        'xllm/python/attention/flashinfer_metadata.py',
        'xllm/python/attention/nvfp4_writer.py',
    ),
}
MX_GATES = ('graph_ready', 'non_split', 'padding_mask_initialized',
            'dynamic_metadata_reference', 'changed_input_replay',
            'prefix_page_copy', 'fast_plan_equal')
NV_GATES = ('nv_packed_equal', 'nv_scale_equal', 'nv_changed_input_graph_equal',
            'invalid_scale_rejected', 'bf16_host_planner_equal',
            'fp8_host_planner_equal', 'nv_host_planner_equal',
            'public_scale_d2h_observed', 'validated_scale_d2h_absent')


def source_files(arm: str) -> tuple[str, ...]:
    return (*SOURCE_FILES, *EXTRAS[arm], TOOLS[arm])


def validate_request(request: dict, approval: dict) -> None:
    from .native_jit_reuse import validate as validate_jit
    from .python_overlay_compatibility import validate_overlay_request
    from .upstream_native import validate_request as validate_original

    arms = request.get('arms')
    arm = arms[0] if isinstance(arms, list) and len(arms) == 1 else None
    phase = request.get('p1_phase')
    hashes = request.get('backend_source_sha256', {})
    if (arm not in TOOLS or phase not in ('pilot', 'main')
            or request.get('workload') != {'kind': KIND, 'scope': SCOPE}
            or request.get('p1_extension') != EXTENSION
            or request.get('p1_protocol') != PROTOCOL
            or request.get('server_binary_execution') != 'none'
            or request.get('max_wall_seconds') != 3600
            or request.get('native_configuration') is not None
            or request.get('operators') != {arm: {'planner': 'frozen_P0_P1', 'decode': 'frozen_P0_P1'}}
            or set(hashes) != set(source_files(arm))
            or any(not isinstance(v, str) or re.fullmatch('[0-9a-f]{64}', v) is None for v in hashes.values())
            or request.get('probe_driver_sha256') != hashes.get(TOOLS[arm])
            or request.get('python_overlay_binary_compatibility') is None):
        raise ApprovalError('P0/P1 qualification requires its exact finite source/protocol/geometry contract')
    validate_overlay_request(request)
    validate_jit(request)
    reference = request.get('p1_pilot_reference')
    if phase == 'main':
        if (not isinstance(reference, dict) or set(reference) != {'run_id', 'result_sha256'}
                or re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,127}', reference.get('run_id', '')) is None
                or reference['run_id'] == request['run_id']
                or re.fullmatch('[0-9a-f]{64}', reference.get('result_sha256', '')) is None):
            raise ApprovalError('Main P0/P1 measurement requires its successful immutable pilot')
    elif reference is not None:
        raise ApprovalError('Pilot cannot claim another run as its own qualification')
    compiled = request.get('compiled_backend_source_sha256')
    if (not isinstance(compiled, dict)
            or set(compiled) != {name for name in SOURCE_FILES if not name.endswith('.py')}
            or any(compiled[name] != hashes[name] for name in compiled)):
        raise ApprovalError('Python-only P0/P1 requires the unchanged compiled source identities')
    common = dict(request)
    common.update(
        workload={'kind': OPERATOR_KIND, 'scope': 'three_frozen_native_operator_candidates_no_model'},
        arms=['fp8_e4m3'],
        operators={'fp8_e4m3': {'planner': 'frozen_three_candidates', 'decode': 'frozen_three_candidates'}},
        python_source_commit=request['server_source_commit'],
        backend_source_sha256={name: hashes[name] for name in SOURCE_FILES},
    )
    validate_original(common, approval)


def operator_acceptance_passed(value: dict, arm: str) -> bool:
    if (value.get('schema') != 'kv_p0_p1_operator_acceptance_v1'
            or value.get('phase') not in ('pilot', 'main')
            or value.get('completed') is not True or value.get('pass') is not True
            or value.get('source_checked') is not True or value.get('model_loaded') is not False
            or value.get('protocol') != PROTOCOL
            or re.fullmatch('[0-9a-f]{40}', value.get('runtime_source_commit', '')) is None
            or any(re.fullmatch('[0-9a-f]{64}', value.get(k, '')) is None
                   for k in ('tool_sha256', 'tool_receipt_sha256'))):
        return False
    gates = value.get('gates', {})
    formats = value.get('qualified_formats')
    expected = MX_GATES if arm == 'mxfp4' else NV_GATES
    return (formats == (['mxfp4'] if arm == 'mxfp4' else ['bf16', 'fp8_e4m3', 'nvfp4'])
            and arm in formats and all(gates.get(key) is True for key in expected))


def _verify_pilot(request: dict, output: Path) -> None:
    if request['p1_phase'] != 'main':
        return
    from .next_stage_deployment import _completed_reference
    previous, old = _completed_reference(request, request['p1_pilot_reference'])
    keys = ('workload', 'p1_extension', 'p1_protocol', 'python_source_commit',
            'backend_source_sha256', 'probe_driver_sha256', 'image_id', 'gpu_uuids',
            'arms', 'python_overlay_binary_compatibility', 'native_jit_source_manifest')
    if old.get('p1_phase') != 'pilot' or any(old.get(key) != request.get(key) for key in keys):
        raise ValueError('Main must use its exact source/protocol/GPU-qualified pilot')
    arm = request['arms'][0]
    value = json.loads((previous/'output'/arm/'operators/acceptance.json').read_bytes())
    if value.get('phase') != 'pilot' or not operator_acceptance_passed(value, arm):
        raise ValueError('Pilot did not pass the actual required P0/P1 gates')
    write_new_json(output/'pilot_reuse.json', dict(request['p1_pilot_reference'],
                   actual_completed_released_pilot=True, source_protocol_GPU_checked=True))


def run(ticket: dict, root: Path, output: Path, arm: str, env: dict) -> dict:
    from .nextgen import native_python_environment
    request = ticket['request']
    _verify_pilot(request, output)
    directory = output/'operators'
    directory.mkdir()
    tool = Path(request['python_source_repo'])/TOOLS[arm]
    if hashlib.sha256(tool.read_bytes()).hexdigest() != request['probe_driver_sha256']:
        raise ValueError('P0/P1 tool identity changed')
    env = native_python_environment(request, {k: v for k, v in env.items() if not k.startswith('XLLM_')})
    env['PYTHONPATH'] = request['python_source_repo']+':'+env.get('PYTHONPATH', '')
    receipt = directory/'tool_result.json'
    command = [sys.executable, '-B', str(tool), '--execute', '--phase', request['p1_phase'],
               '--output', str(receipt)]
    if arm == 'mxfp4':
        command.append('--fast-plan')
    write_new_json(output/'operator_command.json', command)
    with (output/'operator.log').open('x') as log:
        subprocess.run(command, env=env, cwd=request['python_source_repo'], stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=request['max_wall_seconds']-120)
    value = json.loads(receipt.read_bytes())
    gates = value.get('gates', {})
    required = MX_GATES if arm == 'mxfp4' else NV_GATES
    if (value.get('schema') != ('kv_p0_p1_operator_acceptance_v1' if arm == 'mxfp4' else 'native_kv_p1_qualification_v1')
            or value.get('phase') != request['p1_phase']
            or not isinstance(gates, dict) or any(gates.get(name) is not True for name in required)
            or value.get('passed', value.get('pass')) is not True):
        raise ValueError('Tool did not establish every required P0/P1 correctness and mechanism gate')
    acceptance = {
        'schema': 'kv_p0_p1_operator_acceptance_v1', 'phase': request['p1_phase'],
        'completed': True, 'pass': True, 'source_checked': True, 'model_loaded': False,
        'runtime_source_commit': request['python_source_commit'],
        'tool_sha256': request['probe_driver_sha256'],
        'tool_receipt_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest(),
        'qualified_formats': ['mxfp4'] if arm == 'mxfp4' else ['bf16', 'fp8_e4m3', 'nvfp4'],
        'gates': gates, 'protocol': PROTOCOL,
        'service_performance_claim': False,
    }
    if not operator_acceptance_passed(acceptance, arm):
        raise ValueError('Incomplete P0/P1 acceptance receipt')
    write_new_json(directory/'acceptance.json', acceptance)
    return {'classification': 'P0_P1_operator_qualification_only', 'acceptance': acceptance,
            'model_loaded': False, 'performance_claim': False, 'quality_claim': False}
