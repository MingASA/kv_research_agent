"""Separate next-stage operator admission; retain every shared resource guard."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

from .approval import ApprovalError
from .durable import write_new_json
from .upstream_native import SOURCE_FILES, OPERATOR_KIND


KIND = 'upstream_next_stage_native_operator'
FP8_KIND = 'upstream_FP8_optimization_operator'
CALIB_KIND = 'upstream_FP8_independent_numeric_calibration'
EXTRA_SOURCES = ('xllm/python/attention/mxfp4_reference.py',
                 'xllm/python/attention/mxfp4_writer.py',
                 'xllm/python/attention/mxfp4_flashinfer.py')


def validate_request(request: dict, approval: dict) -> None:
    from .upstream_native import validate_request as validate_base
    if request.get('workload',{}).get('kind') == CALIB_KIND:
        _validate_calibration(request, approval)
        return
    fp8 = request.get('workload',{}).get('kind') == FP8_KIND
    expected = set(SOURCE_FILES) | set(('xllm/python/attention/mixed_native_kv.py',) if fp8 else EXTRA_SOURCES)
    arm = 'fp8_e4m3' if fp8 else 'mxfp4'
    workload = ({'kind':FP8_KIND,'scope':'strided_query_and_mixed_XQA_finite_operator_gate'} if fp8 else
                {'kind':KIND,'scope':'OCP_MXFP4_SM90_D256_bitwise_paged_Graph_gate'})
    operators = ({arm:{'planner':'FA3_query_fusion_and_mixed_split','decode':'XQA'}} if fp8 else
                 {arm:{'planner':'private_pinned_FA2_E8M0','decode':'private_pinned_FA2_E8M0'}})
    hashes = request.get('backend_source_sha256', {})
    compiled = request.get('compiled_backend_source_sha256')
    if compiled is not None and (fp8 or set(compiled) != {name for name in SOURCE_FILES if not name.endswith('.py')}
            or any(not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None for value in compiled.values())):
        raise ApprovalError('Unused compiled baseline needs its separate complete C++/header identities')
    if (set(hashes) != expected or any(re.fullmatch('[0-9a-f]{64}', v) is None for v in hashes.values())
            or request.get('arms') != [arm]
            or request.get('workload') != workload
            or request.get('server_binary_execution') != 'none'
            or re.fullmatch('[0-9a-f]{40}', request.get('python_source_commit','')) is None
            or request.get('operators') != operators):
        raise ApprovalError('Next-stage operator requires the exact source/format/geometry contract')
    # Apply the established baseline/assets/runtime constraints. The binary is
    # explicitly unused; the new Python commit is separately pinned above and
    # verified from its bundle at deployment, not relabeled as compiled code.
    common = dict(request)
    common.update(workload={'kind':OPERATOR_KIND,'scope':'three_frozen_native_operator_candidates_no_model'},
        arms=['fp8_e4m3'], operators={'fp8_e4m3':{'planner':'frozen_three_candidates','decode':'frozen_three_candidates'}},
        python_source_commit=request['server_source_commit'],
        backend_source_sha256={k:hashes[k] for k in SOURCE_FILES})
    validate_base(common, approval)


def _validate_calibration(request: dict, approval: dict) -> None:
    from .upstream_native import validate_request as validate_base, B_KIND

    hashes = request.get('backend_source_sha256', {})
    expected = set(SOURCE_FILES) | {'xllm/python/attention/fp8_numeric_diagnostics.py'}
    if (set(hashes) != expected or any(re.fullmatch('[0-9a-f]{64}', value) is None for value in hashes.values())
            or request.get('workload') != {'kind':CALIB_KIND,'scope':'independent_BF16_QKV_ranges_clipping_query_error_136'}
            or request.get('arms') != ['bf16'] or request.get('server_binary_execution') != 'required'
            or request.get('operators') != {'bf16':{'planner':'official_auto','decode':'official_auto'}}
            or request.get('calibration_count') != 136 or request.get('calibration_concurrency') != 16
            or request.get('calibration_sha256') != 'c241e7e8c58d3be456de39ea080a33a2a2e52384221859c4b9b2f7ac3b11885e'
            or re.fullmatch('[0-9a-f]{40}', request.get('python_source_commit','')) is None):
        raise ApprovalError('Numeric calibration requires the frozen independent same-model BF16 cohort')
    common = dict(request)
    common.update(workload={'kind':B_KIND,'scope':'minimal_common_path_BF16_prefix_graph_regression'},
                  python_source_commit=request['server_source_commit'],
                  backend_source_sha256={name:hashes[name] for name in SOURCE_FILES})
    validate_base(common, approval)


def run(ticket: dict, root: Path, output: Path, arm: str, env: dict) -> dict:
    if ticket['request']['workload']['kind'] == CALIB_KIND:
        from .fp8_numeric_model import run as run_calibration
        return run_calibration(ticket, root, output, arm, env)
    from .nextgen import native_python_environment
    request = ticket['request']
    env = native_python_environment(request, {k:v for k,v in env.items() if not k.startswith('XLLM_')})
    env['PYTHONPATH'] = request['python_source_repo'] + ':' + env.get('PYTHONPATH', '')
    module = '.fp8_optimization_operator' if request['workload']['kind'] == FP8_KIND else '.mxfp4_operator'
    command = [sys.executable, '-B', '-m', __package__+module, '--output', str(output/'operators')]
    if request['workload']['kind'] == FP8_KIND:
        command += ['--baseline-query',str(root/'input/query_reference_20261102.py')]
    write_new_json(output/'operator_command.json', command)
    with (output/'operator.log').open('x') as log:
        subprocess.run(command, env=env, cwd=request['python_source_repo'], stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=2400)
    import json
    return {'classification':'SM90_next_stage_native_operator_only',
            'acceptance':json.loads((output/'operators/acceptance.json').read_bytes()),
            'model_loaded':False,'performance_claim':False,'quality_claim':False}
