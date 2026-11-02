"""Native C qualification contracts on the frozen official/build-fix baseline."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

from .approval import ApprovalError
from .durable import write_new_json
from .upstream_qualification import BASELINE_COMMIT, OFFICIAL_COMMIT, MODEL, SERVICE
from .upstream_qualification import SOURCE_FILES as BASE_SOURCE_FILES

OPERATOR_KIND = 'upstream_native_C_operator_qualification'
MODEL_KIND = 'upstream_native_C_model_qualification'
B_KIND = 'upstream_common_path_BF16_regression'
DIAG_KIND = 'upstream_FP8_direct_library_attribution'
VMM_KIND = 'upstream_native_VMM_compatibility'
SOURCE_FILES = tuple(BASE_SOURCE_FILES) + (
    'xllm/python/attention/flashinfer_native_kv.py',
    'xllm/python/attention/native_kv_layout.py',
    'xllm/python/attention/native_kv_writer.py',
    'xllm/python/attention/query_quantization.py',
    'xllm/python/attention/xqa_decode.py',
    'xllm/core/framework/config/kv_cache_config.cpp',
    'xllm/core/framework/config/kv_cache_config.h',
    'xllm/core/kv_cache/layout/kv_cache_capacity.h',
    'xllm/core/kv_cache/layout/native_kv_cache_format.h',
    'xllm/core/kv_cache/layout/virtual_kv_cache_page_layout.h',
    'xllm/core/kv_cache/layout/vmm_kv_cache_layout.cpp',
    'xllm/core/kv_cache/storage/kv_cache_utils.h',
)
CONFIGURATIONS = {
    'fp8_e4m3': {'dtype': 'fp8_e4m3', 'key_scale': .125, 'value_scale': .0625,
                 'prefill': 'fa3', 'decode': 'xqa'},
    'nvfp4': {'dtype': 'nvfp4', 'key_scale': 1., 'value_scale': 1.,
              'prefill': 'fa2', 'decode': 'flashinfer'},
}


def validate_request(request: dict, approval: dict) -> None:
    from .policy import _path

    def require(value: bool, reason: str) -> None:
        if not value:
            raise ApprovalError(reason)

    kind = request.get('workload', {}).get('kind')
    require(request.get('schema') == 'kv-nextgen-run-v1' and kind in (OPERATOR_KIND, MODEL_KIND, B_KIND, DIAG_KIND, VMM_KIND),
            'Unknown latest native C qualification contract')
    require(request.get('upstream_commit') == OFFICIAL_COMMIT
            and request.get('native_base_commit') == BASELINE_COMMIT,
            'Native C must share the frozen official baseline and common CUDA guard')
    require(request.get('model_id') == MODEL and request.get('model_path') == approval['read_only_model_path']
            and request.get('image_id') == approval['approved_image_id'], 'Unpinned model or image')
    require(len(request.get('gpu_uuids', [])) == 1, 'Native qualification uses one admitted GPU')
    require(type(request.get('max_wall_seconds')) is int and 1800 <= request['max_wall_seconds'] <= (5400 if kind == VMM_KIND else 3600),
            'Native qualification needs a bounded startup/operator/cleanup window')
    for name in ('source_repo', 'python_source_repo', 'server_binary', 'dataset_path'):
        require(_path(request.get(name)).startswith(approval['approved_workspace'] + '/'), 'Native asset escaped BASE')
    require(request['python_source_repo'] == request['remote_output'] + '/source_repo'
            and request['dataset_path'] == request['remote_output'] + '/input/validation.json',
            'Native checkout/input escaped its new owned run')
    require(request.get('python_source_commit') == request.get('server_source_commit')
            and re.fullmatch('[0-9a-f]{40}', request.get('server_source_commit', '')) is not None,
            'Native C must use the same compiled and Python source commit')
    for name in ('binary_sha256', 'dataset_sha256', 'model_config_sha256', 'zeromq_library_sha256'):
        require(re.fullmatch('[0-9a-f]{64}', request.get(name, '')) is not None, 'Missing native identity: ' + name)
    hashes = request.get('backend_source_sha256', {})
    require(set(hashes) == set(SOURCE_FILES)
            and all(re.fullmatch('[0-9a-f]{64}', value) is not None for value in hashes.values()),
            'Missing native allocation/attention/Graph identities')
    require(request.get('service') == SERVICE and request.get('graph_mode') == 'cudagraphs'
            and request.get('prefix_cache') is True and request.get('cache_allocator') is None,
            'Native C production configuration differs from A')
    require(request.get('runtime_dependency_build_id') == 'upstream-ac-261274e5-20261102b'
            and request.get('runtime_library_directory') == approval['approved_workspace']
                + '/agent_work/builds/upstream-ac-261274e5-20261102b/zeromq-prefix/lib',
            'Native C must reuse the exact same private runtime library as A')
    require(request.get('startup_recovery') is None and request.get('environment_recovery') is None,
            'Native C cannot silently reuse an old experimental retry')
    if kind == VMM_KIND:
        from .upstream_vmm import PROFILES
        require(request.get('arms') == ['fp8_e4m3'] and request.get('native_configuration') is None
                and request.get('server_binary_execution') == 'required'
                and request.get('max_wall_seconds') == 5400 and request.get('VMM_profiles') == list(PROFILES)
                and request['workload'].get('scope') == 'BF16_FP8_NVFP4_elastic_VMM_model_prefix_Graph_no_benchmark'
                and request.get('operators') == {'fp8_e4m3': {'planner':'frozen_three_VMM_profiles','decode':'flashinfer_or_official_auto'}},
                'VMM compatibility must retain the frozen three untimed model profiles')
    elif kind == DIAG_KIND:
        require(request.get('arms') == ['fp8_e4m3'] and request.get('native_configuration') is None
                and request.get('server_binary_execution') == 'none'
                and request['workload'].get('scope') == 'independent_quant_direct_FA3_XQA_conformance_preserve_dense_failure'
                and request.get('operators') == {'fp8_e4m3': {'planner':'paired_direct_FlashInfer','decode':'paired_direct_XQA'}},
                'Diagnosis is bounded to the observed FP8 failure and independent library/quantization checks')
    elif kind == OPERATOR_KIND:
        require(request.get('arms') == ['fp8_e4m3'] and request.get('native_configuration') is None
                and request.get('operators') == {'fp8_e4m3': {'planner': 'frozen_three_candidates', 'decode': 'frozen_three_candidates'}}
                and request.get('server_binary_execution') == 'none'
                and request['workload'].get('scope') == 'three_frozen_native_operator_candidates_no_model',
                'Operator qualification must explicitly exclude model/service claims')
    elif kind == B_KIND:
        require(request.get('arms') == ['bf16'] and request.get('native_configuration') is None
                and request.get('operators') == {'bf16': {'planner': 'official_auto', 'decode': 'official_auto'}}
                and request.get('server_binary_execution') == 'required'
                and request['workload'].get('scope') == 'minimal_common_path_BF16_prefix_graph_regression',
                'Common-path regression must keep normal BF16 and exclude KV quantization')
    else:
        arm = request.get('arms', [None])[0]
        require(request.get('arms') == [arm] and arm in CONFIGURATIONS
                and request.get('native_configuration') == CONFIGURATIONS[arm]
                and request.get('server_binary_execution') == 'required'
                and request.get('startup_seconds') == 1500
                and request['workload'].get('scope') == 'native_model_prefix_graph_qualification',
                'Native model qualification must use one frozen, explicit supported format')
        require(request.get('operators') == {arm: {'planner': CONFIGURATIONS[arm]['prefill'],
                                                  'decode': CONFIGURATIONS[arm]['decode']}},
                'Native operator labels must match the actual server configuration')


def run(ticket: dict, root: Path, output: Path, arm: str, env: dict) -> dict:
    from .nextgen import native_python_environment
    request = ticket['request']
    if request['workload']['kind'] == VMM_KIND:
        from .upstream_vmm import run as qualify_vmm
        return qualify_vmm(ticket, root, output, arm, env)
    if request['workload']['kind'] in (MODEL_KIND, B_KIND):
        from .upstream_qualification import run as qualify_model
        return qualify_model(ticket, root, output, arm, env)
    env = native_python_environment(request, {k: v for k, v in env.items() if not k.startswith('XLLM_')})
    env['PYTHONPATH'] = request['python_source_repo'] + ':' + env.get('PYTHONPATH', '')
    diagnosis = request['workload']['kind'] == DIAG_KIND
    if diagnosis:
        import hashlib
        reference = request['operator_diagnosis_reference']
        prior = Path(request['approved_workspace'])/'agent_work/runs'/reference['run_id']
        raw = (prior/'result.json').read_bytes()
        original = json.loads(raw)
        if (hashlib.sha256(raw).hexdigest() != reference['result_sha256']
                or original.get('status') != 'COMPLETE' or original.get('gpu_release_verified') is not True
                or original.get('live_owned_containers') != []):
            raise ValueError('Diagnosis must reference the actual released original candidate campaign')
    module = '.upstream_fp8_diagnosis' if diagnosis else '.upstream_operator'
    command = [sys.executable, '-B', '-m', __package__ + module, '--output', str(output / ('diagnosis' if diagnosis else 'operators'))]
    write_new_json(output / 'operator_command.json', command)
    with (output / 'operator.log').open('x') as log:
        subprocess.run(command, cwd=request['python_source_repo'], env=env, stdout=log,
                       stderr=subprocess.STDOUT, timeout=2400, check=True)
    if diagnosis:
        result = json.loads((output/'diagnosis/diagnosis.json').read_bytes())
        return {'classification':'FP8_direct_library_attribution_preserved_original_dense_tolerance',
                'diagnosis':result,'model_loaded':False,'xLLM_server_binary_executed':False,
                'performance_claim':False,'quality_benchmark_claim':False}
    result = json.loads((output / 'operators/acceptance.json').read_bytes())
    if len(result.get('candidates', [])) != 3:
        raise ValueError('Native operator candidate coverage is incomplete')
    return {'classification': 'native_C_real_GPU_operators_not_model_or_service',
            'operator_acceptance': result, 'xLLM_server_binary_executed': False,
            'all_candidates_pass': result['all_candidates_pass'],
            'model_loaded': False, 'performance_claim': False, 'quality_benchmark_claim': False}
