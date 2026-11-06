"""Same-base MXFP4/FP8 deployment contracts, separate from the closed campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .approval import ApprovalError
from .upstream_native import SOURCE_FILES, CONFIGURATIONS, MODEL_KIND as OLD_MODEL_KIND


MODEL_KIND = 'upstream_next_stage_model_qualification'
PROFILES = ('ocp_mxfp4_20261105', 'fp8_optimization_20261105')
MX_CONFIGURATION = {'dtype':'mxfp4', 'key_scale':1., 'value_scale':1.,
                    'prefill':'fa2', 'decode':'flashinfer'}
LEGACY_C_COMMIT = 'da7c6446ff4361c8060633b04d6bfb77e6f737ff'
LEGACY_C_BINARY = '74668d5e5d73a0dc23f70679350851b10c11c99e42c6961f511dee33ab7f60e9'


def validate_pair_sources(reference: dict, candidate: dict, native_reference: str | None = None) -> None:
    """Keep official A comparisons distinct from frozen previous-format pairs."""
    from .upstream_qualification import OFFICIAL_COMMIT, BASELINE_COMMIT

    if (reference.get('upstream_commit') != OFFICIAL_COMMIT
            or candidate.get('upstream_commit') != OFFICIAL_COMMIT
            or candidate.get('native_base_commit') != BASELINE_COMMIT
            or not (candidate.get('arms') in (['fp8_e4m3'],['nvfp4'])
                    or candidate.get('arms') == ['mxfp4'] and candidate.get('next_stage_profile') == PROFILES[0])):
        raise ValueError('Comparison requires the same frozen official/build-fix base')
    if candidate.get('next_stage_profile') in PROFILES:
        expected = 'mxfp4' if candidate['next_stage_profile'] == PROFILES[0] else 'fp8_e4m3'
        if candidate['arms'] != [expected]:
            raise ValueError('Candidate profile and actual format disagree')
    if native_reference is None:
        if (reference.get('arms') != ['bf16'] or reference.get('native_configuration') is not None
                or reference.get('server_source_commit') != BASELINE_COMMIT
                or reference.get('python_source_commit') != BASELINE_COMMIT):
            raise ValueError('Default comparison requires the actual official BF16 A')
    elif (native_reference not in CONFIGURATIONS
            or reference.get('arms') != [native_reference]
            or reference.get('native_configuration') != CONFIGURATIONS[native_reference]
            or reference.get('native_base_commit') != BASELINE_COMMIT
            or reference.get('server_source_commit') != LEGACY_C_COMMIT
            or reference.get('python_source_commit') != LEGACY_C_COMMIT
            or reference.get('binary_sha256') != LEGACY_C_BINARY
            or candidate.get('next_stage_profile') not in PROFILES):
        raise ValueError('Native reference must be the original measured format and binary, with a new-stage candidate')


def validate_deployment(request: dict, approval: dict) -> None:
    """Validate new storage separately; retain every original resource constraint."""
    from .upstream_native import validate_request as validate_original
    from .native_jit_reuse import validate as validate_jit
    validate_jit(request)

    profile = request.get('next_stage_profile')
    mxfp4 = profile == PROFILES[0]
    retained=request.get('fp8_component_selection')=='retained_query_and_range_original_mixed'
    if request.get('fp8_component_selection') is not None and (not retained or mxfp4):
        raise ApprovalError('Unknown retained FP8 component selection')
    extras = (('mxfp4_reference.py','mxfp4_writer.py','mxfp4_flashinfer.py') if mxfp4
              else () if retained else ('mixed_native_kv.py',))
    expected = set(SOURCE_FILES) | {'xllm/python/attention/'+name for name in extras}
    configuration = request.get('native_configuration')
    valid_configurations = ([MX_CONFIGURATION] if mxfp4 else
                            [CONFIGURATIONS['fp8_e4m3'], dict(CONFIGURATIONS['fp8_e4m3'], value_scale=.5)])
    hashes = request.get('backend_source_sha256', {})
    arm = 'mxfp4' if mxfp4 else 'fp8_e4m3'
    if (profile not in PROFILES or configuration not in valid_configurations
            or request.get('arms') != [arm] or set(hashes) != expected
            or any(not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None for value in hashes.values())
            or request.get('operators') != {arm:{'planner':configuration['prefill'],'decode':configuration['decode']}}
            or re.fullmatch('[0-9a-f]{40}',request.get('python_source_commit','')) is None
            or request.get('numeric_diagnostics') is not None
            or request.get('compiled_backend_source_sha256') is not None):
        raise ApprovalError('Unknown next-stage format, sources or diagnostic contamination')
    if mxfp4 and request.get('python_source_commit') != request.get('server_source_commit'):
        raise ApprovalError('MXFP4 needs its actual newly compiled allocator/configuration binary')
    # Explicit projection reuses the proven machine/runtime/path/production
    # envelope. Original validators still reject MXFP4 and Python-only branches.
    # CLI checks actual compiled C++/header hashes; worker checks the new bundle.
    common = dict(request, arms=['fp8_e4m3'], native_configuration=CONFIGURATIONS['fp8_e4m3'],
                  operators={'fp8_e4m3':{'planner':'fa3','decode':'xqa'}},
                  python_source_commit=request['server_source_commit'],
                  backend_source_sha256={name:hashes[name] for name in SOURCE_FILES},
                  workload={'kind':OLD_MODEL_KIND,'scope':'native_model_prefix_graph_qualification'})
    validate_original(common,approval)
    reference = request.get('operator_qualification_reference', {})
    if (re.fullmatch('[A-Za-z0-9_-]{1,128}',reference.get('run_id','')) is None
            or re.fullmatch('[0-9a-f]{64}',reference.get('result_sha256','')) is None):
        raise ApprovalError('Model deployment requires a real source-matched operator qualification')
    if not mxfp4 and configuration['value_scale'] == .5:
        reference = request.get('numeric_calibration_reference', {})
        if (re.fullmatch('[A-Za-z0-9_-]{1,128}',reference.get('run_id','')) is None
                or re.fullmatch('[0-9a-f]{64}',reference.get('result_sha256','')) is None):
            raise ApprovalError('The V-scale candidate requires independent numeric calibration')


def _completed_reference(request: dict, reference: dict) -> tuple[Path, dict]:
    prior = Path(request['approved_workspace'])/'agent_work/runs'/reference['run_id']
    raw = (prior/'result.json').read_bytes()
    result = json.loads(raw)
    if (hashlib.sha256(raw).hexdigest() != reference['result_sha256']
            or result.get('status') != 'COMPLETE' or result.get('gpu_release_verified') is not True
            or result.get('live_owned_containers') != []):
        raise ValueError('Prerequisite is not the actual completed and released original run')
    return prior, json.loads((prior/'ticket.json').read_bytes())['request']


def operator_acceptance_passed(acceptance: dict, arm: str) -> bool:
    if arm=='fp8_e4m3' and acceptance.get('schema')=='FP8_retained_candidate_operator_v1':
        rows=acceptance.get('scale_rows',[])
        return (acceptance.get('query_gate') is True and acceptance.get('query_component_reused') is True
                and acceptance.get('writer_two_scales_bitwise') is True
                and acceptance.get('independent_snapshot_oracle') is True and acceptance.get('finite_outputs') is True
                and acceptance.get('normal_mixed_all_FA3') is True and acceptance.get('pure_XQA') is True
                and acceptance.get('rejected_mixed_split_present') is False
                and len(rows)==2 and [r.get('value_scale') for r in rows]==[.0625,.5]
                and all(r.get('writer_Torch_bitwise') is True and r.get('mixed_all_FA3_direct_pass') is True
                        and r.get('pure_XQA_direct_pass') is True and r.get('GDN_unchanged') is True
                        and r.get('direct_uses_independent_verified_snapshot') is True
                        and r.get('finite_outputs') is True for r in rows))
    mxfp4 = arm == 'mxfp4'
    return (acceptance.get('schema') == 'MXFP4_SM90_operator_gate_v1'
              and acceptance.get('writer_CPU_bitwise') is True
              and acceptance.get('whole_page_copy') is True
              and acceptance.get('results',{}).get('prefill',{}).get('pass') is True
              and acceptance.get('results',{}).get('decode',{}).get('pass') is True
              and acceptance.get('results',{}).get('Graph') == {'changed_inputs':True,'bitwise_equal_direct':True}) if mxfp4 else (
                  acceptance.get('schema') == 'FP8_finite_optimization_operator_v1'
                  and acceptance.get('query_gate') is True and acceptance.get('mixed_gate') is True)


def run(ticket: dict, root: Path, output: Path, arm: str, env: dict) -> dict:
    from .upstream_qualification import run as qualify

    request = ticket['request']
    prior, old = _completed_reference(request,request['operator_qualification_reference'])
    if (old['python_source_commit'] != request['python_source_commit']
            or old['backend_source_sha256'] != request['backend_source_sha256']
            or any(old[name] != request[name] for name in ('model_id','model_config_sha256','image_id'))):
        raise ValueError('Model and operator sources differ')
    acceptance = json.loads((prior/'output'/arm/'operators/acceptance.json').read_bytes())
    if not operator_acceptance_passed(acceptance,arm):
        raise ValueError('Operator correctness/performance gate did not pass')
    if request['native_configuration']['value_scale'] == .5:
        prior, old = _completed_reference(request,request['numeric_calibration_reference'])
        if (old.get('calibration_sha256') != 'c241e7e8c58d3be456de39ea080a33a2a2e52384221859c4b9b2f7ac3b11885e'
                or any(old[name] != request[name] for name in ('model_id','model_config_sha256','image_id','service','upstream_commit','native_base_commit'))):
            raise ValueError('Numeric calibration cohort changed')
        summary = json.loads((prior/'output/bf16/calibration_acceptance.json').read_bytes())
        if summary.get('candidate_numerical_gate') is not True:
            raise ValueError('Independent numeric scale candidate did not qualify')
    return qualify(ticket,root,output,arm,env)
