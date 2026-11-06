"""Explicit user-requested fast admission; legacy jobs keep their 120s gate."""

from .approval import ApprovalError


FAST_ADMISSION = {
    'policy':'user_5s_idle_takeover_20261106',
    'poll_seconds':5,
    'stable_idle_seconds':0,
    'before_create_fresh_snapshot':True,
    'foreign_gpu_action':'stop_owned_then_reobserve',
}
_KINDS = {'upstream_next_stage_native_operator','upstream_FP8_optimization_operator',
          'upstream_FP8_independent_numeric_calibration','upstream_next_stage_model_qualification'}
_FORMAL = {'upstream_AC_fixed_production_service','upstream_AC_frozen_core_quality1720',
           'upstream_AC_real_native_tools32'}


def idle_parameters(request: dict) -> tuple[int,int]:
    value=request.get('resource_admission')
    if value is None:
        return 120,30
    kind=request.get('workload',{}).get('kind')
    stage=(kind in _KINDS or kind in _FORMAL and request.get('next_stage_profile') in
           ('ocp_mxfp4_20261105','fp8_optimization_20261105'))
    if value != FAST_ADMISSION or not stage or len(request.get('gpu_uuids',[])) != 1:
        raise ApprovalError('Fast admission requires the explicit single-GPU next-stage user policy')
    return 0,5
