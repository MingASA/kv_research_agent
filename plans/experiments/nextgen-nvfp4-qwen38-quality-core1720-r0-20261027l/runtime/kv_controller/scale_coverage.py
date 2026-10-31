# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""A single predeclared scale-coverage candidate bound to actual long-QA evidence."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re

from .durable import digest, write_new_json

METHOD = 'long_document_QA_heldout_minimum_power2_coverage_v1'
RUN = 'prod-calib-bf16xqa-longqa128-r0-20261027a'
SCALES = {'key':.125, 'value':.125}
BASE_SCALES = {'key':.125, 'value':.0625}
TOOLS_CHANGED = {'tools/check_native_fa3_operator.py', 'tools/run_qwen25_joint_kv_quality.py'}


def validate_contract(request: dict) -> bool:
    item = request.get('workload', {}).get('scale_calibration')
    if item is None:
        return False
    workload = request['workload']
    if (request.get('schema') != 'kv-production-run-v1'
            or workload.get('kind') != 'native_quality_xqa_production_batch'
            or workload.get('cohort') != 'full_production_batch' or workload.get('samples_per_arm') != 1719
            or request.get('arms') != ['fp8_e4m3'] or request.get('concurrency') != 128
            or request.get('fp8_scales') != SCALES or item.get('method') != METHOD
            or item.get('calibration_run_id') != RUN
            or request.get('max_wall_seconds', 99999) > 3600
            or not 0 < request.get('max_output_gib', 0) <= 5):
        raise ValueError('Coverage calibration is only the unique full1719 single-FP8 C128 V=.125 candidate')
    for field in ('calibration_audit_sha256', 'calibration_terminal_sha256', 'proposal_sha256',
                  'reference_audit_sha256'):
        if re.fullmatch(r'[0-9a-f]{64}', item.get(field, '')) is None:
            raise ValueError('Unpinned scale calibration prerequisite: ' + field)
    for key, name in (('calibration_audit_path','longcontext_range_independent_audit.json'),
                      ('proposal_path','scale_coverage_candidate.json'),
                      ('reference_audit_path','existing_modern_reference_audit.json')):
        if item.get(key) != request['remote_output'] + '/input/' + name:
            raise ValueError('Scale calibration prerequisite escaped the frozen input directory')
    old = item.get('baseline_backend_source_sha256', {})
    new = request.get('backend_source_sha256', {})
    if (not old or set(old) != set(new)
            or any(re.fullmatch(r'[0-9a-f]{64}', v or '') is None for v in old.values())
            or {name for name in old if old[name] != new[name]} != TOOLS_CHANGED):
        raise ValueError('Scale coverage must retain every inference/scoring source; only gate/plan tools change')
    return True


def validate_prerequisites(request: dict, output: Path | None = None,
                           input_root: Path | None = None) -> dict:
    if not validate_contract(request):
        raise ValueError('Missing explicit scale calibration contract')
    item = request['workload']['scale_calibration']
    values = {}
    for key, sha_key in (('calibration_audit_path','calibration_audit_sha256'),
                         ('proposal_path','proposal_sha256'), ('reference_audit_path','reference_audit_sha256')):
        path = Path(item[key]) if input_root is None else input_root / Path(item[key]).name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 4*2**20:
            raise ValueError('Invalid frozen scale prerequisite file')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item[sha_key]:
            raise ValueError('Frozen scale prerequisite SHA differs: ' + key)
        values[key] = json.loads(raw)
    audit, proposal, reference = [values[k] for k in
        ('calibration_audit_path', 'proposal_path', 'reference_audit_path')]
    if (audit.get('classification') != 'confirmed_raw_heldout_BF16_KV_distribution_and_AIPerf_integrity_not_FP8_quality'
            or audit.get('run_id') != RUN or audit.get('original_root_status') != 'COMPLETE'
            or audit.get('content_scope') != 'heldout_original_LongBench_long_document_QA'
            or audit.get('terminal_result_sha256') != item['calibration_terminal_sha256']
            or audit.get('actual_original_requests') != 128
            or audit.get('binary_sha256') != request['binary_sha256']
            or audit.get('model_config_sha256') != request['model_config_sha256']
            or proposal.get('calibration_audit_sha256') != item['calibration_audit_sha256']
            or proposal.get('calibration_terminal_sha256') != item['calibration_terminal_sha256']
            or proposal.get('old_scales') != BASE_SCALES or proposal.get('candidate_scales') != SCALES
            or proposal.get('actual_clipped_count') != {'key':0, 'value':68}
            or proposal.get('actual_absmax') != {'key':45.25, 'value':37.0}
            or proposal.get('quality_labels_used') is not False
            or proposal.get('calibration_exact_overlap_with_old_LongBench400') != 0
            or proposal.get('candidate_quality_accepted') is not False):
        raise ValueError('Actual long-document clipping does not bind this answer-independent coverage candidate')
    geometry = reference.get('production_configuration', {})
    if (reference.get('classification') != 'confirmed_full_production_batch_prediction_integrity_and_rescoring'
            or reference.get('run_id') != item.get('reference_run_id')
            or reference.get('audited_arms') != ['bf16','fp8_e4m3']
            or reference.get('samples_per_arm') != 1719 or reference.get('root_complete') is not True
            or reference.get('gpu_release_verified') is not True
            or any(geometry.get(k) != request[k] for k in ('binary_sha256','model_config_sha256',
                'image_id','operators','service','graph_mode','concurrency'))
            or geometry.get('fp8_scales') != BASE_SCALES
            or any(reference['audited_source_files_sha256'].get(name) != value
                   for name,value in request['backend_source_sha256'].items() if name.startswith('xllm/'))):
        raise ValueError('Retained modern reference or unchanged inference-source identity differs')
    rows = audit['rows']
    clipped = {kind:sum(r['clipped'] for r in rows if r['kind'] == kind) for kind in BASE_SCALES}
    maxima = {kind:max(r['absmax'] for r in rows if r['kind'] == kind) for kind in BASE_SCALES}
    if clipped != proposal['actual_clipped_count'] or maxima != proposal['actual_absmax']:
        raise ValueError('Raw counter-derived coverage proposal differs')
    result = {'classification':'confirmed_frozen_scale_coverage_prerequisites_only_no_new_model_quality',
        'calibration':item, 'actual_clipped_count':clipped, 'actual_absmax':maxima,
        'selected_scales':SCALES, 'BF16_reference_predictions_repeated':False,
        'inference_source_unchanged':True, 'scoring_source_unchanged':True,
        'prerequisite_payload_digest':digest(values), 'model_quality_validated':False}
    if output is not None:
        write_new_json(output / 'scale_coverage_prerequisite.json', result)
    return result
