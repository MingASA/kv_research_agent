# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Owned-process operator admission for one production-directed prefill repair."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from .durable import write_new_json


def run_operator(ticket: dict, output: Path, arm: str, env: dict) -> dict:
    request = ticket['request']
    if arm != 'fp8_e4m3' or request['workload']['kind'] != 'native_prefill_workspace_operator':
        raise ValueError('Operator-only entry is restricted to the explicit prefill repair')
    operator = request['operators'][arm]
    env.update(XLLM_FLASHINFER_NATIVE_PLANNER='fa3', XLLM_FLASHINFER_NATIVE_DECODE='xqa',
               XLLM_FP8_PREFILL_WORKSPACE_MIB=str(operator['prefill_workspace_mib']))
    manifest = json.loads(Path(request['dataset_path']).read_text())
    if (manifest['geometry'] != request['workload']['geometry']
            or manifest['recorded_source_sha256'] != request['workload']['source_content_sha256']
            or manifest['method'] != request['workload']['method']):
        raise ValueError('Recorded geometry input differs from the declared diagnostic')
    write_new_json(output / 'operator_environment.json', {k: v for k, v in env.items() if k.startswith('XLLM_')})
    command = [sys.executable, '-B', request['python_source_repo'] + '/tools/check_fp8_prefill_workspace.py',
               '--input', request['dataset_path'], '--out', str(output / 'operator_acceptance.json')]
    write_new_json(output / 'operator_command.json', command)
    with (output / 'operator_acceptance.log').open('x') as log:
        subprocess.run(command, env=env, cwd=request['python_source_repo'], stdout=log,
                       stderr=subprocess.STDOUT, timeout=1200, check=True)
    result = json.loads((output / 'operator_acceptance.json').read_text())
    if (result.get('classification') != 'confirmed_operator_correctness_and_recorded_geometry_screen_only'
            or result.get('model_loaded') is not False or result.get('AIPerf_ran') is not False
            or result['native_operator'].get('real_capture_and_replay') is not True
            or result['screen'].get('prepare_and_active_page_cost_included') is not True):
        raise ValueError('Actual operator/Graph/recorded-shape acceptance is incomplete')
    pages = result.get('workspace_page_checks', {})
    if any(pages.get(name) is not True for name in ('shared_page_deduplicated', 'noncontiguous_FP8_strides',
            'inactive_pages_not_materialized', 'multiple_layers_reuse_same_storage', 'next_prefill_replaces_active_set')):
        raise ValueError('Actual workspace page lifecycle/stride/dedup checks failed')
    backend = result['screen']['candidate_backend']
    if (backend.get('prefill_kv_data_type') != 'torch.bfloat16'
            or backend.get('kv_data_type') != 'torch.float8_e4m3fn'
            or backend.get('fp8_views_alias_resident_pages') is not True
            or backend.get('profile_scope') != 'production_xqa_bf16_prefill_workspace'
            or not 1536 * 2**20 <= backend.get('prefill_workspace_reserved_bytes', 0) <= 1540 * 2**20
            or not 0 < backend.get('prefill_workspace_view_bytes', 0) <= 1536 * 2**20):
        raise ValueError('Actual prefill dtype/resident alias/complete workspace footprint differs')
    return {'classification': result['classification'], 'protocol': 'production_prefill_repair_operator_only_v1',
            'model_loaded': False, 'official_aiperf_complete': False, 'operator_only': True,
            'actual_operator_graph_verified': True, 'screen': result['screen'],
            'quality_or_service_result': False}
