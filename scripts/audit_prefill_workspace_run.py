#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Verify a sealed operator-only repair gate and recompute its fixture timings."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.controller.durable import digest, write_new_json


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            value.update(chunk)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(mirror: Path, run_id: str, source: Path, recorded: Path) -> dict:
    snapshot = json.loads((mirror / 'SNAPSHOT.json').read_text())
    for row in snapshot['files']:
        path = mirror / row['relative_path']
        require(not path.is_symlink() and path.resolve().is_relative_to(mirror.resolve())
                and path.stat().st_size == row['size'] and sha(path) == row['sha256'],
                'Original mirrored evidence differs')
    root = mirror / 'support/agent_work/runs' / run_id
    ticket = json.loads((root / 'ticket.json').read_text())
    request = ticket['request']
    terminal = json.loads((root / 'result.json').read_text())
    require(request['run_id'] == run_id and digest(request) == ticket['request_sha256']
            and terminal['request_sha256'] == ticket['request_sha256']
            and terminal['run_id'] == run_id and terminal['status'] == 'COMPLETE'
            and terminal['gpu_release_verified'] is True and terminal['live_owned_containers'] == []
            and json.loads((root / 'COMPLETE').read_text())['result_sha256'] == sha(root / 'result.json'),
            'Missing actual terminal seal or driver release')
    require(request['workload']['kind'] == 'native_prefill_workspace_operator'
            and request['arms'] == ['fp8_e4m3'], 'Expected the explicit operator-only run')
    for name, expected in ticket['assets_sha256'].items():
        require(sha(root / name) == expected, 'Frozen runtime/input asset differs: ' + name)
    for name, expected in request['backend_source_sha256'].items():
        require(sha(source / name) == expected, 'Independent local scientific source differs: ' + name)
    geometry = json.loads((root / 'input/geometry.json').read_text())
    require(sha(root / 'input/geometry.json') == request['dataset_sha256']
            and sha(recorded) == geometry['recorded_source_sha256']
            and sha(recorded) == request['workload']['source_content_sha256'],
            'Recorded production geometry provenance differs')
    recorded_data = json.loads(recorded.read_text())
    window = [w for w in recorded_data['arms']['bf16'] if w['index'] == 2]
    require(len(window) == 1 and geometry['recorded_forward'] == window[0]['forward_batches'][0]
            and geometry['geometry'] == {'sequences': 1, 'query_tokens': 510, 'kv_tokens': 1022, 'layers': 48}
            and geometry['synthetic_tensor_values_not_real_model_activations'] is True,
            'Fixture must retain the declared actual production metadata and synthetic-value scope')
    output = root / 'output/fp8_e4m3'
    measurement = json.loads((output / 'measurement.json').read_text())
    require(measurement['request_sha256'] == ticket['request_sha256']
            and measurement['measurement_success'] is True and measurement['operator_only'] is True
            and measurement['model_loaded'] is False and measurement['official_aiperf_complete'] is False
            and measurement['quality_or_service_result'] is False
            and terminal['arm_results'] == [{'arm': 'fp8_e4m3', 'result_sha256': sha(output / 'measurement.json')}],
            'Operator measurement must not certify a model, AIPerf or serving result')
    for name, expected in measurement['evidence_sha256'].items():
        path = output / name
        require(path.resolve().is_relative_to(output.resolve()) and sha(path) == expected,
                'Sealed operator evidence differs')
    require(not (output / 'server.log').exists() and not (output / 'artifacts').exists(),
            'Unexpected model/service output in an operator-only run')
    result = json.loads((output / 'operator_acceptance.json').read_text())
    require(result['classification'] == 'confirmed_operator_correctness_and_recorded_geometry_screen_only'
            and result['model_loaded'] is False and result['AIPerf_ran'] is False
            and result['production_quality_or_service_certified'] is False
            and result['input_sha256'] == request['dataset_sha256']
            and result['script_sha256'] == sha(source / 'tools/check_fp8_prefill_workspace.py'),
            'Actual acceptance script/input/scope differs')
    native = result['native_operator']
    require(all(native.get(name) is True for name in (
        'real_capture_and_replay', 'partial_page_and_replan', 'noncontiguous_q_checked',
        'padded_graph_bucket_checked', 'fresh_dirty_cache_initialized_to_zero')),
        'Actual numerical/Graph/page initialization checks are incomplete')
    numerical = {}
    for line in (output / 'operator_acceptance.log').read_text().splitlines():
        if 'OPERATOR_REFERENCE ' in line:
            name, data = line.split('OPERATOR_REFERENCE ', 1)[1].split(' ', 1)
            require(name not in numerical, 'Duplicated numerical reference evidence')
            numerical[name] = ast.literal_eval(data)
    require(set(numerical) == {'prefill', 'capture_replay', 'changed_context', 'padded_graph',
                              'restored_finite_padding'}
            and all(math.isfinite(row['relative_l2']) and 0 <= row['relative_l2'] <= .05
                    and math.isfinite(row['max_abs']) and row['max_abs'] >= 0
                    for row in numerical.values()), 'Actual numerical reference logs exceed the fixed gate')
    pages = result['workspace_page_checks']
    require(all(pages.get(name) is True for name in (
        'shared_page_deduplicated', 'noncontiguous_FP8_strides', 'inactive_pages_not_materialized',
        'multiple_layers_reuse_same_storage', 'next_prefill_replaces_active_set')),
        'Actual workspace page lifecycle checks are incomplete')
    screen = result['screen']
    require(screen == measurement['screen'] and screen['geometry'] == geometry['geometry']
            and screen['same_resident_pages_and_queries'] is True and screen['noncontiguous_query'] is True
            and screen['prepare_and_active_page_cost_included'] is True and screen['actual_model_loaded'] is False,
            'Timing scope or actual inputs differ')
    backend = screen['candidate_backend']
    baseline_backend = screen['baseline_backend']
    for observed, planner in ((backend, 'fa3'), (baseline_backend, 'fa2')):
        require(all(observed.get(key) == value for key, value in {
            'planner_backend': planner, 'decode_impl': 'xqa',
            'q_data_type': 'torch.bfloat16', 'q_scale': None,
            'kv_data_type': 'torch.float8_e4m3fn',
            'decode_internal_q_data_type': 'torch.float8_e4m3fn',
            'k_scale': .125, 'v_scale': .0625,
            'fp8_views_alias_resident_pages': True,
        }.items()), 'Actual compared planner/precision/scale/resident alias differs')
    require(backend['profile_scope'] == 'production_xqa_bf16_prefill_workspace'
            and backend['prefill_kv_data_type'] == 'torch.bfloat16'
            and backend['kv_data_type'] == 'torch.float8_e4m3fn'
            and backend['fp8_views_alias_resident_pages'] is True
            and backend['prefill_workspace_reserved_bytes'] == 1539 * 2**20 + 4
            and 0 < backend['prefill_workspace_view_bytes'] <= 1536 * 2**20,
            'Resident alias, prefill precision or full reserved footprint differs')
    timings = {}
    names = ('resident_FP8_FA2', 'resident_FP8_FA3_BF16_workspace')
    for raw_key, median_key, label in (
        ('raw_CUDA_elapsed_ms', 'median_CUDA_elapsed_ms', 'CUDA_elapsed'),
        ('raw_wall_ms', 'median_wall_ms', 'wall')):
        raw = screen[raw_key]
        require(set(raw) == set(names), 'Missing paired timing arm')
        for name in names:
            require(len(raw[name]) == 8 and all(type(x) in (float, int) and math.isfinite(x) and x > 0
                    for x in raw[name]), 'Invalid or incomplete original timing samples')
        medians = {name: statistics.median(raw[name]) for name in names}
        require(medians == screen[median_key], 'Worker timing median differs from original samples')
        baseline, candidate = (medians[name] for name in names)
        timings[label] = {'raw_ms': raw, 'median_ms': medians,
            'candidate_over_baseline': candidate / baseline,
            'candidate_median_reduction_pct': 100 * (1 - candidate / baseline)}
    return {'classification': 'confirmed_operator_evidence_integrity_and_recorded_fixture_cost_only',
        'run_id': run_id, 'source_commit': request['python_source_commit'],
        'terminal_result_sha256': sha(root / 'result.json'),
        'files_SHA_verified': len(snapshot['files']),
        'bytes_SHA_verified': sum(row['size'] for row in snapshot['files']),
        'actual_numerical_Graph_and_workspace_checks_reported_and_sealed': True,
        'logged_numerical_reference_errors': numerical,
        'numerical_execution_independently_repeated': False,
        'workspace_reserved_bytes': backend['prefill_workspace_reserved_bytes'],
        'geometry': geometry['geometry'], 'timings': timings, 'model_loaded': False,
        'AIPerf_ran': False, 'production_service_or_quality_certified': False,
        'limits': ['Synthetic values at one actually recorded shape, not original activations',
            'Includes plan/active-page collection/48 materialize+attention calls; excludes full model',
            'Eight alternating samples describe this fixture, not a production confidence interval',
            'Fullmodel capacity/quality/real service still require separate validation']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mirror', 'source', 'recorded', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    result = audit(args.mirror, args.run_id, args.source, args.recorded)
    write_new_json(args.out, result)
    print(json.dumps({key: result[key] for key in ('classification', 'run_id', 'files_SHA_verified', 'timings')}))


if __name__ == '__main__':
    main()
