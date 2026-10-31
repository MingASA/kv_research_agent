#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Compare independently verified AB/BA serving runs without request-level CIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(audit_paths: list[Path], mirrors: list[Path]) -> dict:
    if len(audit_paths) != 2 or len(mirrors) != 2:
        raise ValueError('This protocol requires exactly one verified AB and one BA run')
    audits, tickets = [], []
    for path, mirror in zip(audit_paths, mirrors):
        audit = json.loads(path.read_text()); audits.append(audit)
        if audit.get('classification') != 'measurement_integrity_verified' or not audit.get('root_complete'):
            raise ValueError('Require clean independently verified serving pairs')
        root = mirror / 'support/agent_work/runs' / audit['run_id']
        ticket = json.loads((root / 'ticket.json').read_text()); tickets.append(ticket)
        if ticket['request']['run_id'] != audit['run_id']:
            raise ValueError('Run identity differs')
    first, second = [ticket['request'] for ticket in tickets]
    if (first['run_id'] == second['run_id'] or first['replicate'] == second['replicate']
            or {tuple(first['arms']), tuple(second['arms'])} != {('bf16', 'fp8_e4m3'), ('fp8_e4m3', 'bf16')}):
        raise ValueError('Two distinct runs with opposite arm order are required')
    for key in ('gpu_uuids', 'binary_sha256', 'server_source_commit', 'python_source_commit',
                'backend_source_sha256', 'dataset_sha256', 'image_id', 'graph_mode', 'prefix_cache',
                'operators', 'service', 'concurrency', 'fp8_scales', 'workload', 'experiment_key'):
        if first[key] != second[key]:
            raise ValueError('Replicates changed a scientific/hardware condition: ' + key)
    if tickets[0]['assets_sha256'] != tickets[1]['assets_sha256']:
        raise ValueError('Frozen inference/input/bundle/runtime assets differ')
    rows = []
    baseline_metrics = audits[0]['sealed_arms']['bf16']['metrics']
    for audit, ticket, path in zip(audits, tickets, audit_paths):
        for arm in ('bf16', 'fp8_e4m3'):
            metrics = audit['sealed_arms'][arm]['metrics']
            if any(metrics[key] != baseline_metrics[key] for key in (
                    'profiling_records', 'warmup_records', 'profile_session_ids_sha256', 'profile_prompt_counts_sha256')):
                raise ValueError('Paired requests or measured/warmup identities differ')
        rows.append({'run_id': audit['run_id'], 'arm_order': ticket['request']['arms'],
                     'audit_sha256': _sha(path), 'request_sha256': ticket['request_sha256'],
                     'metrics': {arm: audit['sealed_arms'][arm]['metrics'] for arm in ('bf16', 'fp8_e4m3')},
                     'delta_pct': audit['delta_pct'], 'actual_gpu_hours': audit['actual_gpu_hours']})
    pooled = {}
    for arm in ('bf16', 'fp8_e4m3'):
        count = sum(row['metrics'][arm]['profiling_records'] for row in rows)
        duration = sum(row['metrics'][arm]['observation_seconds'] for row in rows)
        tokens = sum(row['metrics'][arm]['actual_output_tokens'] for row in rows)
        pooled[arm] = {'request_rps': count/duration, 'output_tps': tokens/duration,
                       'observed_seconds_sum': duration, 'requests_sum': count, 'actual_output_tokens_sum': tokens}
    changes = {}
    for key in ('request_rps', 'output_tps', 'ttft_p95_ms', 'itl_p50_ms', 'e2e_p95_ms'):
        values = [row['delta_pct'][key] for row in rows]
        changes[key] = {'observed_min_pct': min(values), 'observed_max_pct': max(values),
                        'geometric_mean_ratio_delta_pct': (math.exp(sum(math.log(1+value/100) for value in values)/2)-1)*100}
    return {'schema': 'kv-serving-ABBA-replication-comparison-v1',
            'classification': 'confirmed_two_sameGPU_opposite_order_deployment_pairs',
            'independent_paired_runs': 2, 'GPU_UUID': first['gpu_uuids'][0],
            'source_commit': first['python_source_commit'], 'dataset_sha256': first['dataset_sha256'],
            'frozen_assets_identical': True, 'rows': rows, 'observed_run_delta_ranges': changes,
            'duration_weighted_observation_pool': pooled,
            'duration_weighted_request_rps_delta_pct': (pooled['fp8_e4m3']['request_rps']/pooled['bf16']['request_rps']-1)*100,
            'duration_weighted_output_tps_delta_pct': (pooled['fp8_e4m3']['output_tps']/pooled['bf16']['output_tps']-1)*100,
            'quality_accepted': False, 'run_level_confidence_interval': None,
            'limitations': ['Two runs give an observed range, not a reliable population confidence interval',
                           'The same original512 measured and16 warmup requests are reused, not1024 independent source users',
                           'Observation-pool rates exclude JIT/model-load/cold-start time; report whole-job costs separately',
                           'Actual natural output lengths differ and all per-run values are retained',
                           'BF16 and FP8 prefill/internal-Q/writer configurations differ; not a pure KV-dtype causal effect',
                           'Measured quality cost and other workloads/SLOs remain separate production gates']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audits', nargs=2, type=Path, required=True)
    parser.add_argument('--mirrors', nargs=2, type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.audits, args.mirrors)
    with args.out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result['observed_run_delta_ranges']))


if __name__ == '__main__':
    main()
