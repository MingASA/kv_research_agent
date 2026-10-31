#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Join audited pressure points with identical payloads and inference runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_common(requests: list[dict], tickets: list[dict], manifests: list[dict]) -> None:
    first = requests[0]
    fields = ('arms', 'binary_sha256', 'server_source_commit', 'python_source_commit',
              'backend_source_sha256', 'image_id', 'graph_mode', 'prefix_cache',
              'operators', 'service', 'concurrency', 'fp8_scales', 'model_config_sha256')
    workload_fields = ('kind', 'family', 'requests', 'warmup', 'content_scope',
                       'source_content_sha256', 'ignore_eos', 'arrival_mode')
    asset_names = {name for name in tickets[0]['assets_sha256']
                   if name.startswith('runtime/')} | {'input/source.bundle'}
    normalized = [{k: v for k, v in m.items() if k not in
                   ('arrival_scaling', 'payload_trace_sha256', 'transformation')}
                  for m in manifests]
    if any(m != normalized[0] for m in normalized[1:]):
        raise ValueError('Source records/payload SHA/prefix/window/cohort changed')
    for request, ticket in zip(requests[1:], tickets[1:]):
        if any(request[k] != first[k] for k in fields):
            raise ValueError('Inference/model/service conditions changed')
        if any(request['workload'][k] != first['workload'][k] for k in workload_fields):
            raise ValueError('Workload source or replay protocol changed')
        names = {name for name in ticket['assets_sha256'] if name.startswith('runtime/')} | {'input/source.bundle'}
        if names != asset_names or any(ticket['assets_sha256'][k] != tickets[0]['assets_sha256'][k]
                                       for k in asset_names):
            raise ValueError('Frozen runtime or source bundle changed')
    if len({r['run_id'] for r in requests}) != len(requests):
        raise ValueError('Repeated run is not another pressure point')
    if len({r['workload']['arrival_scaling'] for r in requests}) != len(requests):
        raise ValueError('Repeated pressure is not a distinct load point')


def compare(audit_paths: list[Path], tail_paths: list[Path], mirrors: list[Path]) -> dict:
    if len(audit_paths) < 2 or len(audit_paths) != len(tail_paths) or len(mirrors) != len(audit_paths):
        raise ValueError('Need at least two corresponding independently audited points')
    audits, tails, tickets, manifests = [], [], [], []
    for path, tail_path, mirror in zip(audit_paths, tail_paths, mirrors):
        audit = json.loads(path.read_text()); tail = json.loads(tail_path.read_text())
        if (audit.get('classification') != 'confirmed_trace_measurement_integrity'
                or not audit.get('root_complete') or not audit.get('gpu_release_verified')):
            raise ValueError('Require clean terminal independently audited trace pairs')
        if (tail.get('run_id') != audit['run_id'] or tail.get('audit_sha256') != _sha(path)
                or tail.get('original_metrics_changed') is not False):
            raise ValueError('Completion-timing analysis does not bind original audit')
        root = mirror / 'support/agent_work/runs' / audit['run_id']
        ticket = json.loads((root / 'ticket.json').read_text())
        manifest = json.loads((root / 'input/trace_manifest.json').read_text())
        if (ticket['request']['run_id'] != audit['run_id']
                or ticket['request']['dataset_sha256'] != manifest['payload_trace_sha256']
                or ticket['request']['workload'] != audit['workload']):
            raise ValueError('Run/input/protocol identity differs')
        audits.append(audit); tails.append(tail); tickets.append(ticket); manifests.append(manifest)
    requests = [t['request'] for t in tickets]
    _validate_common(requests, tickets, manifests)
    expected = {row['replay_id']: row for row in manifests[0]['records']}
    if len(expected) != len(manifests[0]['records']):
        raise ValueError('Duplicate source ID')
    points = []
    for audit, tail, ticket, path, tail_path in zip(audits, tails, tickets, audit_paths, tail_paths):
        arms = {}
        for arm in ('bf16', 'fp8_e4m3'):
            data = audit['sealed_arms'][arm]; metrics = data['metrics']
            seen = set()
            for row in metrics['per_request']:
                identifier = row['session_id']
                if identifier not in expected or identifier in seen:
                    raise ValueError('Unexpected/duplicate attempt')
                seen.add(identifier); source = expected[identifier]
                if row['logical_phase'] != source['logical_phase']:
                    raise ValueError('Warm/measurement identity differs')
                offset = (source['source_timestamp_ms'] - manifests[0]['first_retained_timestamp_ms'])
                offset_ns = int(offset / audit['workload']['arrival_scaling'] * 1_000_000)
                if row['scheduled_ns'] != metrics['phase_start_ns'] + offset_ns:
                    raise ValueError('Offered clock changed beyond declared scaling')
                if row['success'] and (row['output_tokens'] != source['source_output_tokens']
                                       or row['input_tokens'] != source['rendered_chat_tokens_cpu']):
                    raise ValueError('Completed source token work differs')
            if seen != set(expected):
                raise ValueError('Missing attempted source requests')
            measure = metrics['phases']['measurement']
            arms[arm] = {'measurement': measure, 'warm_cache': metrics['phases']['warm_cache'],
                         'drain_seconds': metrics['drain_seconds_after_offered_window'],
                         'allocation': data['allocation'], 'backend': data['backend'],
                         'application_observations': data['application_observations'],
                         'completion_timing': tail['arms'][arm]}
        bf, fp = [arms[a]['measurement'] for a in ('bf16', 'fp8_e4m3')]
        points.append({'run_id': audit['run_id'], 'gpu_uuids': ticket['request']['gpu_uuids'],
                       'arrival_scaling': audit['workload']['arrival_scaling'],
                       'offered_rps': bf['offered_rps'], 'arms': arms,
                       'completion_rps_delta_pct': (fp['completion_rps_including_drain'] /
                                                    bf['completion_rps_including_drain'] - 1) * 100,
                       'audit_sha256': _sha(path), 'tail_audit_sha256': _sha(tail_path),
                       'actual_gpu_hours': audit['actual_gpu_hours']})
    return {'schema': 'kv-production-trace-pressure-comparison-v1',
            'classification': 'confirmed_same_payload_runtime_two_pressure_observations',
            'script_sha256': _sha(Path(__file__)), 'source_commit': requests[0]['python_source_commit'],
            'same_frozen_runtime_and_bundle': True, 'identical_source_requests': len(expected),
            'points': sorted(points, key=lambda p: p['offered_rps']), 'quality_accepted': False,
            'limitations': ['One paired run per load; points used different GPUs/times, each arm pair same GPU',
                           'Two observed loads do not locate an exact SLO capacity or provide run-level CIs',
                           'Anonymous source lengths/prefixes/arrivals are not semantic quality or natural EOS',
                           'BF16/FP8 prefill/internal-Q/writer differ; no isolated KV-dtype causal effect',
                           'SSE-inclusive per-token mean is not hardware decode-step or token-stall timing',
                           'Old FA3-decode BF16 overload point is a different baseline and is excluded']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audits', nargs='+', type=Path, required=True)
    parser.add_argument('--tails', nargs='+', type=Path, required=True)
    parser.add_argument('--mirrors', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.audits, args.tails, args.mirrors)
    with args.out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'classification': result['classification'],
                      'offered_rps': [p['offered_rps'] for p in result['points']]}))


if __name__ == '__main__':
    main()
