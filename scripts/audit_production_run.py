#!/usr/bin/env python3
"""Independently audit frozen real-service exports, retaining terminal conflicts."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.controller.durable import digest
from scripts.controller.production import audit_requests, verify_backend
from scripts.controller.evidence import verify_allocation


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(mirror: Path, run_id: str) -> dict:
    snapshot = json.loads((mirror / 'SNAPSHOT.json').read_text())
    for entry in snapshot['files']:
        path = mirror / entry['relative_path']
        if path.is_symlink() or path.stat().st_size != entry['size'] or _sha(path) != entry['sha256']:
            raise ValueError('Mirrored source bytes changed: ' + entry['relative_path'])
    root = mirror / 'support/agent_work/runs' / run_id
    ticket = json.loads((root / 'ticket.json').read_text())
    request = ticket['request']
    terminal = json.loads((root / 'result.json').read_text())
    if (digest(request) != ticket['request_sha256'] or terminal['request_sha256'] != ticket['request_sha256']
            or terminal['run_id'] != run_id or terminal['gpu_release_verified'] is not True
            or terminal['live_owned_containers']):
        raise ValueError('Request/terminal identity or GPU release differs')
    source = root / 'input/canonical_inputs.json'
    if _sha(source) != request['dataset_sha256']:
        raise ValueError('Genuine frozen input hash differs')
    arms = {}
    for arm in request['arms']:
        directory = root / 'output' / arm
        measured = json.loads((directory / 'measurement.json').read_text())
        if (measured['measurement_success'] is not True or measured['official_aiperf_complete'] is not True
                or measured['request_sha256'] != ticket['request_sha256'] or measured['arm'] != arm):
            raise ValueError('Per-arm measurement did not actually seal')
        for name, expected in measured['evidence_sha256'].items():
            path = directory / name
            if not path.resolve().is_relative_to(directory.resolve()) or _sha(path) != expected:
                raise ValueError('Sealed per-arm evidence hash differs: ' + name)
        backend = verify_backend((directory / 'server.log').read_text(), arm, request['operators'][arm])
        allocation = verify_allocation(json.loads((directory / 'allocation.json').read_text()), arm)
        metrics = audit_requests(directory / 'artifacts', source, request['workload']['requests'], request['workload']['warmup'])
        if metrics != measured['metrics']:
            raise ValueError('Independently recomputed metrics differ from the sealed arm')
        rows = [json.loads(line) for line in (directory / 'artifacts/profile_export.jsonl').read_text().splitlines()]
        rows = [row for row in rows if row['metadata']['benchmark_phase'] == 'profiling']
        end_ns = max(row['metadata']['request_end_ns'] for row in rows)
        latency_curve = []
        for ttft_ms in (1000, 2000, 5000, 10000, 20000):
            qualified = sum(row['metrics']['time_to_first_token']['value'] <= ttft_ms
                            and row['metrics']['inter_token_latency']['value'] <= 100 for row in rows)
            latency_curve.append({'ttft_limit_ms': ttft_ms, 'request_mean_itl_limit_ms': 100,
                                  'completed_requests': qualified, 'qualified_rps': qualified / metrics['observation_seconds']})
        arms[arm] = {'metrics': metrics, 'backend': backend, 'allocation': allocation,
                     'profiling_last_response_utc': datetime.datetime.fromtimestamp(end_ns / 1e9, datetime.timezone.utc).isoformat(),
                     'latency_constraint_curve_not_quality_goodput': latency_curve,
                     'measurement_sha256': _sha(directory / 'measurement.json')}
    baseline, candidate = arms['bf16'], arms['fp8_e4m3']
    for key in ('profile_session_ids_sha256', 'profile_prompt_counts_sha256'):
        if baseline['metrics'][key] != candidate['metrics'][key]:
            raise ValueError('Real session/prompt counts do not pair')
    delta = {key: 100 * (candidate['metrics'][key] / baseline['metrics'][key] - 1) for key in (
        'request_rps', 'output_tps', 'osl_mean', 'ttft_p50_ms', 'ttft_p95_ms', 'ttft_p99_ms',
        'itl_p50_ms', 'itl_p95_ms', 'itl_p99_ms', 'e2e_p50_ms', 'e2e_p95_ms', 'e2e_p99_ms')}
    conflicts = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()
                 if json.loads(line).get('action') == 'STOP_OR_FAILURE']
    conflict_after = None
    if terminal['status'] == 'CONFLICT':
        last = datetime.datetime.fromisoformat(candidate['profiling_last_response_utc'])
        conflict_after = (datetime.datetime.fromisoformat(conflicts[-1]['utc']) - last).total_seconds()
        if conflict_after <= 0:
            raise ValueError('Conflict occurred during the measured request window; do not recover clean measurements')
    elif terminal['status'] != 'COMPLETE':
        raise ValueError('Unsupported non-complete terminal status')
    else:
        if json.loads((root/'COMPLETE').read_text())['result_sha256'] != _sha(root/'result.json'):
            raise ValueError('Production root completion/result seal differs')
        for arm in request['arms']:
            if next(r['result_sha256'] for r in terminal['arm_results'] if r['arm']==arm) != arms[arm]['measurement_sha256']:
                raise ValueError('Supervisor/worker production arm seal differs')
    return {'schema': 'kv-production-local-audit-v1', 'run_id': run_id, 'terminal_status_preserved': terminal['status'],
            'classification': 'measurement_integrity_verified_runtime_conflict_retained' if conflict_after else 'measurement_integrity_verified',
            'root_complete': (root / 'COMPLETE').exists(), 'sealed_arms': arms, 'delta_pct': delta,
            'conflict_seconds_after_last_profiling_response': conflict_after,
            'conflict_cause': ('unverified; old monitor can classify an already exited GPU PID as foreign'
                               if terminal['status']=='CONFLICT' else None),
            'capacity_ratio': candidate['allocation']['token_capacity'] / baseline['allocation']['token_capacity'],
            'actual_gpu_hours': sum((i['end_ns'] - i['start_ns']) / 3.6e12 for i in terminal['allocation_intervals_ns']),
            'quality_validated': False, 'replicates': 1, 'production_support_decision': 'insufficient_quality_repeatability_and_trace_coverage',
            'comparison_operators':request['operators'],
            'pure_KV_dtype_effect_identified':False,'best_BF16_kernel_proven':False,
            'limitations': [('Runtime conflict is preserved, not converted to root COMPLETE; ownership trigger not retrospectively proved'
                             if terminal['status']=='CONFLICT' else 'Clean root/arm COMPLETE and driver release verified; this remains one paired run'),
                           'Closed-loop ShareGPT C128 is one workload; no open-loop trace or low-load curve yet',
                           ('FP8 KV: BF16-Q FA2 prefill plus XQA decode with internal FP8 Q; this is a kernel/configuration comparison'
                            if request['operators']['fp8_e4m3'].get('decode')=='xqa'
                            else 'FA3 candidate quantizes both Q and KV; this is a kernel/configuration comparison'),
                           'Quality acceptance is a separate gate, not established by this serving audit; generation differs naturally',
                           'No measured hardware HBM traffic or scheduler/cache occupancy mechanism attribution',
                           'Latency-constraint curves use illustrative thresholds and request-mean ITL, not user-approved SLOs']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mirror', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.mirror, args.run_id)
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'classification': result['classification'], 'terminal_status_preserved': result['terminal_status_preserved'],
                      'delta_pct': result['delta_pct'], 'quality_validated': False}))


if __name__ == '__main__':
    main()
