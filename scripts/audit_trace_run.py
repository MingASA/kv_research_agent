#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Independently verify sealed trace service outcomes and owned bvar observations."""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.controller.durable import digest
from scripts.controller.evidence import verify_allocation
from scripts.controller.production import verify_backend
from scripts.controller.trace_serving import audit_trace, _distribution


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _terminal_context(root: Path, ticket: dict, result: dict) -> datetime.datetime | None:
    """Keep conflicts visible; only complete, driver-released evidence is auditable."""
    request = ticket['request']
    if (digest(request) != ticket['request_sha256'] or result['request_sha256'] != ticket['request_sha256']
            or result.get('run_id') != request['run_id'] or result['gpu_release_verified'] is not True
            or result['live_owned_containers'] or result['status'] not in ('COMPLETE', 'CONFLICT')):
        raise ValueError('Trace terminal identity/status or driver release differs')
    if result['status'] == 'COMPLETE':
        if json.loads((root/'COMPLETE').read_text())['result_sha256'] != _sha(root/'result.json'):
            raise ValueError('Trace COMPLETE does not seal its exact result')
        return None
    if (root/'COMPLETE').exists():
        raise ValueError('A conflict cannot also claim root COMPLETE')
    stops = [row for row in (json.loads(line) for line in (root/'events.jsonl').read_text().splitlines())
             if row.get('action') == 'STOP_OR_FAILURE']
    if (len(stops) != 1 or stops[0].get('status') != 'CONFLICT'
            or stops[0].get('error') != result.get('error')):
        raise ValueError('Conflict has no unique matching original stop event')
    when = datetime.datetime.fromisoformat(stops[0]['utc'])
    if when.tzinfo is None:
        raise ValueError('Conflict timing must have an explicit timezone')
    return when


def _conflict_after_responses(when: datetime.datetime | None, metrics: dict) -> float | None:
    if when is None:
        return None
    last_response = max(row['request_end_ns'] for row in metrics['per_request'])
    delta = (round(when.timestamp()*1e9) - last_response) / 1e9
    if delta <= 0:
        raise ValueError('Conflict occurred during request measurement; do not recover this arm')
    return delta


def _application(path: Path, run_id: str, request_sha: str, pages: int, metrics: dict) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    header = rows.pop(0)
    if (header.get('schema') != 'kv-owned-service-metrics-v1' or header.get('run_id') != run_id
            or header.get('request_sha256') != request_sha or header.get('physical_pages') != pages
            or header.get('reserved_padding_pages') != 1):
        raise ValueError('Application observation identity/capacity differs')
    per_request = metrics['per_request']
    measured_start = metrics['warm_cache_end_epoch_ns']
    end = max(r['request_end_ns'] for r in per_request)
    selected = [r for r in rows if measured_start <= round(datetime.datetime.fromisoformat(r['utc']).timestamp()*1e9) <= end]
    errors = collections.Counter(name for r in selected for name in r['errors'])
    gauges = {name: _distribution([r['values'][name] for r in selected if name in r['values']])
              for name, unit in header['units'].items() if unit != 'cumulative_tokens'}
    observed = {}
    for name in ('prefix_cache_prompt_tokens_total', 'prefix_cache_hit_tokens_total'):
        values = [r['values'][name] for r in selected if name in r['values']]
        observed[name] = {'samples': len(values), 'monotonic': all(b >= a for a,b in zip(values, values[1:])),
                          'delta': values[-1]-values[0] if len(values)>1 else None}
    prompt, hit = observed['prefix_cache_prompt_tokens_total'], observed['prefix_cache_hit_tokens_total']
    rate = hit['delta']/prompt['delta'] if (prompt['monotonic'] and hit['monotonic'] and prompt['delta']
        and hit['delta'] is not None) else None
    return {'measurement_and_drain_samples': len(selected), 'total_samples': len(rows), 'metric_error_counts': dict(errors),
            'gauges': gauges, 'sample_bounded_counter_changes': observed, 'sample_bounded_prefix_hit_token_fraction': rate,
            'physical_pages': pages, 'usable_physical_pages_after_reserved0': pages-1,
            'used_blocks_include_reserved_page0': True,
            'limitation': 'Sequential1s application samples, not atomic snapshots/exact event counters; prefix pool can be reclaimable; no HBM counters'}


def audit(mirror: Path, run_id: str, source_repo: Path) -> dict:
    snapshot = json.loads((mirror/'SNAPSHOT.json').read_text())
    for entry in snapshot['files']:
        path = mirror/entry['relative_path']
        if path.is_symlink() or path.stat().st_size != entry['size'] or _sha(path) != entry['sha256']:
            raise ValueError('Trace mirror inventory/hash changed')
    root = mirror/'support/agent_work/runs'/run_id
    ticket = json.loads((root/'ticket.json').read_text()); request = ticket['request']
    result = json.loads((root/'result.json').read_text())
    conflict_time = _terminal_context(root, ticket, result)
    diagnostic = request['workload']['kind'] == 'anonymous_trace_component_diagnostic'
    if request['workload']['kind'] not in ('anonymous_trace_xqa', 'anonymous_trace_xqa_bf16_reference', 'anonymous_trace_component_diagnostic'):
        raise ValueError('Different service protocol')
    if diagnostic and conflict_time is not None:
        raise ValueError('Component observer diagnostic requires clean complete windows, not conflict recovery')
    for relative, expected in ticket['assets_sha256'].items():
        if _sha(root/relative) != expected:
            raise ValueError('Frozen runtime/input asset differs: '+relative)
    for relative, expected in request['backend_source_sha256'].items():
        code = subprocess.check_output(['git', 'show', request['python_source_commit']+':'+relative], cwd=source_repo)
        if hashlib.sha256(code).hexdigest() != expected:
            raise ValueError('Actual declared source differs from the frozen Git commit')
    source = root/Path(request['dataset_path']).relative_to(request['remote_output'])
    manifest = root/Path(request['workload']['trace_manifest']).relative_to(request['remote_output'])
    if _sha(source) != request['dataset_sha256']:
        raise ValueError('Actual uncompressed source input differs')
    arms = {}
    for arm in request['arms']:
        directory = root/'output'/arm
        measured = json.loads((directory/'measurement.json').read_text())
        arm_result = next((r for r in result['arm_results'] if r['arm'] == arm), None)
        if ((arm_result is None and conflict_time is None)
                or (arm_result is not None and _sha(directory/'measurement.json') != arm_result['result_sha256'])
                or measured['measurement_success'] is not True or measured.get('arm') != arm
                or measured['request_sha256'] != ticket['request_sha256'] or measured['official_aiperf_complete'] is not True):
            raise ValueError('Trace measurement did not seal completely')
        for relative, expected in measured['evidence_sha256'].items():
            path = directory/relative
            if not path.resolve().is_relative_to(directory.resolve()) or _sha(path) != expected:
                raise ValueError('Sealed trace evidence differs')
        backend = verify_backend((directory/'server.log').read_text(), arm, request['operators'][arm])
        allocation = verify_allocation(json.loads((directory/'allocation.json').read_text()), arm)
        recomputed = audit_trace(directory/'artifacts', source, manifest)
        recomputed['aiperf_exit_code'] = measured['metrics']['aiperf_exit_code']
        if recomputed != measured['metrics']:
            raise ValueError('Independent replay outcome/clock/usage audit differs')
        component = None
        if diagnostic:
            from scripts.controller.component_profile import validate_windows, CONFIG
            config = {**CONFIG, 'run_id': run_id, 'request_sha256': ticket['request_sha256'], 'arm': arm}
            component = validate_windows(directory / 'diagnostic', config, recomputed)
            if (component != measured.get('component_diagnostic')
                    or measured.get('scope') != 'production_component_observer_diagnostic_not_headline_performance'):
                raise ValueError('Independent component activity validation/scope differs')
        conflict_after = _conflict_after_responses(conflict_time, recomputed)
        observations = _application(directory/'service_metrics.jsonl', run_id, ticket['request_sha256'],
                                    allocation['pages_per_layer'], recomputed)
        completed = [r for r in recomputed['per_request'] if r['logical_phase']=='measurement' and r['success']]
        elapsed = recomputed['phases']['measurement']['elapsed_including_drain_seconds']
        e2e_curve = [{'offered_e2e_limit_seconds':seconds,
            'qualified_completion_rps_including_drain':sum(r['offered_e2e_ms']<=seconds*1000 for r in completed)/elapsed}
            for seconds in (.1,.25,.5,1,2,5,10,30,60)]
        arms[arm] = {'metrics': recomputed, 'allocation': allocation, 'backend': backend,
                    **({'component_diagnostic': component} if diagnostic else {}),
                    'application_observations': observations,
                    'e2e_slo_curve_illustrative_not_user_accepted_or_quality_goodput':e2e_curve,
                    'warm_steady_state_established':False,
                    'conflict_seconds_after_last_response':conflict_after}
    return {'schema': 'kv-production-trace-local-audit-v1',
            'classification': ('confirmed_owned_process_component_diagnostic_integrity' if diagnostic else
                               'confirmed_trace_measurement_integrity_runtime_conflict_retained' if conflict_time
                               else 'confirmed_trace_measurement_integrity'),
            'run_id': run_id, 'terminal_result_sha256': _sha(root/'result.json'), 'root_complete': conflict_time is None,
            'terminal_status_preserved':result['status'],
            'supervisor_sealed_arms':[row['arm'] for row in result['arm_results']],
            'gpu_release_verified': True, 'workload': request['workload'], 'sealed_arms': arms,
            'actual_gpu_hours': sum((i['end_ns']-i['start_ns'])/3.6e12 for i in result['allocation_intervals_ns']),
            'quality_accepted': False, 'weight_contents_rehashed_in_this_performance_run': False,
            'limitations': ['Anonymous source arrival/length/prefix replay, no original semantic text',
                           'Complete service support also requires matched quality and pressure/repeatability coverage',
                           'SLO thresholds are illustrative, not user accepted; ITL is per-request mean',
                           'No hardware HBM traffic; total KV allocation is not maximum usable service capacity']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mirror', 'source-repo', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    result = audit(args.mirror, args.run_id, args.source_repo)
    with args.out.open('x') as f: f.write(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'run_id': args.run_id, 'classification': result['classification'], 'gpu_release_verified': True,
        'terminal_status_preserved':result['terminal_status_preserved'],
        'phases': {arm: data['metrics']['phases'] for arm,data in result['sealed_arms'].items()}}))


if __name__ == '__main__':
    main()
