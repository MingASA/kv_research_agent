#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Separate CPU scopes, correlated CUDA kernel time and overlapping busy time.

This does not estimate HBM traffic, headline serving performance or causality.
Unmatched/ambiguous launch correlations remain explicitly unassigned.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import collections
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.controller.component_profile import CONFIG, validate_windows
from scripts.controller.durable import write_new_json


def _interval(event: dict) -> tuple[float, float]:
    start, duration = event.get('ts'), event.get('dur')
    if (type(start) not in (int, float) or type(duration) not in (int, float)
            or not math.isfinite(start) or not math.isfinite(duration) or duration < 0):
        raise ValueError('Chrome complete activity requires finite microsecond start/duration')
    return start, start + duration


def _union_us(events: list[dict]) -> float:
    intervals = sorted(_interval(event) for event in events)
    total, right = 0.0, -math.inf
    for start, end in intervals:
        total += max(0, end - max(start, right))
        right = max(right, end)
    return total


class _Scopes:
    def __init__(self, scopes: list[dict]) -> None:
        self._threads: dict[tuple, list[dict]] = collections.defaultdict(list)
        for scope in scopes:
            _interval(scope)
            self._threads[(scope.get('pid'), scope.get('tid'))].append(scope)
        self._starts = {}
        for thread, rows in self._threads.items():
            rows.sort(key=lambda row: row['ts'])
            # Instrumented component scopes are siblings; refuse a silently
            # ambiguous containment model if future instrumentation nests them.
            if any(_interval(left)[1] > right['ts'] + .001 for left, right in zip(rows, rows[1:])):
                raise ValueError('Overlapping sibling CPU scopes need an explicit attribution contract')
            self._starts[thread] = [row['ts'] for row in rows]

    def containing(self, event: dict) -> dict | None:
        thread = (event.get('pid'), event.get('tid'))
        start, end = _interval(event)
        index = bisect_right(self._starts.get(thread, []), start) - 1
        if index < 0:
            return None
        candidate = self._threads[thread][index]
        return candidate if end <= _interval(candidate)[1] + .001 else None


def analyze_trace(trace: dict, metadata: dict) -> dict:
    events = [event for event in trace.get('traceEvents', []) if event.get('ph') == 'X']
    kernels = [event for event in events if event.get('cat') == 'kernel']
    transfers = [event for event in events if event.get('cat') in ('gpu_memcpy', 'gpu_memset')]
    if (not kernels or len(trace.get('deviceProperties', [])) != 1
            or any(event.get('args', {}).get('device') != 0 for event in kernels + transfers)):
        raise ValueError('Require actual single-visible-device CUDA activities')
    annotations = [event for event in events if event.get('cat') == 'user_annotation']
    components = [event for event in annotations if event.get('name', '').startswith('xllm.component.')]
    forwards = sorted([event for event in annotations if event.get('name') in
                       ('xllm.forward.prefill', 'xllm.forward.decode')], key=lambda event: event['ts'])
    if (len(forwards) != len(metadata['steps'])
            or any((event['name'] == 'xllm.forward.prefill') != bool(step['prefill'] or step['chunked_prefill'])
                   for event, step in zip(forwards, metadata['steps']))):
        raise ValueError('Actual CPU forward annotations do not bind the recorded batch metadata')
    component_scopes, forward_scopes = _Scopes(components), _Scopes(forwards)
    forward_ids = {id(event): index for index, event in enumerate(forwards)}
    launches: dict[int, list[dict]] = collections.defaultdict(list)
    for event in events:
        correlation = event.get('args', {}).get('correlation')
        if event.get('cat') in ('cuda_runtime', 'cuda_driver') and type(correlation) is int:
            launches[correlation].append(event)
    component_gpu: dict[str, list[dict]] = collections.defaultdict(list)
    forward_gpu: dict[int, list[dict]] = collections.defaultdict(list)
    forward_components: dict[int, dict[str, list[dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    reasons: collections.Counter = collections.Counter()
    for event in kernels + transfers:
        correlation = event.get('args', {}).get('correlation')
        candidates = launches.get(correlation, []) if type(correlation) is int else []
        parents = [component_scopes.containing(candidate) for candidate in candidates]
        labels = {parent['name'] if parent is not None else 'unassigned' for parent in parents}
        label = next(iter(labels)) if len(labels) == 1 else 'unassigned'
        if label == 'unassigned':
            reasons['missing_launch_correlation' if not candidates else 'missing_or_ambiguous_component_scope'] += 1
        component_gpu[label].append(event)
        parents = [forward_scopes.containing(candidate) for candidate in candidates]
        indices = {forward_ids[id(parent)] if parent is not None else -1 for parent in parents}
        forward_id = next(iter(indices)) if len(indices) == 1 else -1
        forward_gpu[forward_id].append(event)
        forward_components[forward_id][label].append(event)
    summary = {}
    labels = set(component_gpu) | {event['name'] for event in components}
    for label in sorted(labels):
        host = [event for event in components if event['name'] == label]
        device = component_gpu[label]
        compute = [event for event in device if event.get('cat') == 'kernel']
        copies = [event for event in device if event.get('cat') != 'kernel']
        summary[label] = {'cpu_scope_calls': len(host),
            'cpu_scope_sum_ms': sum(event['dur'] for event in host) / 1000,
            'cuda_kernel_events': len(compute), 'cuda_kernel_sum_ms': sum(event['dur'] for event in compute) / 1000,
            'cuda_kernel_busy_union_ms': _union_us(compute) / 1000,
            'cuda_explicit_copy_or_memset_events': len(copies),
            'cuda_explicit_copy_or_memset_sum_ms': sum(event['dur'] for event in copies) / 1000}
    names: dict[str, list[dict]] = collections.defaultdict(list)
    for kernel in kernels:
        names[kernel['name']].append(kernel)
    return {'schema': 'kv-owned-component-trace-time-analysis-v1',
        'baseTimeNanoseconds': trace.get('baseTimeNanoseconds'),
        'cuda_kernel_events': len(kernels), 'cuda_kernel_sum_ms': sum(event['dur'] for event in kernels) / 1000,
        'cuda_all_activity_busy_union_ms': _union_us(kernels + transfers) / 1000,
        'outside_recorded_forward_cuda_kernel_sum_ms': sum(event['dur'] for event in forward_gpu[-1]
                                                           if event.get('cat') == 'kernel') / 1000,
        'unassigned_activity_reasons': dict(reasons), 'components': summary,
        'forward_batches': [{**step, 'trace_annotation': annotation['name'],
            'cpu_scope_ms': annotation['dur'] / 1000,
            'component_cuda_kernel_sum_ms': {label: sum(event['dur'] for event in rows
                if event.get('cat') == 'kernel') / 1000 for label, rows in forward_components[index].items()},
            'correlated_cuda_kernel_sum_ms': sum(event['dur'] for event in forward_gpu[index]
                                                if event.get('cat') == 'kernel') / 1000}
                           for index, (annotation, step) in enumerate(zip(forwards, metadata['steps']))],
        'kernel_names': {name: {'calls': len(rows), 'cuda_kernel_sum_ms': sum(row['dur'] for row in rows) / 1000}
                         for name, rows in sorted(names.items())},
        'limitations': ['CPU scopes are host durations; CUDA activities are device durations',
            'Kernel sums include overlap and are not end-to-end latency; busy union is separate',
            'Graph replay is attributed as a whole; no invented per-component graph breakdown',
            'Unmatched launch correlations are retained; no HBM bandwidth or causal speedup claim']}


def analyze(mirror: Path, audit_path: Path, run_id: str) -> dict:
    audit = json.loads(audit_path.read_text())
    if (audit.get('run_id') != run_id or audit.get('classification') !=
            'confirmed_owned_process_component_diagnostic_integrity' or not audit.get('gpu_release_verified')):
        raise ValueError('Require independent complete component integrity audit after driver release')
    root = mirror / 'support/agent_work/runs' / run_id
    ticket = json.loads((root / 'ticket.json').read_text())
    if hashlib.sha256((root / 'result.json').read_bytes()).hexdigest() != audit['terminal_result_sha256']:
        raise ValueError('Audited terminal result changed')
    arms = {}
    for arm in ticket['request']['arms']:
        directory = root / 'output' / arm / 'diagnostic'
        cfg = {**CONFIG, 'run_id': run_id, 'request_sha256': ticket['request_sha256'], 'arm': arm}
        if validate_windows(directory, cfg, audit['sealed_arms'][arm]['metrics']) != audit['sealed_arms'][arm]['component_diagnostic']:
            raise ValueError('Diagnostic metadata no longer matches the independently verified audit')
        windows = []
        for index in range(3):
            metadata = json.loads((directory / f'window-{index:02d}.json').read_text())
            trace = json.loads((directory / metadata['trace_file']).read_text())
            windows.append({'index': index, 'trace_sha256': metadata['trace_sha256'], **analyze_trace(trace, metadata)})
        arms[arm] = windows
    return {'classification': 'confirmed_diagnostic_activity_times_not_headline_performance', 'run_id': run_id,
        'independent_audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'arms': arms,
        'new_headline_performance_replicate': False, 'quality_accepted': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mirror', 'audit', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    value = analyze(args.mirror, args.audit, args.run_id)
    write_new_json(args.out, value)
    print(json.dumps({'run_id': args.run_id, 'classification': value['classification'], 'out': str(args.out)}))


if __name__ == '__main__':
    main()
