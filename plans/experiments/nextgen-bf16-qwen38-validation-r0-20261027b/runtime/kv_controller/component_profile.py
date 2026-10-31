# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Bounded windows and independent actual activity checks for an owned server."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from .durable import write_new_json


CONFIG = {"schema": "xllm-owned-component-profile-v1", "max_forwards": 16,
          "max_seconds": 6, "max_trace_mib": 32}


def validate_windows(root: Path, config: dict, audit: dict) -> dict:
    plan = json.loads((root / 'window_plan.json').read_text())
    process = json.loads((root / 'PROCESS.json').read_text())
    if (process.get('config') != config or process.get('scope') != 'owned_process_only_no_system_attach'
            or plan.get('request_sha256') != config['request_sha256']):
        raise ValueError('Diagnostic process/plan does not bind the admitted run')
    warm_end = audit['warm_cache_end_epoch_ns']
    offered_end = audit['phase_start_ns'] + round(audit['offered_window_seconds'] * 1e9)
    windows, replay_calls = [], 0
    for planned in plan['windows']:
        index = planned['index']
        result = json.loads((root / f'window-{index:02d}.json').read_text())
        if (result.get('status') != 'CAPTURED_REQUIRES_INDEPENDENT_TRACE_VALIDATION'
                or result.get('pid') != process['pid']
                or any(result.get(k) != config[k] for k in ('run_id', 'request_sha256', 'arm'))
                or not warm_end <= result['start_epoch_ns'] < result['end_epoch_ns'] <= offered_end
                or not planned['not_before_epoch_ns'] <= result['start_epoch_ns'] <= planned['start_deadline_epoch_ns']
                or not 1 <= len(result['steps']) <= config['max_forwards']
                or any(not row['completed'] or type(row.get('sequences')) is not int
                       or not 1 <= row['sequences'] <= 128 or type(row.get('input_tokens')) is not int
                       or not 1 <= row['input_tokens'] <= 10240
                       or not result['start_epoch_ns'] <= row['start_epoch_ns'] <= row['end_epoch_ns'] <= result['end_epoch_ns']
                       for row in result['steps'])
                or result['steps'][-1]['start_epoch_ns'] - result['start_epoch_ns'] > config['max_seconds'] * 10**9):
            raise ValueError('Profiling window missed its real measurement scope or failed')
        expected = f'window-{index:02d}.trace.json'
        path = root / expected
        if (result.get('trace_file') != expected or path.is_symlink()
                or not 0 < path.stat().st_size <= config['max_trace_mib'] * 2**20
                or hashlib.sha256(path.read_bytes()).hexdigest() != result['trace_sha256']):
            raise ValueError('Actual profiling trace size/path/SHA drift')
        trace = json.loads(path.read_text())
        events = trace.get('traceEvents', [])
        kernels = [row for row in events if row.get('cat') == 'kernel']
        base_ns = trace.get('baseTimeNanoseconds')
        if type(base_ns) is not int or any(type(row.get('ts')) not in (int, float)
                or type(row.get('dur')) not in (int, float) or row['dur'] < 0
                or not warm_end <= base_ns + round(row['ts'] * 1000)
                or base_ns + round((row['ts'] + row['dur']) * 1000) > offered_end
                for row in kernels):
            raise ValueError('Actual CUDA activities are outside the declared measurement clock')
        prefix = 'bf16' if config['arm'] == 'bf16' else 'fp8'
        required = {'xllm.component.' + prefix + '.' + name
                    for name in ('kv_materialize', 'kv_write', 'prefill_attention')}
        if (not kernels or len(trace.get('deviceProperties', [])) != 1
                or any(row.get('args', {}).get('device') != 0 for row in kernels)
                or not required.issubset({row.get('name') for row in events if row.get('cat') == 'user_annotation'})
                or not any(row['prefill'] or row['chunked_prefill'] for row in result['steps'])):
            raise ValueError('Actual owned CUDA kernels or prefill component activities missing')
        replays = sum(row['components'].get('graph_replay', 0) for row in result['steps'])
        launches = sum('cudaGraphLaunch' in row.get('name', '') for row in events)
        if replays > 0 and launches < replays:
            raise ValueError('Declared Graph replay lacks actual CUDA Graph launch activity')
        replay_calls += replays
        windows.append({'index': index, 'trace_sha256': result['trace_sha256'],
                        'kernel_events': len(kernels), 'graph_replays': replays,
                        'start_epoch_ns': result['start_epoch_ns'], 'end_epoch_ns': result['end_epoch_ns'],
                        'trace_size': path.stat().st_size, 'forward_calls': len(result['steps'])})
    if len(windows) != 3 or replay_calls == 0:
        raise ValueError('Require three prefill windows and actual production Graph replay')
    return {'schema': 'kv-owned-component-activity-verification-v1', 'windows': windows,
            'actual_owned_CUDA_kernel_activities_verified': True, 'real_Graph_replay_calls': replay_calls,
            'new_headline_performance_replicate': False, 'model_quality_validated': False,
            'limitations': ['Only declared windows; observation overhead belongs to diagnostic',
                           'CPU annotations are not GPU durations; overlapped kernel sums are not end-to-end latency',
                           'Anonymous payloads reproduce source lengths/arrivals/prefixes, not semantic text']}


class ComponentProfileCoordinator:
    def __init__(self, output: Path, ticket: dict, arm: str, manifest: dict) -> None:
        self.root = output / 'diagnostic'
        self.root.mkdir(mode=0o700)
        self.config = {**CONFIG, 'run_id': ticket['request']['run_id'],
                       'request_sha256': ticket['request_sha256'], 'arm': arm}
        write_new_json(self.root / 'config.json', self.config)
        self._warm_seconds = ((manifest['window_begin_ms'] + 60000 - manifest['first_retained_timestamp_ms'])
                              / 1000 / manifest['arrival_scaling'])
        self._client_started_ns: int | None = None
        self._counter_baseline: float | None = None
        self._armed = False

    def client_started(self) -> None:
        self._client_started_ns = time.time_ns()

    def on_sample(self, row: dict) -> None:
        if self._armed or self._client_started_ns is None:
            return
        counter = row['values'].get('prefix_cache_prompt_tokens_total')
        if counter is None:
            return
        if self._counter_baseline is None:
            self._counter_baseline = counter
            return
        if counter <= self._counter_baseline:
            return
        # This is a coarse, observed post-client activity anchor, not an asserted
        # offered/arrival origin. Actual exported phase clocks are checked later.
        anchor = time.time_ns()
        windows = [{'index': i, 'not_before_epoch_ns': anchor + round((self._warm_seconds + offset) * 1e9),
                    'start_deadline_epoch_ns': anchor + round((self._warm_seconds + offset + 30) * 1e9)}
                   for i, offset in enumerate((15, 75, 135))]
        write_new_json(self.root / 'window_plan.json', {
            'schema': 'xllm-owned-profile-windows-v1',
            **{k: self.config[k] for k in ('run_id', 'request_sha256', 'arm')},
            'windows': windows, 'anchor_epoch_ns': anchor, 'anchor_sample': row,
            'anchor_kind': 'observed_prefix_token_counter_increase_after_client_start',
            'anchor_is_exact_offered_origin': False})
        self._armed = True

    def verify(self, audit: dict) -> dict:
        if not self._armed:
            raise ValueError('No observed post-client application activity; no profiling windows armed')
        result = validate_windows(self.root, self.config, audit)
        write_new_json(self.root / 'activity_verification.json', result)
        return result
