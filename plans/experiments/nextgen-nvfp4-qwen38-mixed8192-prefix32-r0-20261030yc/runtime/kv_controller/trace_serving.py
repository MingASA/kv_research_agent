# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Validate anonymous arrival/length/prefix replay without hiding overload errors."""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path
import re

from .durable import digest
from . import evidence


def validate_inputs(source: Path, manifest_path: Path) -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    manifest = json.loads(manifest_path.read_text())
    if len(rows) != manifest['selected_requests'] or not rows:
        raise ValueError('Trace input count differs from the immutable manifest')
    records = {r['replay_id']: r for r in manifest['records']}
    if len(records) != len(rows) or len({r['session_id'] for r in rows}) != len(rows):
        raise ValueError('Trace has duplicate or missing replay identities')
    previous = -1.0
    for row in rows:
        record = records[row['session_id']]
        payload = row['payload']
        timestamp = row['timestamp']
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < previous:
            raise ValueError('Trace arrival clock is invalid or reordered')
        previous = timestamp
        expected = ((record['source_timestamp_ms'] - manifest['first_retained_timestamp_ms'])
                    / manifest['arrival_scaling'])
        if abs(timestamp - expected) > 1e-6:
            raise ValueError('Trace timestamps changed beyond the explicitly frozen scaling')
        if digest(payload) != record['payload_sha256']:
            raise ValueError('Frozen anonymous workload payload changed')
        if (payload.get('ignore_eos') is not True or payload.get('max_tokens') != record['source_output_tokens']
                or payload.get('stream') is not True or payload.get('stream_options') != {'include_usage': True}
                or record['rendered_chat_tokens_cpu'] + record['source_output_tokens'] > 32768):
            raise ValueError('Trace output length, streaming or model context contract differs')
    return rows, manifest


def _phase_start_ns(artifacts: Path) -> int:
    # Installed AIPerf0.13 exports naive local timestamps rounded to microseconds.
    # Its phase-completion journal retains the exact shared epoch-ns clock.
    log = (artifacts / 'logs/aiperf.log').read_text()
    values = {int(v) for v in re.findall(
        r'PhaseRecordsStats\(phase=CreditPhase\.PROFILING,[^\n]*?\bstart_ns=(\d+),', log)}
    if len(values) != 1:
        raise ValueError('Missing or ambiguous exact AIPerf profiling clock')
    return values.pop()


def _distribution(values: list[float]) -> dict:
    if not values:
        return {'count': 0}
    return {'count': len(values), 'mean': sum(values) / len(values), 'min': min(values), 'max': max(values),
            **{name: evidence._percentile(values, q) for name, q in (('p50', .5), ('p95', .95), ('p99', .99))}}


def audit_trace(artifacts: Path, source: Path, manifest_path: Path) -> dict:
    rows, manifest = validate_inputs(source, manifest_path)
    expected = {r['session_id']: r for r in rows}
    records = {r['replay_id']: r for r in manifest['records']}
    summary = json.loads((artifacts / 'profile_export_aiperf.json').read_text())
    if (summary.get('aiperf_version') != '0.13.0' or summary.get('is_complete') is not True
            or summary.get('was_cancelled') is not False):
        raise ValueError('Actual trace AIPerf did not complete and drain')
    origin = _phase_start_ns(artifacts)
    metrics = {}
    for line in (artifacts / 'profile_export.jsonl').read_text().splitlines():
        row = json.loads(line); meta = row['metadata']; identity = meta['conversation_id']
        if (identity in metrics or identity not in expected or meta.get('benchmark_phase') != 'profiling'
                or meta.get('turn_index') != 0 or meta.get('was_cancelled') is not False
                or meta.get('context_overflow_skip') is not False):
            raise ValueError('Missing/duplicated/skipped/cancelled trace request')
        metrics[identity] = row
    if set(metrics) != set(expected):
        raise ValueError('Trace source was not attempted exactly once')
    seen, statuses, outcomes = set(), collections.Counter(), []
    for line in (artifacts / 'profile_export_raw.jsonl').read_text().splitlines():
        raw = json.loads(line); meta = raw['metadata']; identity = meta['conversation_id']
        if identity in seen or identity not in metrics or raw.get('payload') != expected[identity]['payload']:
            raise ValueError('Trace wire payload/identity differs or is repeated')
        metric = metrics[identity]
        if any(meta.get(k) != metric['metadata'].get(k) for k in ('x_request_id', 'request_start_ns', 'request_end_ns', 'benchmark_phase')):
            raise ValueError('Trace metric/wire clocks or request identity differ')
        seen.add(identity); statuses[str(raw.get('status'))] += 1
        scheduled = origin + round(expected[identity]['timestamp'] * 1e6)
        start, end, issued = (meta['request_start_ns'], meta['request_end_ns'], meta['credit_issued_ns'])
        if not all(type(v) is int for v in (start, end, issued)) or end < start or start < scheduled - 100_000_000:
            raise ValueError('Invalid replay request/credit clocks')
        done, finish, usage, sse_error = False, None, None, False
        for response in raw.get('responses', []):
            for packet in response.get('packets', []):
                if packet.get('name') != 'data':
                    continue
                value = packet.get('value', '')
                if value == '[DONE]':
                    done = True
                    continue
                item = json.loads(value)
                sse_error = sse_error or bool(item.get('error'))
                usage = item.get('usage') or usage
                for choice in item.get('choices', []):
                    finish = choice.get('finish_reason') or finish
        success = (raw.get('status') == 200 and not raw.get('error') and not metric.get('error') and not sse_error)
        result = {'session_id': identity, 'logical_phase': records[identity]['logical_phase'],
                  'scheduled_ns': scheduled, 'request_start_ns': start, 'request_end_ns': end,
                  'client_start_lag_ms': (start - scheduled) / 1e6, 'credit_issue_lag_ms': (issued - scheduled) / 1e6,
                  'credit_to_start_ms': (start - issued) / 1e6, 'status': raw.get('status'), 'success': success}
        if success:
            if not done or finish != 'length' or usage is None:
                raise ValueError('Trace length-emulation SSE did not finish with explicit length/usage')
            wanted = records[identity]
            if (usage.get('completion_tokens') != wanted['source_output_tokens']
                    or usage.get('prompt_tokens') != wanted['rendered_chat_tokens_cpu']
                    or usage.get('total_tokens') != usage['prompt_tokens'] + usage['completion_tokens']):
                raise ValueError('Actual trace input/output tokens do not reproduce the frozen length workload')
            for field, name in (('completion_tokens', 'output_token_count'), ('prompt_tokens', 'input_sequence_length')):
                evidence._close(usage[field], evidence._value(metric['metrics'], name), 'Trace raw/metric usage differs')
            result.update(output_tokens=usage['completion_tokens'], input_tokens=usage['prompt_tokens'])
            for name, label in (('time_to_first_token', 'ttft_ms'), ('request_latency', 'e2e_ms')):
                result[label] = evidence._value(metric['metrics'], name)
            result['offered_ttft_ms'] = result['ttft_ms'] + result['client_start_lag_ms']
            result['offered_e2e_ms'] = (end - scheduled) / 1e6
            result['itl_ms'] = (evidence._value(metric['metrics'], 'inter_token_latency')
                                if usage['completion_tokens'] > 1 else None)
        else:
            # Rejection/HTTP errors are valid service outcomes, never silent successes.
            result['error'] = {'wire': raw.get('error'), 'metric': metric.get('error'), 'sse_error': sse_error}
            result.update(partial_server_usage_if_available=usage, stream_done=done, finish_reason=finish)
        outcomes.append(result)
    if seen != set(expected) or len({r['metadata']['x_request_id'] for r in metrics.values()}) != len(rows):
        raise ValueError('Trace raw records are incomplete or request IDs repeat')
    window = (manifest['window_end_ms'] - manifest['first_retained_timestamp_ms']) / 1000 / manifest['arrival_scaling']
    finish_ns = max(r['request_end_ns'] for r in outcomes)
    result = {'protocol': 'anonymous_arrival_length_prefix_replay_v1', 'all_attempts_verified': len(outcomes),
              'http_status_counts': dict(statuses), 'aiperf_error_summary': summary.get('error_summary'),
              'phase_start_ns': origin, 'offered_window_seconds': window,
              'warm_cache_end_epoch_ns': origin+round((manifest['window_begin_ms']+60000-manifest['first_retained_timestamp_ms'])
                                                    / manifest['arrival_scaling']*1e6),
              'drain_seconds_after_offered_window': max(0, (finish_ns - origin) / 1e9 - window),
              'arrival_scaling': manifest['arrival_scaling'], 'per_request': outcomes,
              'limitation': 'Anonymous token workload, no original semantic text; client request-start clock is not server arrival or hardware wire timestamp'}
    result['phases'] = {}
    for phase in ('warm_cache', 'measurement'):
        selected = [r for r in outcomes if r['logical_phase'] == phase]
        completed = [r for r in selected if r['success']]
        start_seconds = 0 if phase == 'warm_cache' else (manifest['window_begin_ms'] + 60000 - manifest['first_retained_timestamp_ms']) / 1000 / manifest['arrival_scaling']
        phase_window = (manifest['window_begin_ms'] + 60000 - manifest['first_retained_timestamp_ms']) / 1000 / manifest['arrival_scaling'] if phase == 'warm_cache' else window - start_seconds
        elapsed = max(phase_window, (max((r['request_end_ns'] for r in selected), default=origin) - origin) / 1e9 - start_seconds)
        stats = {'attempts': len(selected), 'successes': len(completed), 'errors': len(selected) - len(completed),
                 'offered_rps': len(selected) / phase_window, 'completion_rps_including_drain': len(completed) / elapsed,
                 'actual_output_tokens': sum(r['output_tokens'] for r in completed), 'elapsed_including_drain_seconds': elapsed}
        for field in ('client_start_lag_ms', 'credit_issue_lag_ms', 'credit_to_start_ms'):
            stats[field] = _distribution([r[field] for r in selected])
        for field in ('ttft_ms', 'offered_ttft_ms', 'itl_ms', 'e2e_ms', 'offered_e2e_ms'):
            stats[field] = _distribution([r[field] for r in completed if r.get(field) is not None])
        stats['candidate_slo_not_user_accepted'] = {str(seconds): sum(r['offered_ttft_ms'] <= seconds * 1000
            and (r['itl_ms'] is None or r['itl_ms'] <= 100) for r in completed) / elapsed for seconds in (1, 5, 10)}
        result['phases'][phase] = stats
    return result
