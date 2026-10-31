#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Explain final-content versus final-SSE clocks in verified trace records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.controller.trace_serving import _distribution


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(audit_path: Path, mirror: Path, installed_source: Path) -> dict:
    audit = json.loads(audit_path.read_text())
    if audit['classification'] not in ('confirmed_trace_measurement_integrity',
                                      'confirmed_trace_measurement_integrity_runtime_conflict_retained'):
        raise ValueError('Require a previously independently verified trace')
    root = mirror / 'support/agent_work/runs' / audit['run_id']
    result = {'classification': 'confirmed_raw_timing_and_installed_source_analysis', 'run_id': audit['run_id'],
              'audit_sha256': _sha(audit_path), 'script_sha256': _sha(Path(__file__)), 'arms': {},
              'source_sha256': {name: _sha(installed_source / name) for name in (
                  'metrics/types/request_latency_metric.py', 'records/record_processor_service.py',
                  'common/models/record_models.py', 'endpoints/openai_chat.py')},
              'definitions': {'request_latency': 'AIPerf start to last nonempty parsed content response',
                              'request_end_ns': 'Epoch-mapped timestamp of the last raw SSE response, including content-free terminal packets',
                              'offered_e2e_ms': 'Source offered timestamp to last raw SSE response; includes client start lag and content-free terminal tail',
                              'stream_tail_gap_ms': 'Last raw SSE response minus last nonempty content response'},
              'new_GPU_work': False, 'original_metrics_changed': False}
    for arm, evidence in audit['sealed_arms'].items():
        metrics = {row['session_id']: row for row in evidence['metrics']['per_request']}
        raw_path = root / 'output' / arm / 'artifacts/profile_export_raw.jsonl'
        gaps, short, long, inclusive, seen = [], [], [], [], set()
        qualified = {str(seconds): 0 for seconds in (1, 5, 10)}
        top = []
        with raw_path.open() as stream:
            for line in stream:
                raw = json.loads(line); identity = raw['metadata']['conversation_id']; row = metrics[identity]
                if identity in seen:
                    raise ValueError('Repeated raw identity')
                seen.add(identity)
                if row['logical_phase'] != 'measurement' or not row['success']:
                    continue
                responses = raw['responses']; start_perf = raw['start_perf_ns']
                if raw['metadata']['request_end_ns']-raw['metadata']['request_start_ns'] != responses[-1]['perf_ns']-start_perf:
                    raise ValueError('Final raw response and exported end clock differ')
                content_times = []
                for response in responses:
                    for packet in response.get('packets', []):
                        if packet.get('name') != 'data' or packet.get('value') == '[DONE]':
                            continue
                        item = json.loads(packet['value'])
                        for choice in item.get('choices', []):
                            delta = choice.get('delta') or choice.get('message') or {}
                            if delta.get('tool_calls') or delta.get('reasoning_content') or delta.get('reasoning'):
                                raise ValueError('This analysis only handles the observed plain-text anonymous trace protocol')
                            if delta.get('content'):
                                content_times.append(response['perf_ns'])
                if not content_times or abs((content_times[-1]-start_perf)/1e6-row['e2e_ms']) > 1e-8:
                    raise ValueError('Independent final-content timestamp does not reproduce AIPerf request_latency')
                gap = (responses[-1]['perf_ns']-content_times[-1])/1e6
                if abs(row['offered_e2e_ms']-row['client_start_lag_ms']-row['e2e_ms']-gap) > 1e-7:
                    raise ValueError('Offered/content/terminal clock identity failed')
                gaps.append(gap)
                (short if row['output_tokens'] <= 4 else long).append(gap)
                mean = None
                if row['output_tokens'] > 1:
                    mean = ((row['request_end_ns']-row['request_start_ns'])/1e6-row['ttft_ms'])/(row['output_tokens']-1)
                    if abs(mean-(row['itl_ms']+gap/(row['output_tokens']-1))) > 1e-7:
                        raise ValueError('Official ITL and final-SSE-inclusive token-average identity failed')
                    inclusive.append(mean)
                for limit in qualified:
                    qualified[limit] += row['offered_ttft_ms'] <= int(limit)*1000 and (mean is None or mean <= 100)
                top.append({'session_id': identity, 'server_output_tokens': row['output_tokens'],
                            'terminal_minus_content_ms': gap, 'official_request_mean_ITL_ms': row['itl_ms'],
                            'final_SSE_inclusive_per_server_output_token_ms': mean})
        if seen != set(metrics):
            raise ValueError('Missing raw trace record')
        result['arms'][arm] = {'raw_sha256': _sha(raw_path), 'successful_measurement_records': len(gaps),
                               'terminal_minus_content_ms': _distribution(gaps),
                               'OSL_le4_terminal_minus_content_ms': _distribution(short),
                               'OSL_gt4_terminal_minus_content_ms': _distribution(long),
                               'final_SSE_inclusive_per_server_output_token_ms': _distribution(inclusive),
                               'illustrative_TTFT_and_final_SSE_inclusive_mean100ms_qualified_RPS': {
                                   key: value/evidence['metrics']['phases']['measurement']['elapsed_including_drain_seconds']
                                   for key, value in qualified.items()},
                               'largest_terminal_tails': sorted(top, key=lambda item:item['terminal_minus_content_ms'], reverse=True)[:5]}
    result['limitations'] = ['Final SSE is an application observation, not TCP EOF or a hardware wire timestamp',
                             'These timestamps do not identify a GPU/kernel/queue cause for the terminal tail',
                             'Final-SSE-inclusive per-server-token mean is an added completion-time proxy, not hardware step timing or per-token tail latency',
                             'Anonymous ignore-EOS replay can include output tokens without visible content; their exact token IDs were not exported',
                             'Both latency definitions are retained; no metric or original record is rewritten']
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audit', 'mirror', 'installed-source', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.audit, args.mirror, args.installed_source)
    with args.out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result['arms']))


if __name__ == '__main__':
    main()
