#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Describe bounded production component traces without claiming causal ratios."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.controller.durable import write_new_json


def summarize(path: Path) -> dict:
    analysis = json.loads(path.read_text())
    if analysis['classification'] != 'confirmed_diagnostic_activity_times_not_headline_performance':
        raise ValueError('Require audited diagnostic time analysis')
    arms, indexed = {}, {}
    for arm, windows in analysis['arms'].items():
        prefix = 'bf16' if arm == 'bf16' else 'fp8'
        components, cpu, rows = collections.Counter(), collections.Counter(), []
        index = collections.defaultdict(list)
        for window in windows:
            for name, value in window['components'].items():
                label = name.replace('xllm.component.' + prefix + '.', '').replace('xllm.component.', '')
                components[label] += value['cuda_kernel_sum_ms']
                cpu[label] += value['cpu_scope_sum_ms']
            for i, step in enumerate(window['forward_batches']):
                values = step['kv_context_tokens']
                if (len(values) != step['sequences'] + 1 or values[0] != 0
                        or any(not 0 < right - left <= 32768 for left, right in zip(values, values[1:]))):
                    raise ValueError('CUDA cumulative KV length vector changed')
                row = {**step, 'window': window['index'], 'forward_index': i,
                       'kv_lengths_including_current_query': [b - a for a, b in zip(values, values[1:])]}
                rows.append(row)
                key = (step['prefill'], step['chunked_prefill'], step['input_tokens'], step['sequences'], tuple(values))
                index[key].append(row)
        prefill = [row for row in rows if row['prefill'] or row['chunked_prefill']]
        decode = [row for row in rows if not (row['prefill'] or row['chunked_prefill'])]
        kernel = sum(window['cuda_kernel_sum_ms'] for window in windows)
        forward = sum(row['correlated_cuda_kernel_sum_ms'] for row in rows)
        outside = sum(window['outside_recorded_forward_cuda_kernel_sum_ms'] for window in windows)
        if abs(kernel - forward - outside) > .001:
            raise ValueError('Forward attribution lost actual kernel time')
        arms[arm] = {'windows': len(windows), 'forwards': len(rows), 'prefill_forwards': len(prefill),
            'prefill_input_tokens': sum(row['input_tokens'] for row in prefill), 'Graph_decode_forwards': len(decode),
            'cuda_kernel_sum_ms': kernel, 'outside_recorded_forward_cuda_kernel_sum_ms': outside,
            'cuda_all_activity_busy_union_ms': sum(window['cuda_all_activity_busy_union_ms'] for window in windows),
            'forward_correlated_kernel_coverage': forward / kernel, 'component_kernel_sum_ms': dict(components),
            'CPU_scope_sum_ms': dict(cpu), 'writer_share_of_all_kernel_time': components['kv_write'] / kernel,
            'prefill_attention_share_of_all_kernel_time': components['prefill_attention'] / kernel,
            'decode_kernel_ms_per_forward_median': statistics.median(row['correlated_cuda_kernel_sum_ms'] for row in decode),
            'forward_batches': rows}
        indexed[arm] = index
    matches = [{'recorded_metadata': [*key[:4], list(key[4])],
                'bf16': indexed['bf16'][key], 'fp8_e4m3': indexed['fp8_e4m3'][key]}
               for key in sorted(indexed['bf16'].keys() & indexed['fp8_e4m3'].keys()) if key[0] or key[1]]
    return {'classification': 'confirmed_descriptive_diagnostic_not_causal_speedup', 'run_id': analysis['run_id'],
        'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'arms': arms,
        'matched_recorded_batch_metadata': matches,
        'limitations': ['Different scheduled query/context batches; aggregate component ratios are not causal',
            'Raw kv_context_tokens is the CUDA cumulative KV vector; adjacent differences recover sequence lengths',
            'KV lengths include current query, not just the previously cached prefix',
            'One equal-metadata batch is not an independent repeated kernel benchmark or exact tensor identity',
            'Profiler overhead; Graph time is whole-model; no hardware HBM bytes or headline performance claim']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.input)
    write_new_json(args.out, result)
    print(json.dumps({'run_id': result['run_id'], 'matched_metadata_batches': len(result['matched_recorded_batch_metadata'])}))


if __name__ == '__main__':
    main()
