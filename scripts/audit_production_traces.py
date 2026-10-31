#!/usr/bin/env python3
"""Audit mirrored original trace semantics; no inference or performance claims."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def _stats(values: list[int]) -> dict:
    ordered = sorted(values)
    def percentile(q: float) -> float:
        rank = (len(ordered) - 1) * q
        low, high = int(rank), math.ceil(rank)
        return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)
    return {'min': min(values), 'mean': statistics.mean(values), 'p50': percentile(.5),
            'p95': percentile(.95), 'p99': percentile(.99), 'max': max(values)}


def audit(base: Path) -> dict:
    report = {'classification': 'confirmed_input_and_loader_audit_only', 'gpu_experiment': False, 'traces': {},
              'sources': ['https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/README.md',
                          'https://github.com/HPMLL/BurstGPT/blob/main/README.md']}
    for path in sorted((base / 'v3_sources').glob('*.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        stamps = [row['timestamp'] for row in rows]
        seen = set()
        reusable = total = 0
        for row in rows:
            prefix = tuple(row['hash_ids'])
            match = 0
            for length in range(1, len(prefix) + 1):
                if prefix[:length] not in seen:
                    break
                match = length
            reusable += min(match, row['input_length'] // 512) * 512
            total += row['input_length']
            seen.update(prefix[:length] for length in range(1, len(prefix) + 1))
        frozen = base / 'v3_frozen' / (path.stem + '_264_seed20261020.jsonl')
        manifest = json.loads(Path(str(frozen) + '.manifest.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['source_sha256']:
            raise ValueError('Original source hash differs from frozen manifest')
        subset = [json.loads(line) for line in frozen.read_text().splitlines()]
        mappings = [offset for offset in (0, 1) if subset == [rows[i - offset] for i in manifest['origin_line_indices']]]
        if len(mappings) != 1 or hashlib.sha256(frozen.read_bytes()).hexdigest() != manifest['subset_sha256']:
            raise ValueError('Frozen subset origin line mapping/hash is ambiguous or invalid')
        report['traces'][path.stem] = {
            'sha256': manifest['source_sha256'], 'requests': len(rows), 'timestamp_unit': 'ms',
            'timestamp_monotonic': all(a <= b for a, b in zip(stamps, stamps[1:])),
            'duration_seconds': (max(stamps) - min(stamps)) / 1000,
            'input_tokens': _stats([r['input_length'] for r in rows]),
            'output_tokens': _stats([r['output_length'] for r in rows]),
            'zero_outputs': sum(r['output_length'] == 0 for r in rows),
            'over_32k_input_plus_output': sum(r['input_length'] + r['output_length'] > 32768 for r in rows),
            'unbounded_previous_prefix_token_fraction': reusable / total,
            'prefix_reference_scope': 'source 512-token blocks with unlimited retention; not engine cache hits',
            'old_264_source_mapping_verified': True, 'origin_line_number_base': mappings[0],
            'old_264_kept_fraction': len(subset) / len(rows)}
    path = base / 'traces/frozen/burstgpt_peak10m.csv'
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    stamps = [int(r['Timestamp']) for r in rows]
    inputs = [int(r['Request tokens']) for r in rows]
    outputs = [int(r['Response tokens']) for r in rows]
    report['burstgpt_peak10m'] = {
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'requests': len(rows),
        'timestamp_unit': 'seconds', 'first_timestamp': min(stamps), 'last_timestamp': max(stamps),
        'span_seconds': max(stamps) - min(stamps), 'declared_window_seconds': 600,
        'offered_rps_over_600s': len(rows) / 600,
        'one_second_peak_requests': max(collections.Counter(stamps).values()),
        'input_tokens': _stats(inputs), 'output_tokens': _stats(outputs), 'zero_outputs': outputs.count(0),
        'model_counts': dict(collections.Counter(r['Model'] for r in rows)),
        'session_ids_available': 'Session ID' in rows[0], 'raw_text_available': False}
    report['required_before_replay'] = [
        'Use contiguous original windows and timestamp normalization; sparse 264-row samples are not full traces',
        'Preserve 512 source-token prefix blocks regardless of xLLM page128; verify retokenized shared prefixes',
        'Preserve open-loop arrivals with adequate inflight allowance; report offered/actual rate and drain time',
        'Disclose missing semantic text and distinguish natural generation from response-length emulation',
        'Account for original failed/zero-output requests and context filtering instead of fabricating successes',
        'Normalize aggregate multi-model demand to a deployment instance explicitly; retain original burst shape']
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.base)
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'report': str(args.out), 'source_traces': len(result['traces']), 'gpu_experiment': False}))


if __name__ == '__main__':
    main()
