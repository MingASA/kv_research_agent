#!/usr/bin/env python3
"""Freeze only the 1,639 quality records absent from the original 80-record pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from controller.durable import canonical_bytes, secure_directory, write_new_bytes, write_new_json
from controller.protocols import TASKS, restore_quality_records, sha


def prepare(batch: Path, pilot: Path, out: Path) -> dict:
    records, source = restore_quality_records(batch)
    parent = json.loads((pilot / 'pilot_manifest.json').read_text())
    if sha(batch / 'shard_manifest.json') != parent['parent_shard_manifest_sha256']:
        raise ValueError('Pilot does not originate from this frozen full cohort')
    selected = {}
    for task in TASKS:
        if sha(pilot / (task + '.jsonl')) != parent['data_sha256'][task]:
            raise ValueError('Pilot input changed; do not skip or regenerate records')
        indices = parent['selected_source_indices'][task]
        rows = [json.loads(line) for line in (pilot / (task + '.jsonl')).read_text().splitlines()]
        if rows != [records[task][index] for index in indices]:
            raise ValueError('Pilot original record mapping differs')
        selected[task] = [index for index in range(len(records[task])) if index not in set(indices)]
        if set(indices).intersection(selected[task]) or set(indices).union(selected[task]) != set(range(len(records[task]))):
            raise ValueError('Completion would repeat or omit frozen records')
    if sum(map(len, selected.values())) != 1639:
        raise ValueError('Expected exactly 1,719 original records minus the 80-record pilot')
    if out.exists():
        raise FileExistsError('Completion data requires a fresh immutable directory')
    secure_directory(out)
    manifest = {'schema': 'kv-quality-completion-fragment-v1', 'classification': 'proposal',
                'scope': 'evaluation_fragment_no_full_quality_claim', 'samples_per_arm': 1639,
                'original_total_records': 1719, 'parent_pilot_input_sha256': sha(pilot / 'pilot_manifest.json'),
                'parent_shard_manifest_sha256': sha(batch / 'shard_manifest.json'),
                'source_task_sha256': {task: source['sources'][task]['sha256'] for task in TASKS},
                'source_indices': selected, 'parent_pilot_source_indices': parent['selected_source_indices'],
                'parent_results_must_be_independently_verified_before_launch': True,
                'model_predictions_generated': False, 'gpu_used': False, 'data_sha256': {},
                'preparation_script_sha256': sha(Path(__file__))}
    for task in TASKS:
        data = b''.join(canonical_bytes(records[task][index]) for index in selected[task])
        write_new_bytes(out / (task + '.jsonl'), data)
        manifest['data_sha256'][task] = hashlib.sha256(data).hexdigest()
    write_new_json(out / 'completion_manifest.json', manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.batch, args.pilot, args.out)
    print(json.dumps({'out': str(args.out), 'samples_per_arm': result['samples_per_arm'],
                      'counts': {task: len(indices) for task, indices in result['source_indices'].items()},
                      'gpu_used': False, 'parent_predictions_not_yet_accepted': True}))


if __name__ == '__main__':
    main()
