#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Analyze an audited full C128 quality pair and its C8 reference sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(batch_path: Path, c8_comparison_path: Path, c8_reference_path: Path) -> dict:
    batch, old, reference = [json.loads(p.read_text()) for p in
                             (batch_path, c8_comparison_path, c8_reference_path)]
    if (batch.get('classification') != 'confirmed_full_production_batch_prediction_integrity_and_rescoring'
            or not batch.get('root_complete') or not batch.get('gpu_release_verified')
            or batch.get('samples_per_arm') != 1719 or not batch.get('full_1719_evaluation')
            or set(batch.get('audited_arms', [])) != {'bf16', 'fp8_e4m3'}
            or batch['production_configuration']['concurrency'] != 128):
        raise ValueError('Require independently verified complete paired C1281719, not pilot or partial')
    if (old.get('classification') != 'confirmed_quality_reference_comparison_across_runs'
            or old.get('records_per_arm') != 1719
            or old.get('reference_audit_sha256') != _sha(c8_reference_path)
            or reference.get('classification') != 'confirmed_full_bf16_reference_prediction_integrity_and_rescoring'
            or reference.get('samples_per_arm') != 1719):
        raise ValueError('C8 reference does not bind the independent full reference audit')
    for key in ('model_weights_sha256', 'model_metadata_sha256', 'versions',
                'protocol_files_sha256', 'evaluator_sha256', 'audited_source_commit',
                'runtime_sources_digest'):
        if batch[key] != reference[key]:
            raise ValueError('C128/C8 reference provenance differs: ' + key)
    new_config, old_config = [dict(d['production_configuration']) for d in (batch, reference)]
    if new_config.pop('concurrency') != 128 or old_config.pop('concurrency') != 8 or new_config != old_config:
        raise ValueError('C128/C8 reference serving settings changed beyond declared client concurrency')
    tasks = ['gsm8k', 'narrativeqa', 'qasper', 'multifieldqa_en', '2wikimqa',
             'hotpotqa', 'musique', 'lcc', 'repobench-p']
    if set(batch['paired_items']) != set(tasks) or set(old['paired_items']) != set(tasks):
        raise ValueError('Missing quality task')
    task_deltas, changes = {}, {}
    for task in tasks:
        new_rows, old_rows = batch['paired_items'][task], old['paired_items'][task]
        count = 1319 if task == 'gsm8k' else 50
        if len(new_rows) != count or len(old_rows) != count:
            raise ValueError('Incomplete original task')
        changes[task] = {arm: {'up': 0, 'down': 0, 'tie': 0} for arm in ('bf16', 'fp8_e4m3')}
        for i, (new, previous) in enumerate(zip(new_rows, old_rows)):
            if (new['index'] != i or previous['index'] != i or new['original_source_index'] != i
                    or str(new['id']) != str(previous['id'])):
                raise ValueError('Original item ID/order differs across C128 and C8')
            for field in (('scores', 'flexible_scores') if task == 'gsm8k' else ('scores',)):
                for arm in ('bf16', 'fp8_e4m3'):
                    value = new[field][arm]
                    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError('Invalid per-item quality score')
            field = 'flexible_scores' if task == 'gsm8k' else 'scores'
            for arm in ('bf16', 'fp8_e4m3'):
                delta = new[field][arm] - previous[field][arm]
                changes[task][arm]['up' if delta > 0 else 'down' if delta < 0 else 'tie'] += 1
        if task == 'gsm8k':
            task_deltas[task] = {suffix: batch['task_scores'][task]['fp8_e4m3_' + suffix] -
                                batch['task_scores'][task]['bf16_' + suffix]
                                for suffix in ('strict_em', 'flexible_em')}
        else:
            task_deltas[task] = batch['task_scores'][task]['fp8_e4m3'] - batch['task_scores'][task]['bf16']
    rng = np.random.default_rng(20261026)
    gsm = np.asarray([r['flexible_scores']['fp8_e4m3'] - r['flexible_scores']['bf16']
                      for r in batch['paired_items']['gsm8k']])
    lb = np.asarray([[r['scores']['fp8_e4m3'] - r['scores']['bf16']
                      for r in batch['paired_items'][task]] for task in tasks[1:]])
    gsm_boot = np.concatenate([gsm[rng.integers(0, 1319, size=(100, 1319))].mean(axis=1)
                               for _ in range(50)])
    lb_boot = np.stack([lb[i][rng.integers(0, 50, size=(5000, 50))].mean(axis=1)
                        for i in range(8)]).mean(axis=0)
    macro = batch['longbench_8_task_macro']; previous_macro = old['longbench_8_task_macro']
    return {'schema': 'kv-production-C128-quality-analysis-v1',
            'classification': 'confirmed_C128_quality_tradeoff_and_C8_descriptive_sensitivity',
            'run_id': batch['run_id'], 'source_commit': batch['audited_source_commit'],
            'batch_audit_sha256': _sha(batch_path), 'c8_reference_audit_sha256': _sha(c8_reference_path),
            'c8_comparison_sha256': _sha(c8_comparison_path), 'script_sha256': _sha(Path(__file__)),
            'samples_per_arm': 1719, 'task_scores': batch['task_scores'], 'task_deltas': task_deltas,
            'longbench_8_task_macro': macro, 'longbench_macro_delta': macro['fp8_e4m3'] - macro['bf16'],
            'paired_flips': batch['paired_flips'], 'length_finish_counts': batch['length_finish_counts'],
            'C128_minus_C8_per_arm_flips': changes,
            'C128_minus_C8_macro': {a: macro[a] - previous_macro[a] for a in ('bf16', 'fp8_e4m3')},
            'C128_minus_C8_task_scores': {task: {key: value - old['task_scores'][task][key]
                                                for key, value in scores.items()}
                                          for task, scores in batch['task_scores'].items()},
            'paired_bootstrap_95ci': {'gsm8k_flexible_delta': np.quantile(gsm_boot, [.025, .975]).tolist(),
                                       'longbench_macro_delta': np.quantile(lb_boot, [.025, .975]).tolist()},
            'bootstrap_draws': 5000, 'bootstrap_seed': 20261026,
            'production_configuration': batch['production_configuration'],
            'model_weights_sha256': batch['model_weights_sha256'],
            'actual_gpu_hours': batch['actual_gpu_hours'], 'quality_accepted': False,
            'user_has_accepted_margin': False,
            'limitations': ['C128 client limit is not a fixed actual GPU batch128',
                           'C8 used a separate BF16 reference and retained FP8 80+1639 fragments; C128 is one full ordered pair',
                           'Same source questions are reused under a changed batching control, not more independent questions',
                           'Cross-run/GPU/cache/scheduling changes are not isolated batch-size causal effects',
                           'Per-item bootstrap describes task sampling, not GPU-run repeatability or equivalence',
                           'Code string similarity is not execution or agent/tool success',
                           'No user quality/SLO margin has been accepted; throughput cannot erase measured quality loss']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('batch-audit', 'c8-comparison', 'c8-reference', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.batch_audit, args.c8_comparison, args.c8_reference)
    with args.out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('classification', 'task_deltas',
                                            'longbench_8_task_macro', 'paired_bootstrap_95ci')}))


if __name__ == '__main__':
    main()
