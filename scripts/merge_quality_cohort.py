#!/usr/bin/env python3
"""Merge verified, disjoint native quality predictions and rescore all original 1,719 records."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_quality_run import audit
from scripts.controller.protocols import TASKS, restore_quality_records


def merge(pilot_mirror: Path, completion_mirror: Path, pilot_id: str, completion_id: str,
          batch: Path, source_repo: Path, tokenizer: Path,
          missing_mirror: Path | None = None, missing_id: str | None = None) -> dict:
    import numpy as np
    sys.path.insert(0, str(source_repo))
    from tools import eval_kv_cache_quality as evaluator

    if (missing_mirror is None) != (missing_id is None):
        raise ValueError('Missing-arm recovery requires both the exact mirror and runID')
    recovered = missing_mirror is not None
    verified = [audit(pilot_mirror, pilot_id, source_repo, tokenizer),
                audit(completion_mirror, completion_id, source_repo, tokenizer,
                      'fp8_e4m3' if recovered else None, recovered)]
    fragments = [(pilot_mirror, pilot_id), (completion_mirror, completion_id)]
    if recovered:
        verified.append(audit(missing_mirror, missing_id, source_repo, tokenizer, 'bf16'))
        fragments.append((missing_mirror, missing_id))
    if [v['samples_per_arm'] for v in verified] != ([80, 1639, 1639] if recovered else [80, 1639]):
        raise ValueError('Expected the verified80 and its exact1639 completion')
    for fragment in verified[1:]:
        for key in ('model_weights_sha256', 'runtime_sources_digest', 'model_metadata_sha256', 'evaluator_sha256',
                    'versions', 'production_configuration', 'protocol_files_sha256'):
            if verified[0][key] != fragment[key]:
                raise ValueError('Completion cannot reuse quality from a different implementation/protocol: ' + key)
    if hashlib.sha256((source_repo / 'tools/eval_kv_cache_quality.py').read_bytes()).hexdigest() != verified[0]['evaluator_sha256']:
        raise ValueError('Independent evaluator source changed from the real model run')
    records, original_manifest = restore_quality_records(batch)
    collected = {arm: {task: {} for task in TASKS} for arm in ('bf16', 'fp8_e4m3')}
    prompt_counts = {task: {} for task in TASKS}
    pilot_request = None
    for (mirror, run_id), evidence in zip(fragments, verified):
        root = mirror / 'support/agent_work/runs' / run_id
        request = json.loads((root / 'ticket.json').read_text())['request']
        manifest_path = root / Path(request['dataset_path']).relative_to(request['remote_output'])
        manifest = json.loads(manifest_path.read_text())
        if any(manifest['source_task_sha256'][task] != original_manifest['sources'][task]['sha256'] for task in TASKS):
            raise ValueError('Predictions do not refer to the frozen original population')
        if pilot_request is None:
            pilot_request = request
        elif (request['workload']['parent_result_sha256'] != verified[0]['terminal_result_sha256']
              or manifest['parent_pilot_input_sha256'] != pilot_request['dataset_sha256']):
            raise ValueError('Completion refers to a different parent input/result')
        if recovered and run_id == missing_id:
            guard = request['workload']['recovery']
            if (guard['prior_run_id'] != completion_id or guard['prior_terminal_result_sha256'] != verified[1]['terminal_result_sha256']
                    or guard['arm_measurement_sha256'] != verified[1]['arm_measurement_sha256']['fp8_e4m3']):
                raise ValueError('Missing BF16 refers to a different recovered FP8 fragment')
        for arm in evidence['audited_arms']:
            label = 'auto' if arm == 'bf16' else 'fp8-e4m3'
            quality = root / 'output' / arm / 'quality'
            provenance = json.loads((quality / 'experiment_manifest.json').read_text())
            for task in TASKS:
                for line in (quality / 'runs' / task / (label + '.jsonl')).read_text().splitlines():
                    row = json.loads(line)
                    index = evidence['reuse_mapping'][task][row['_eval_index']]
                    if index in collected[arm][task]:
                        raise ValueError('Repeated original sample across pilot/completion')
                    expected = hashlib.sha256(json.dumps(records[task][index], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                    if expected != row['_record_sha256']:
                        raise ValueError('Prediction refers to a different original record')
                    collected[arm][task][index] = row
                    count = provenance['prepared'][task]['prompt_tokens'][row['_eval_index']]
                    if index in prompt_counts[task] and prompt_counts[task][index] != count:
                        raise ValueError('Paired real prompt lengths differ')
                    prompt_counts[task][index] = count
    first_root = pilot_mirror / 'support/agent_work/runs' / pilot_id
    scorer = evaluator._longbench_assets(first_root / 'input/scorer')[2]
    scores, pairs, length_finishes = {}, {}, {}
    for task in TASKS:
        count = 1319 if task == 'gsm8k' else 50
        for arm in collected:
            if set(collected[arm][task]) != set(range(count)):
                raise ValueError('Full original cohort has missing/unknown records')
        scores[task], pairs[task] = evaluator._score_pair(task, records[task], {
            arm: [collected[arm][task][i]['prediction'] for i in range(count)] for arm in collected}, scorer)
        length_finishes[task] = {arm: sum(collected[arm][task][i]['api_response']['choices'][0]['finish_reason'] == 'length'
                                        for i in range(count)) for arm in collected}
    macro = {arm: sum(scores[task][arm] for task in TASKS if task != 'gsm8k') / 8 for arm in collected}
    rng = np.random.default_rng(20261026)
    gsm_diff = np.asarray([row['flexible_scores']['fp8_e4m3'] - row['flexible_scores']['bf16'] for row in pairs['gsm8k']])
    lb_diff = np.asarray([[row['scores']['fp8_e4m3'] - row['scores']['bf16'] for row in pairs[task]] for task in TASKS if task != 'gsm8k'])
    gsm_boot = np.concatenate([gsm_diff[rng.integers(0, 1319, size=(100, 1319))].mean(axis=1) for _ in range(50)])
    lb_boot = np.stack([lb_diff[i][rng.integers(0, 50, size=(5000, 50))].mean(axis=1) for i in range(8)]).mean(axis=0)
    intervals = {'gsm8k_flexible_delta': np.quantile(gsm_boot, [.025, .975]).tolist(),
                 'longbench_macro_delta': np.quantile(lb_boot, [.025, .975]).tolist()}
    candidate_margin = -.01
    supported = all(v[0] >= candidate_margin for v in intervals.values())
    loss = any(v[1] < candidate_margin for v in intervals.values())
    by_length = {}
    for low, high in ((0, 4096), (4096, 8192), (8192, 16384), (16384, 24576), (24576, 32769)):
        selected = [(task, i) for task in TASKS if task != 'gsm8k' for i in range(50)
                    if low <= prompt_counts[task][i] < high]
        if selected:
            by_length[str(low) + '-' + str(high)] = {'n': len(selected), **{
                arm: sum(pairs[task][i]['scores'][arm] for task, i in selected) / len(selected) for arm in collected}}
    return {'schema': 'kv-native-full-frozen-quality-v1', 'classification': 'confirmed_full_frozen_cohort_independent_rescoring',
            'records_per_arm': 1719, 'gsm8k_n': 1319, 'longbench_n_per_task': 50,
            'pilot_run': pilot_id, 'completion_run': completion_id, 'task_scores': scores,
            'missing_bf16_recovery_run': missing_id,
            'fragment_terminal_statuses_preserved': {v['run_id']:v['terminal_status_preserved'] for v in verified},
            'longbench_8_task_macro': macro, 'paired_items': pairs,
            'paired_flips': {task: dict(collections.Counter(row.get('flexible_flip', row['flip']) for row in items)) for task, items in pairs.items()},
            'length_finish_counts': length_finishes, 'longbench_by_prompt_length': by_length,
            'paired_bootstrap_95ci': intervals, 'bootstrap_seed': 20261026, 'bootstrap_draws': 5000,
            'candidate_engineering_noninferiority_margin': .01, 'user_has_accepted_margin': False,
            'engineering_assessment': 'supported_for_these_tasks' if supported else 'loss_above_candidate_margin' if loss else 'inconclusive',
            'provenance': {key: verified[0][key] for key in ('model_weights_sha256', 'runtime_sources_digest',
                'model_metadata_sha256', 'evaluator_sha256', 'production_configuration', 'versions', 'protocol_files_sha256')},
            'quality_gpu_hours': sum(v['actual_gpu_hours'] for v in verified),
            'limitations': ['Full frozen1319+8x50 cohort, not all of LongBench or every production task',
                           'Bootstrap is paired and stratified by task; future-production generalization is conditional',
                           'The1pp engineering margin is a proposal, not user SLO acceptance',
                           'Quality alone does not complete the production support decision or real-trace service evaluation']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot-mirror', 'completion-mirror', 'batch', 'source-repo', 'tokenizer', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--pilot-id', required=True); parser.add_argument('--completion-id', required=True)
    parser.add_argument('--missing-mirror', type=Path); parser.add_argument('--missing-id')
    args = parser.parse_args()
    result = merge(args.pilot_mirror, args.completion_mirror, args.pilot_id, args.completion_id,
                   args.batch, args.source_repo, args.tokenizer, args.missing_mirror, args.missing_id)
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('records_per_arm', 'task_scores', 'longbench_8_task_macro', 'engineering_assessment')}))


if __name__ == '__main__':
    main()
