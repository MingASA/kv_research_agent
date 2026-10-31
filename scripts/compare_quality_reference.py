#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Compare a new full BF16-XQA reference with already verified FP8 predictions.

This is a comparison of deployment references across runs, never a claim that
FP8 predictions were regenerated in the new single-arm reference run.
"""

from __future__ import annotations

import argparse
import ast
import collections
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.controller.policy import BF16_XQA_OPERATORS, XQA_OPERATORS
from scripts.controller.protocols import TASKS, restore_quality_records

CANDIDATE_AUDIT_SHA = '7caebece84319d1bc21a612c13eb9acd1bb2a5179357f0daccbc06a97940d39e'


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluation_logic(path: Path) -> str:
    tree = ast.parse(path.read_text())
    # The only permitted difference in the official evaluator is recognition of
    # the new strict server attestation profile. Scoring, extraction, prompts and
    # all other code must remain identical.
    loader = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_load_server_manifest']
    if len(loader) != 1:
        raise ValueError('Missing unique strict manifest loader')
    for node in ast.walk(loader[0]):
        if isinstance(node, ast.Tuple):
            node.elts = [item for item in node.elts if not (
                isinstance(item, ast.Constant) and item.value in (
                    'native-flashinfer-bf16-xqa-reference-v1','native-flashinfer-fp8-bf16q-decode-v1'))]
    return ast.dump(tree, include_attributes=False)


def _combine_verified(reference: dict, candidate: dict) -> dict:
    if (reference.get('classification') != 'confirmed_full_bf16_reference_prediction_integrity_and_rescoring'
            or reference.get('samples_per_arm') != 1719 or reference.get('audited_arms') != ['bf16']
            or not reference.get('full_1719_evaluation')
            or candidate.get('classification') != 'confirmed_full_frozen_cohort_independent_rescoring'
            or candidate.get('records_per_arm') != 1719):
        raise ValueError('Require independently verified full cohorts, not pilot or planned predictions')
    provenance = candidate['provenance']
    for key in ('model_weights_sha256', 'model_metadata_sha256', 'versions', 'protocol_files_sha256'):
        if reference[key] != provenance[key]:
            raise ValueError('Quality model/tokenizer/runtime/scoring identity differs: ' + key)
    old_config = copy.deepcopy(provenance['production_configuration'])
    new_config = copy.deepcopy(reference['production_configuration'])
    if old_config.pop('operators') != XQA_OPERATORS or new_config.pop('operators') != BF16_XQA_OPERATORS:
        raise ValueError('The intended BF16 operator change or retained FP8 candidate differs')
    if old_config != new_config:
        raise ValueError('Quality serving settings differ beyond the declared BF16 operator')
    if set(reference['paired_items']) != set(TASKS) or set(candidate['paired_items']) != set(TASKS):
        raise ValueError('Missing quality task')
    pairs, scores = {}, {}
    for task in TASKS:
        new, old = reference['paired_items'][task], candidate['paired_items'][task]
        count = 1319 if task == 'gsm8k' else 50
        if (len(new) != count or len(old) != count
                or reference['reuse_mapping'][task] != list(range(count))):
            raise ValueError('Original record count/order differs')
        pairs[task] = []
        for index, (bf16, fp8) in enumerate(zip(new, old)):
            if (bf16['original_source_index'] != index or fp8['index'] != index
                    or str(bf16['id']) != str(fp8['id'])):
                raise ValueError('Quality records do not pair by original identity')
            item = {'index': index, 'id': fp8['id'], 'scores': {
                'bf16': bf16['scores']['bf16'], 'fp8_e4m3': fp8['scores']['fp8_e4m3']}}
            if task == 'gsm8k':
                item['flexible_scores'] = {'bf16': bf16['flexible_scores']['bf16'],
                                            'fp8_e4m3': fp8['flexible_scores']['fp8_e4m3']}
            for name, flip in [('scores', 'flip')] + ([('flexible_scores', 'flexible_flip')] if task == 'gsm8k' else []):
                values = item[name]
                if any(type(value) not in (int, float) or not 0 <= value <= 1 for value in values.values()):
                    raise ValueError('Invalid independent per-record score')
                item[flip] = ('up' if values['fp8_e4m3'] > values['bf16'] else
                              'down' if values['fp8_e4m3'] < values['bf16'] else 'tie')
            pairs[task].append(item)
        scores[task] = ({arm + suffix: sum(row[field][arm] for row in pairs[task]) / count
                        for arm in ('bf16', 'fp8_e4m3')
                        for suffix, field in (('_strict_em', 'scores'), ('_flexible_em', 'flexible_scores'))}
                       if task == 'gsm8k' else {arm: sum(row['scores'][arm] for row in pairs[task]) / count
                                                for arm in ('bf16', 'fp8_e4m3')})
        expected_fp8 = {key: value for key, value in candidate['task_scores'][task].items() if key.startswith('fp8_e4m3')}
        if any(abs(scores[task][key] - value) > 1e-12 for key, value in expected_fp8.items()):
            raise ValueError('Retained FP8 scores changed during comparison')
    return {'task_scores': scores, 'paired_items': pairs,
            'paired_flips': {task: dict(collections.Counter(row.get('flexible_flip', row['flip']) for row in rows))
                             for task, rows in pairs.items()},
            'longbench_8_task_macro': {arm: sum(scores[task][arm] for task in TASKS if task != 'gsm8k') / 8
                                      for arm in ('bf16', 'fp8_e4m3')}}


def compare(reference_path: Path, candidate_path: Path, reference_mirror: Path, batch: Path,
            reference_source: Path, candidate_source: Path) -> dict:
    import numpy as np

    if _sha(candidate_path) != CANDIDATE_AUDIT_SHA:
        raise ValueError('Original FP8 full-cohort audit changed')
    reference, candidate = json.loads(reference_path.read_text()), json.loads(candidate_path.read_text())
    result = _combine_verified(reference, candidate)
    reference_eval = reference_source / 'tools/eval_kv_cache_quality.py'
    candidate_eval = candidate_source / 'tools/eval_kv_cache_quality.py'
    if (_sha(reference_eval) != reference['evaluator_sha256']
            or _sha(candidate_eval) != candidate['provenance']['evaluator_sha256']
            or _evaluation_logic(reference_eval) != _evaluation_logic(candidate_eval)):
        raise ValueError('Independent scoring/extraction logic changed')
    root = reference_mirror / 'support/agent_work/runs' / reference['run_id']
    request = json.loads((root / 'ticket.json').read_text())['request']
    manifest = json.loads((root / Path(request['dataset_path']).relative_to(request['remote_output'])).read_text())
    records, original = restore_quality_records(batch)
    for task in TASKS:
        actual = [json.loads(line) for line in (root / 'input/cohort' / (task + '.jsonl')).read_text().splitlines()]
        if actual != records[task] or manifest['source_task_sha256'][task] != original['sources'][task]['sha256']:
            raise ValueError('New reference does not use the exact original frozen records')
    rng = np.random.default_rng(20261026)
    gsm = np.asarray([row['flexible_scores']['fp8_e4m3']-row['flexible_scores']['bf16'] for row in result['paired_items']['gsm8k']])
    lb = np.asarray([[row['scores']['fp8_e4m3']-row['scores']['bf16'] for row in result['paired_items'][task]]
                     for task in TASKS if task != 'gsm8k'])
    gsm_boot = np.concatenate([gsm[rng.integers(0, 1319, size=(100, 1319))].mean(axis=1) for _ in range(50)])
    lb_boot = np.stack([lb[i][rng.integers(0, 50, size=(5000, 50))].mean(axis=1) for i in range(8)]).mean(axis=0)
    result.update(schema='kv-cross-deployment-quality-reference-comparison-v1',
                  classification='confirmed_quality_reference_comparison_across_runs', records_per_arm=1719,
                  reference_run=reference['run_id'], new_FP8_predictions=0, same_paired_GPU_run=False,
                  reference_audit_sha256=_sha(reference_path), retained_candidate_audit_sha256=_sha(candidate_path),
                  evaluation_logic_identical_except_strict_manifest_profile=True,
                  source_commits={'new_BF16': reference['audited_source_commit'],
                                  'retained_FP8': 'ede2b9e7c61573825645a6b710a7534d0091bac2'},
                  paired_bootstrap_95ci={'gsm8k_flexible_delta': np.quantile(gsm_boot, [.025, .975]).tolist(),
                                         'longbench_macro_delta': np.quantile(lb_boot, [.025, .975]).tolist()},
                  bootstrap_draws=5000, bootstrap_seed=20261026, quality_accepted=False,
                  user_has_accepted_margin=False,
                  limitations=['New BF16 deployment reference and retained FP8 predictions are different real runs',
                               'Original FP8 used80+1639 fragments; new BF16 uses full1719 in source order, with different cache/batch history',
                               'Greedy decoding does not guarantee identical outcomes across scheduling/numerical paths',
                               'Per-record bootstrap describes task sampling, not independent GPU-run repeatability',
                               'C8 benchmark proxies do not certify all production batches or user quality tolerance'])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-audit', 'candidate-audit', 'reference-mirror', 'batch', 'reference-source', 'candidate-source', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference_audit, args.candidate_audit, args.reference_mirror, args.batch,
                     args.reference_source, args.candidate_source)
    with args.out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ('classification', 'task_scores', 'longbench_8_task_macro')}))


if __name__ == '__main__':
    main()
