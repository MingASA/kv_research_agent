#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Compare one predeclared coverage recipe with sealed retained quality references."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_quality_cohort_latency import _rows
from scripts.controller.durable import write_new_json
from scripts.controller.policy import BF16_XQA_OPERATORS
from scripts.controller.protocols import TASKS
from scripts.controller.scale_coverage import BASE_SCALES, SCALES

LABEL = 'FP8_V_COVERAGE'


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(reference: Path, candidate: Path, proof: Path,
            reference_mirror: Path, candidate_mirror: Path) -> dict:
    import numpy as np

    b, c, p = [json.loads(path.read_text()) for path in (reference,candidate,proof)]
    if (b.get('classification') != 'confirmed_full_production_batch_prediction_integrity_and_rescoring'
            or c.get('classification') != 'confirmed_full_scale_coverage_candidate_prediction_integrity_and_rescoring'
            or b.get('audited_arms') != ['bf16','fp8_e4m3'] or c.get('audited_arms') != ['fp8_e4m3']
            or any(x.get('samples_per_arm') != 1719 or not x.get('root_complete')
                   or not x.get('gpu_release_verified') or not x.get('full_1719_evaluation') for x in (b,c))):
        raise ValueError('Both exact full1719 deployment results require independent scoring and completion audits')
    for name in ('model_weights_sha256','model_metadata_sha256','versions','protocol_files_sha256',
                 'reuse_mapping','audited_source_files_sha256'):
        if b[name] != c[name]:
            raise ValueError('Model/protocol/inference-source identity differs: '+name)
    if (p.get('evaluator_and_scoring_source_bytes_identical') is not True
            or p.get('all_inference_source_bytes_identical') is not True
            or b['evaluator_sha256'] != p['baseline_evaluator_sha256']
            or c['evaluator_sha256'] != p['candidate_evaluator_sha256']
            or b['evaluator_sha256'] != c['evaluator_sha256']
            or c.get('scale_quality_provenance_sha256') != _sha(proof)
            or p.get('reference_audit_sha256') != _sha(reference)):
        raise ValueError('Source/scoring equivalence proof does not bind the actual sealed candidate')
    old_cfg, new_cfg = [dict(x['production_configuration']) for x in (b,c)]
    if (old_cfg.pop('fp8_scales') != BASE_SCALES or new_cfg.pop('fp8_scales') != SCALES
            or old_cfg != new_cfg or old_cfg['operators'] != BF16_XQA_OPERATORS
            or old_cfg['concurrency'] != 128):
        raise ValueError('The deployment recipe changed beyond the unique declared V-scale coverage correction')
    scores, deltas, flips = {}, {}, {}
    differences = {'versus_BF16':{},'versus_FP8_XQA':{}}
    identity_records = 0
    for task in TASKS:
        count = 1319 if task == 'gsm8k' else 50
        before, after = b['paired_items'][task], c['paired_items'][task]
        if len(before) != count or len(after) != count or c['reuse_mapping'][task] != list(range(count)):
            raise ValueError('Missing, repeated or changed original task rows')
        groups = {'BF16':_rows(reference_mirror,b,'bf16',task),
                  'FP8_XQA':_rows(reference_mirror,b,'fp8_e4m3',task),
                  LABEL:_rows(candidate_mirror,c,'fp8_e4m3',task)}
        for rows in groups.values():
            if any(x['_record_sha256'] != y['_record_sha256'] or x['_prompt_sha256'] != y['_prompt_sha256']
                    or x['api_response']['usage']['prompt_tokens'] != y['api_response']['usage']['prompt_tokens']
                    for x,y in zip(groups['BF16'],rows)):
                raise ValueError('The original sealed record/prompt/actual prompt-token identity differs')
        identity_records += count
        scores[task], deltas[task] = {}, {}
        for field in (('scores','flexible_scores') if task == 'gsm8k' else ('scores',)):
            columns = {'BF16':[],'FP8_XQA':[],LABEL:[]}
            for index,(x,y) in enumerate(zip(before,after)):
                if (x['index'] != index or y['index'] != index or x['original_source_index'] != index
                        or y['original_source_index'] != index or str(x['id']) != str(y['id'])):
                    raise ValueError('Original global task ID/order differs')
                values = (x[field]['bf16'], x[field]['fp8_e4m3'], y[field]['fp8_e4m3'])
                if any(type(v) not in (int,float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                    raise ValueError('Invalid per-record official task score')
                for label,value in zip(columns,values):columns[label].append(value)
            metric = ('strict_em' if field == 'scores' else 'flexible_em') if task == 'gsm8k' else 'official_score'
            scores[task][metric] = {name:float(np.mean(values)) for name,values in columns.items()}
            for name, arm, source in (('BF16','bf16',b),('FP8_XQA','fp8_e4m3',b),(LABEL,'fp8_e4m3',c)):
                key = arm+'_'+metric if task == 'gsm8k' else arm
                if not math.isclose(scores[task][metric][name],source['task_scores'][task][key],rel_tol=0,abs_tol=1e-12):
                    raise ValueError('Aggregate score differs from independent per-record official scores')
            deltas[task][metric] = {}
            for label, previous in (('versus_BF16','BF16'),('versus_FP8_XQA','FP8_XQA')):
                difference = np.asarray(columns[LABEL])-np.asarray(columns[previous])
                deltas[task][metric][label+'_pp'] = 100*float(difference.mean())
                flips[task+'_'+metric+'_'+label] = dict(collections.Counter(
                    'up' if value > 0 else 'down' if value < 0 else 'tie' for value in difference))
                if task != 'gsm8k' or field == 'flexible_scores':differences[label][task] = difference
    ci = {}
    for label, tasks in differences.items():
        rng = np.random.default_rng(20261026); macro = np.zeros(5000)
        for task in TASKS:
            sample = tasks[task]
            draws = sample[rng.integers(0,len(sample),size=(5000,len(sample)))].mean(axis=1)
            if task == 'gsm8k':gsm = draws
            else:macro += draws/8
        ci[label] = {'GSM_flexible_delta_pp':(100*np.quantile(gsm,[.025,.975])).tolist(),
                     'LongBench_macro_delta_pp':(100*np.quantile(macro,[.025,.975])).tolist()}
    return {'classification':'confirmed_deployment_scale_coverage_quality_comparison_across_runs',
        'reference_run_id':b['run_id'],'candidate_run_id':c['run_id'],'records':identity_records,
        'independent_new_users':0,'reference_audit_sha256':_sha(reference),'candidate_audit_sha256':_sha(candidate),
        'frozen_source_proof_sha256':_sha(proof),'actual_sealed_record_prompt_token_identity_verified':True,
        'all_inference_and_evaluator_source_bytes_identical':True,'candidate_scales':SCALES,
        'task_scores':scores,'task_deltas':deltas,'paired_score_changes':flips,'paired_bootstrap_95ci':ci,
        'LongBench_macro':{name:float(np.mean([scores[t]['official_score'][name] for t in TASKS if t != 'gsm8k']))
                           for name in ('BF16','FP8_XQA',LABEL)},
        'bootstrap':{'draws':5000,'seed':20261026,'LongBench':'paired within-task resampling,equal8-task macro'},
        'cap_finish_counts':{'reference_pair':b['length_finish_counts'],'candidate':c['length_finish_counts']},
        'model_weights_sha256':c['model_weights_sha256'],'quality_accepted':False,'new_GPU_service_runs':0,
        'limits':['Same inference source/recipe apart from Vscale; different runs/GPU/cache/scheduling do not isolate all numeric causality',
            'Same benchmark questions reused; conditional paired intervals are not new independent users or production-population guarantees',
            'Official code similarity is not code execution; no streaming/SLO/service benefit inferred or inherited',
            'Coverage of rare saturation is not proof that saturation caused observed overall quality loss']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference','candidate','proof','reference-mirror','candidate-mirror','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    result = compare(args.reference,args.candidate,args.proof,args.reference_mirror,args.candidate_mirror)
    write_new_json(args.out,result)
    print(json.dumps({k:result[k] for k in ('classification','records','LongBench_macro','paired_bootstrap_95ci')}))


if __name__ == '__main__':main()
