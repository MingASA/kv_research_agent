#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Join independently audited original1719 deployment results across runs."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.controller.durable import write_new_json
from scripts.controller.policy import BF16Q_DECODE_OPERATORS,BF16_XQA_OPERATORS
from scripts.controller.protocols import TASKS


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(reference_path: Path,candidate_path: Path,logic_path: Path) -> dict:
    import numpy as np

    b,c,logic=[json.loads(p.read_text()) for p in (reference_path,candidate_path,logic_path)]
    if (b.get('classification')!='confirmed_full_production_batch_prediction_integrity_and_rescoring'
            or c.get('classification')!='confirmed_full_BF16Q_candidate_prediction_integrity_and_rescoring'
            or b.get('audited_arms')!=['bf16','fp8_e4m3'] or c.get('audited_arms')!=['fp8_e4m3']
            or any(x.get('samples_per_arm')!=1719 or not x.get('root_complete') or not x.get('gpu_release_verified')
                   or not x.get('full_1719_evaluation') for x in (b,c))):
        raise ValueError('Both deployment results require independent complete original1719 audits')
    for key in ('model_weights_sha256','model_metadata_sha256','versions','protocol_files_sha256','reuse_mapping'):
        if b[key]!=c[key]:raise ValueError('Original quality provenance differs: '+key)
    if (logic.get('only_new_strict_profile_literal_diff') is not True
            or b['evaluator_sha256']!=logic['baseline_evaluator_sha256']
            or c['evaluator_sha256']!=logic['candidate_evaluator_sha256']):
        raise ValueError('Scoring/extraction/prompt equivalence proof does not bind both actual evaluators')
    bc,cc=[dict(x['production_configuration']) for x in (b,c)]
    if bc.pop('operators')!=BF16_XQA_OPERATORS or cc.pop('operators')!=BF16Q_DECODE_OPERATORS or bc!=cc or bc['concurrency']!=128:
        raise ValueError('Serving settings changed beyond the declared deployment operator profile')
    old,new=[x['audited_source_files_sha256'] for x in (b,c)]
    allowed={'xllm/python/attention/flashinfer.py','xllm/python/attention/native_profile.py'}
    if set(old)!=set(new) or any(old[n]!=new[n] for n in old if n not in allowed):
        raise ValueError('Runtime architecture/scheduler changed outside the explicit attention candidate')
    if set(b['paired_items'])!=set(TASKS) or set(c['paired_items'])!=set(TASKS):raise ValueError('Missing task')
    scores={};deltas={};arrays={'versus_BF16':{},'versus_FP8_XQA':{}};flips={}
    for task in TASKS:
        count=1319 if task=='gsm8k' else 50;before,after=b['paired_items'][task],c['paired_items'][task]
        if len(before)!=count or len(after)!=count or c['reuse_mapping'][task]!=list(range(count)):
            raise ValueError('Original global records omitted or duplicated')
        fields=('scores','flexible_scores') if task=='gsm8k' else ('scores',)
        scores[task]={};deltas[task]={}
        for field in fields:
            columns={'BF16':[],'FP8_XQA':[],'FP8_BF16Q':[]}
            for i,(x,y) in enumerate(zip(before,after)):
                if (x['index']!=i or y['index']!=i or x['original_source_index']!=i or y['original_source_index']!=i
                        or str(x['id'])!=str(y['id'])):raise ValueError('Original item ID/order differs')
                values=(x[field]['bf16'],x[field]['fp8_e4m3'],y[field]['fp8_e4m3'])
                if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in values):
                    raise ValueError('Invalid individual task score')
                for name,value in zip(columns,values):columns[name].append(value)
            metric=('strict_em' if field=='scores' else 'flexible_em') if task=='gsm8k' else 'official_score'
            scores[task][metric]={k:float(np.mean(v)) for k,v in columns.items()}
            for label,arm,source in (('BF16','bf16',b),('FP8_XQA','fp8_e4m3',b),('FP8_BF16Q','fp8_e4m3',c)):
                key=arm+'_'+metric if task=='gsm8k' else arm
                if not math.isclose(scores[task][metric][label],source['task_scores'][task][key],abs_tol=1e-12):
                    raise ValueError('Published score differs from the actual per-item scores')
            deltas[task][metric]={}
            for label,previous in (('versus_BF16','BF16'),('versus_FP8_XQA','FP8_XQA')):
                differences=np.asarray(columns['FP8_BF16Q'])-np.asarray(columns[previous])
                deltas[task][metric][label+'_pp']=100*float(differences.mean())
                flips[task+'_'+metric+'_'+label]=dict(collections.Counter(
                    'up' if v>0 else 'down' if v<0 else 'tie' for v in differences))
                if task!='gsm8k' or field=='flexible_scores':arrays[label][task]=differences
    macro={name:float(np.mean([scores[t]['official_score'][name] for t in TASKS if t!='gsm8k']))
           for name in ('BF16','FP8_XQA','FP8_BF16Q')}
    ci={}
    for label,data in arrays.items():
        rng=np.random.default_rng(20261026);lb=np.zeros(5000)
        for task in TASKS:
            sample=data[task];draws=sample[rng.integers(0,len(sample),size=(5000,len(sample)))].mean(axis=1)
            if task=='gsm8k':gsm=draws
            else:lb+=draws/8
        ci[label]={'GSM_flexible_delta_pp':(np.quantile(gsm,[.025,.975])*100).tolist(),
                   'LongBench_macro_delta_pp':(np.quantile(lb,[.025,.975])*100).tolist()}
    return {'classification':'confirmed_deployment_quality_comparison_across_runs_not_Q_causality',
        'reference_run_id':b['run_id'],'candidate_run_id':c['run_id'],'records':1719,'independent_new_users':0,
        'reference_audit_sha256':_sha(reference_path),'candidate_audit_sha256':_sha(candidate_path),
        'scoring_logic_proof_sha256':_sha(logic_path),'task_scores':scores,'task_deltas':deltas,
        'LongBench_macro':macro,'paired_score_changes':flips,'paired_bootstrap_95ci':ci,
        'bootstrap':{'draws':5000,'seed':20261026,'LongBench':'resample each selected task separately then equal8-task mean'},
        'cap_finish_counts':{'reference_pair':b['length_finish_counts'],'candidate':c['length_finish_counts']},
        'model_weights_sha256':c['model_weights_sha256'],'quality_accepted':False,
        'limits':['Different runs/cache history/scheduling and attention kernels; no isolated Q-causality',
                  'Same fixed benchmark cohort; bootstrap is conditional, not a production population guarantee',
                  'Code similarity is not code execution; no new service benefit measured; no user numeric threshold accepted']}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('reference','candidate','logic-proof','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();r=compare(a.reference,a.candidate,a.logic_proof);write_new_json(a.out,r)
    print(json.dumps({k:r[k] for k in ('classification','LongBench_macro','paired_bootstrap_95ci')}))


if __name__=='__main__':main()
