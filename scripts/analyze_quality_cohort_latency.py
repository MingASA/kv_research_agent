#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Reuse sealed quality-request clocks for a bounded cross-run description.

These finite buffered cohorts are not an AIPerf steady-state serving test.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.controller.durable import write_new_json
from scripts.controller.production import _jsonl_rows
from scripts.controller.protocols import TASKS


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rows(mirror: Path,audit: dict,arm: str,task: str) -> list:
    root=mirror/'support/agent_work/runs'/audit['run_id'];out=root/'output'/arm
    label='auto' if arm=='bf16' else 'fp8-e4m3';relative='quality/runs/'+task+'/'+label+'.jsonl'
    measurement=json.loads((out/'measurement.json').read_text())
    if (_sha(root/'result.json')!=audit['terminal_result_sha256']
            or _sha(out/'measurement.json')!=audit['arm_measurement_sha256'][arm]
            or _sha(out/relative)!=measurement['evidence_sha256'][relative]):
        raise ValueError('Latency source is not bound to the independently verified original seals')
    rows=sorted(_jsonl_rows(out/relative),key=lambda x:x['_eval_index'])
    if [r['_eval_index'] for r in rows]!=list(range(1319 if task=='gsm8k' else 50)):
        raise ValueError('Finite cohort omitted/duplicated original records')
    return rows


def _describe(rows: list) -> dict:
    import numpy as np

    lat=[r['client_latency_seconds'] for r in rows]
    starts=[datetime.datetime.fromisoformat(r['request_started_utc']) for r in rows]
    ends=[datetime.datetime.fromisoformat(r['request_finished_utc']) for r in rows]
    if any(not math.isfinite(x) or x<=0 for x in lat) or any(e<=s for s,e in zip(starts,ends)):
        raise ValueError('Missing positive actual client clock')
    span=(max(ends)-min(starts)).total_seconds();usage=[r['api_response']['usage'] for r in rows]
    outputs=[u['completion_tokens'] for u in usage];prompts=[u['prompt_tokens'] for u in usage]
    if any(type(v) is not int or v<=0 for v in outputs+prompts):raise ValueError('Invalid actual usage')
    return {'completed_requests':len(rows),'first_request_to_last_response_seconds':span,
        'arrival_span_seconds':(max(starts)-min(starts)).total_seconds(),
        'finite_cohort_completed_RPS':len(rows)/span,'actual_output_tokens':sum(outputs),
        'finite_cohort_actual_output_TPS':sum(outputs)/span,'mean_output_tokens':statistics.mean(outputs),
        'mean_prompt_tokens':statistics.mean(prompts),'client_complete_latency_seconds':{
            'p50':float(np.quantile(lat,.5)),'p95':float(np.quantile(lat,.95)),
            'p99':float(np.quantile(lat,.99)),'max':max(lat)},
        'length_finish_count':sum(r['api_response']['choices'][0]['finish_reason']=='length' for r in rows)}


def analyze(reference: Path,candidate: Path,reference_audit: Path,candidate_audit: Path,comparison: Path) -> dict:
    b,c,p=[json.loads(x.read_text()) for x in (reference_audit,candidate_audit,comparison)]
    if (p.get('classification')!='confirmed_deployment_quality_comparison_across_runs_not_Q_causality'
            or p['reference_audit_sha256']!=_sha(reference_audit) or p['candidate_audit_sha256']!=_sha(candidate_audit)):
        raise ValueError('Require the complete original1719 provenance/score comparison before latency reuse')
    tasks={}
    for task in TASKS:
        groups={'BF16':_rows(reference,b,'bf16',task),'FP8_XQA':_rows(reference,b,'fp8_e4m3',task),
                'FP8_BF16Q':_rows(candidate,c,'fp8_e4m3',task)}
        first=groups['BF16']
        for rows in groups.values():
            if any(x['_record_sha256']!=y['_record_sha256'] or x['_prompt_sha256']!=y['_prompt_sha256']
                   or x['api_response']['usage']['prompt_tokens']!=y['api_response']['usage']['prompt_tokens']
                   for x,y in zip(first,rows)):
                raise ValueError('Cross-run latency rows do not use identical original records/prompts')
        metrics={name:_describe(rows) for name,rows in groups.items()};changes={}
        for name in ('BF16','FP8_XQA'):
            differences=[1000*(y['client_latency_seconds']-x['client_latency_seconds'])
                         for x,y in zip(groups[name],groups['FP8_BF16Q'])]
            changes[name]={'candidate_complete_latency_delta_median_ms':statistics.median(differences),
                'fraction_candidate_slower':sum(x>0 for x in differences)/len(differences),
                'finite_cohort_RPS_change_pct':100*(metrics['FP8_BF16Q']['finite_cohort_completed_RPS']/metrics[name]['finite_cohort_completed_RPS']-1)}
        tasks[task]={'metrics':metrics,'paired_descriptive_changes':changes}
    return {'classification':'confirmed_descriptive_buffered_quality_cohort_clocks_across_runs',
        'reference_run_id':b['run_id'],'candidate_run_id':c['run_id'],
        'comparison_sha256':_sha(comparison),'tasks':tasks,'new_GPU_runs':0,
        'limits':['Fixed benchmark cohorts and caps, buffered JSON, different GPU/time/cache/scheduling histories',
            'First request to last response includes client offering and drain; not steady-state peak or AIPerf certification',
            'No streaming TTFT/ITL or user-accepted SLO goodput inferred; no independent Q-causality',
            'Per-task N=50 tail quantiles are descriptive; observations and repeated questions are not independent new users']}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('reference-mirror','candidate-mirror','reference-audit','candidate-audit','comparison','out'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();r=analyze(a.reference_mirror,a.candidate_mirror,a.reference_audit,a.candidate_audit,a.comparison)
    write_new_json(a.out,r);print(json.dumps({'classification':r['classification'],'tasks':len(r['tasks']),'new_GPU_runs':0}))


if __name__=='__main__':main()
