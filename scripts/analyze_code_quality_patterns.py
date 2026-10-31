#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Explain frozen code-similarity changes without changing the official scores."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

from controller.durable import write_new_json


def _first_line(prediction: str) -> str:
    return next((line for line in prediction.lstrip('\n').split('\n')
                 if not any(marker in line for marker in ('`','#','//'))),'')


def analyze(full_audit: Path, pilot: Path, fp8: Path, bf16: Path) -> dict:
    audit=json.loads(full_audit.read_text())
    scorers=list(pilot.glob('support/agent_work/runs/*/input/scorer/LongBench/metrics.py'))
    if len(scorers)!=1:
        raise ValueError('Official frozen scorer is ambiguous')
    spec=importlib.util.spec_from_file_location('frozen_longbench_code_patterns',scorers[0])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    hashes={str(full_audit):hashlib.sha256(full_audit.read_bytes()).hexdigest(),
            str(scorers[0]):hashlib.sha256(scorers[0].read_bytes()).hexdigest()}
    result={}
    for task in ('lcc','repobench-p'):
        arms={'bf16':{},'fp8_e4m3':{}}
        for arm,fragment in (('bf16',bf16),('fp8_e4m3',fp8)):
            label='auto' if arm=='bf16' else 'fp8-e4m3'
            for mirror in (pilot,fragment):
                paths=list(mirror.glob('support/agent_work/runs/*/output/'+arm+'/quality/runs/'+task+'/'+label+'.jsonl'))
                if len(paths)!=1:
                    raise ValueError('Prediction file missing or ambiguous')
                path=paths[0];hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                for line in path.read_text().splitlines():
                    row=json.loads(line);identity=row['_id']
                    if identity in arms[arm]:
                        raise ValueError('Repeated original code record')
                    arms[arm][identity]=row
        pairs=audit['paired_items'][task]
        if any(set(arms[a])!={v['id'] for v in pairs} for a in arms):
            raise ValueError('Code cohort differs from the audited50 original records')
        cases=[]
        for pair in pairs:
            details={}
            for arm in arms:
                row=arms[arm][pair['id']]
                score=max(module.code_sim_score(row['prediction'],answer) for answer in row['answers'])
                if abs(score-pair['scores'][arm])>1e-12:
                    raise ValueError('Official code score differs; do not explain invalid evidence')
                selected=_first_line(row['prediction'])
                eligible_found=any(not any(marker in line for marker in ('`','#','//'))
                                   for line in row['prediction'].lstrip('\n').split('\n'))
                details[arm]={'score':score,'finish_reason':row['api_response']['choices'][0]['finish_reason'],
                    'prediction_chars':len(row['prediction']),
                    'prediction_whitespace_only':not row['prediction'].strip(),
                    'prediction_nonblank_lines':sum(bool(line.strip()) for line in row['prediction'].splitlines()),
                    'selected_line_sha256':hashlib.sha256(selected.encode()).hexdigest(),
                    'eligible_line_found':eligible_found,
                    'selected_line_chars':len(selected),'selected_line_whitespace_only':eligible_found and not selected.strip(),
                    'scored_string_whitespace_only':not selected.strip(),
                    'opening_explanation_prefix':bool(re.match(r'^(Certainly|Here|The|This|Sure|Based|To|It)\b',row['prediction'].lstrip())),
                    'heuristic_explanation_prefix':bool(re.match(r'^(Here|The|This|Sure|Based|To|It)\b',selected)),
                    'selected_line_in_original_context':bool(selected.strip()) and selected.strip() in row['context'],
                    'prompt_tokens':row['usage'].get('prompt_tokens')}
            cases.append({'index':pair['index'],'source_id':pair['id'],'delta':details['fp8_e4m3']['score']-details['bf16']['score'],'arms':details})
        result[task]={'records':len(cases),'improved':sum(c['delta']>0 for c in cases),
            'degraded':sum(c['delta']<0 for c in cases),'unchanged':sum(c['delta']==0 for c in cases),
            'gross_score_loss_sum':-sum(min(0,c['delta']) for c in cases),
            'gross_score_gain_sum':sum(max(0,c['delta']) for c in cases),
            'exact_scored_line_matches':{a:sum(c['arms'][a]['score']==1 for c in cases) for a in arms},
            'whitespace_selected_lines':{a:sum(c['arms'][a]['selected_line_whitespace_only'] for c in cases) for a in arms},
            'no_eligible_line':{a:sum(not c['arms'][a]['eligible_line_found'] for c in cases) for a in arms},
            'empty_effective_scored_string':{a:sum(c['arms'][a]['scored_string_whitespace_only'] for c in cases) for a in arms},
            'opening_explanation_prefix_count':{a:sum(c['arms'][a]['opening_explanation_prefix'] for c in cases) for a in arms},
            'entire_prediction_whitespace_only':{a:sum(c['arms'][a]['prediction_whitespace_only'] for c in cases) for a in arms},
            'explanation_prefix_heuristic_count':{a:sum(c['arms'][a]['heuristic_explanation_prefix'] for c in cases) for a in arms},
            'cases':sorted(cases,key=lambda c:c['delta'])}
    return {'schema':'kv-code-quality-error-patterns-v1','classification':'confirmed_official_score_recheck_and_descriptive_patterns',
        'tasks':result,'source_sha256':hashes,'gpu_used':False,'model_inference_performed':False,
        'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations':['Frozen first-eligible-line fuzzy similarity, not code execution or agent success',
                       'No eligible line and an eligible blank line are separate; both may yield empty scored strings',
                       'Heuristic formatting flags are not causal explanations',
                       'No scorer, prediction, cap or original terminal state was changed']}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('full-audit','pilot','fp8','bf16','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();value=analyze(args.full_audit,args.pilot,args.fp8,args.bf16)
    write_new_json(args.out,value)
    print(json.dumps({task:{k:v for k,v in data.items() if k!='cases'} for task,data in value['tasks'].items()}))


if __name__=='__main__':
    main()
