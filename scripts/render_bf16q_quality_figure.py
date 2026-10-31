#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Export actual fixed-cohort deployment quality deltas; no new measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();d=json.loads(a.comparison.read_text())
    if d['classification']!='confirmed_deployment_quality_comparison_across_runs_not_Q_causality':
        raise ValueError('Independently audited full deployment comparison required')
    a.out.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    names=['gsm8k','narrativeqa','qasper','multifieldqa_en','2wikimqa','hotpotqa','musique','lcc','repobench-p']
    labels=['GSM flex','NarrativeQA','Qasper','MultiFieldQA','2WikiMQA','HotpotQA','MuSiQue','LCC','RepoBench']
    rows=[]
    for name,label in zip(names,labels):
        scores=d['task_scores'][name]['flexible_em' if name=='gsm8k' else 'official_score']
        rows.append({'label':label,'FP8_XQA_minus_BF16_pp':100*(scores['FP8_XQA']-scores['BF16']),
            'FP8_BF16Q_minus_BF16_pp':100*(scores['FP8_BF16Q']-scores['BF16'])})
    m=d['LongBench_macro'];rows.append({'label':'LB macro','FP8_XQA_minus_BF16_pp':100*(m['FP8_XQA']-m['BF16']),
        'FP8_BF16Q_minus_BF16_pp':100*(m['FP8_BF16Q']-m['BF16'])})
    with (a.out/'quality_delta_data.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    fig,ax=plt.subplots(figsize=(13,5.9));x=np.arange(len(rows));width=.35
    ax.bar(x-width/2,[r['FP8_XQA_minus_BF16_pp'] for r in rows],width,label='Existing resident FP8 + XQA',color='#457b9d')
    ax.bar(x+width/2,[r['FP8_BF16Q_minus_BF16_pp'] for r in rows],width,label='Resident FP8 + BF16-Q FA2 decode',color='#d17b32')
    ax.axhline(0,color='#333333',linewidth=.8);ax.axvline(8.5,color='#aaaaaa',linewidth=.8,linestyle=':')
    ax.set_xticks(x,[r['label'] for r in rows],rotation=25,ha='right')
    ax.set_ylabel('Score difference from modern BF16 reference (percentage points)')
    ax.set_title('Preserving BF16 Q did not remove the observed task-quality deficit\nOriginal 1,719 questions; modern BF16 FA3 + XQA reference')
    ax.legend(loc='upper right',frameon=False);ax.spines[['top','right']].set_visible(False)
    ax.set_ylim(-5.5,3.2);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    ci=d['paired_bootstrap_95ci'];new=ci['versus_BF16']['LongBench_macro_delta_pp'];old=ci['versus_FP8_XQA']['LongBench_macro_delta_pp']
    fig.text(.06,.055,f'New LB macro vs BF16: {rows[-1]["FP8_BF16Q_minus_BF16_pp"]:.3f} pp; conditional paired 95% CI [{new[0]:.3f}, {new[1]:.3f}] pp.',fontsize=9)
    fig.text(.06,.03,f'New vs existing FP8: -0.379 pp; CI [{old[0]:.3f}, {old[1]:.3f}] crosses zero. Different runs/GPU/cache histories; no isolated Q-causality.',fontsize=9)
    fig.subplots_adjust(bottom=.24,top=.82,left=.075,right=.985)
    for suffix in ('svg','png'):fig.savefig(a.out/('quality_deltas.'+suffix),dpi=160)
    plt.close(fig)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'classification':'derived_verified_fixed_cohort_quality_figure_no_new_GPU',
        'comparison_sha256':sha(a.comparison),'generator_sha256':sha(Path(__file__)),
        'matplotlib_version':matplotlib.__version__,'files':{p.name:sha(p) for p in sorted(a.out.iterdir())}}
    (a.out/'manifest.json').open('x').write(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'figure_files':list(manifest['files']),'new_GPU_runs':0}))


if __name__=='__main__':main()
