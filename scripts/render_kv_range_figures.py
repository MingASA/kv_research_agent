#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Render the independently recovered real KV diagnostic; no model/GPU use."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def render(audit_path: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    audit=json.loads(audit_path.read_text())
    if audit['classification']!='confirmed_raw_heldout_BF16_KV_distribution_and_AIPerf_integrity_not_FP8_quality':
        raise ValueError('Require independently verified actual observations')
    out.mkdir(parents=True,exist_ok=False)
    data=[]
    for layer in range(48):
        for kind,scale in [('key',.125),('value',.0625)]:
            rows=[r for r in audit['rows'] if r['layer']==layer and r['kind']==kind]
            if len(rows)!=16:raise ValueError('Incomplete layer/phase/head observations')
            energy=sum(r['squared_energy'] for r in rows);error=sum(r['squared_error'] for r in rows)
            data.append({'layer':layer,'kind':kind,'finite_values':sum(r['finite'] for r in rows),
                'absmax':max(r['absmax'] for r in rows),'fixed_coverage':448*scale,
                'clipped_values':sum(r['clipped'] for r in rows),
                'underflow_values':sum(r['underflow_nonzero_to_zero'] for r in rows),
                'relative_l2_pct':100*math.sqrt(error/energy)})
    with (out/'layer_data.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
    with (out/'group_data.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(audit['rows'][0]),lineterminator='\n');w.writeheader();w.writerows(audit['rows'])
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for kind,color,label in [('key','#2563eb','K'),('value','#d97706','V')]:
        selected=[r for r in data if r['kind']==kind]
        axes[0].plot([r['layer'] for r in selected],[r['absmax']/r['fixed_coverage'] for r in selected],color=color,label=label,lw=1.7)
        axes[1].plot([r['layer'] for r in selected],[r['relative_l2_pct'] for r in selected],color=color,label=label,lw=1.7)
    axes[0].axhline(1,color='#374151',ls='--',lw=1,label='current FP8 range limit')
    axes[0].set(ylabel='Observed absmax / current coverage',title='No finite-value clipping observed',ylim=(0,1.1))
    axes[1].set(ylabel='QDQ relative L2 (%)',title='Representational error remains')
    for ax in axes:
        ax.set(xlabel='Layer (0–47)',xlim=(0,47));ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.suptitle('BF16 KV → fixed-scale FP8 E4M3 proxy · heldout 128 chats',fontsize=13)
    fig.text(.5,-.035,'Actual valid writes; includes Graph replays. Short chat only. No FP8 generation-quality claim.\nOriginal run FAILED in postprocessing; observations independently recovered.',ha='center',fontsize=9,color='#374151')
    for extension in ('svg','png'):
        fig.savefig(out/('kv_range.'+extension),dpi=160,bbox_inches='tight')
    plt.close(fig)
    manifest={'classification':'derived_real_observation_figure_no_new_GPU',
        'run_id':audit['run_id'],'original_root_status':audit['original_root_status'],
        'audit_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'matplotlib_version':matplotlib.__version__,
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'out':str(out),'files':len(manifest['files']),'GPU_used':False}))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();render(args.audit,args.out)


if __name__=='__main__':main()
