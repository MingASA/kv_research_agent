"""Plot the audited diagnostic pilot without implying formal performance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def render(proof_path: Path, manifest_path: Path, observations: Path, output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    proof=json.loads(proof_path.read_text());manifest=json.loads(manifest_path.read_text())
    raw=(observations/'application_metrics.jsonl').read_bytes()
    if (proof['state']!='MIXED_PREFILL_BF16_PILOT_INDEPENDENTLY_VERIFIED'
            or proof['source_SHA_manifest_sha256']!=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            or proof['application_raw_sha256']!=hashlib.sha256(raw).hexdigest()):
        raise ValueError('Plot requires the exact independently verified pilot inputs')
    root=Path(manifest['mirror_directory'])/'files'
    requests=[json.loads(line) for line in (root/'output/bf16/calibration_raw.jsonl').read_bytes().splitlines()]
    long=[row for row in requests if row['id'].startswith('long-holdout-')]
    rows=[json.loads(line) for line in raw.splitlines()[1:]]
    start=min(row['request_started_monotonic_ns'] for row in long)
    end=max(row['request_ended_monotonic_ns'] for row in long)
    selected=[row for row in rows if not row['errors'] and start<=row['monotonic_ns']<=end]
    if not selected:raise ValueError('No actual long-request observations')
    points=[{'seconds':(row['monotonic_ns']-start)/1e9,
             'running':row['values']['num_running_requests'],
             'pages':row['values']['num_used_blocks'],
             'long_client_inflight':sum(r['request_started_monotonic_ns']<=row['monotonic_ns']
                                       <r['request_ended_monotonic_ns'] for r in long)} for row in selected]
    qualified=next(row for row in selected if row['monotonic_ns']==proof['qualifying_same_sample'])
    x=[row['seconds'] for row in points];at=(qualified['monotonic_ns']-start)/1e9
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    fig,axes=plt.subplots(2,1,figsize=(9.5,5.6),sharex=True,layout='constrained')
    axes[0].step(x,[row['running'] for row in points],where='post',color='#287f9d',label='Server running requests')
    axes[0].step(x,[row['long_client_inflight'] for row in points],where='post',color='#777777',
                 linestyle='--',label='Long client requests in flight')
    axes[0].scatter([at],[qualified['values']['num_running_requests']],color='#aa743e',zorder=4)
    axes[0].set_ylabel('Requests');axes[0].set_ylim(bottom=0);axes[0].legend(loc='lower left',frameon=False)
    axes[1].step(x,[row['pages'] for row in points],where='post',color='#287f9d',label='Active Full-KV pages')
    bound=proof['largest_single_request_page_upper_bound']
    axes[1].axhline(bound,color='#777777',linestyle='--',label=f'Single-request conservative upper bound: {bound}')
    pages=qualified['values']['num_used_blocks'];axes[1].scatter([at],[pages],color='#aa743e',zorder=4)
    axes[1].annotate(f'Same sample: {int(pages)} pages, {int(qualified["values"]["num_running_requests"])} running',
                     (at,pages),xytext=(8,-20),textcoords='offset points',fontsize=9)
    axes[1].set_ylabel('64-token Full-KV pages');axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel('Seconds since first held-out long request began')
    axes[1].legend(loc='lower left',frameon=False)
    fig.suptitle('Qwen3.8-27B BF16 / one H800: real overlapping long-request residency\n'
                 'Batch budget 8192; chunk configuration 2048; diagnostic range observer enabled',fontsize=11)
    fig.supxlabel('Sequential application samples; no formal speed or maximum-capacity claim',fontsize=9)
    output.mkdir(parents=True,exist_ok=False)
    fig.savefig(output/'mixed_prefill_pilot.png',dpi=180,bbox_inches='tight')
    fig.savefig(output/'mixed_prefill_pilot.svg',bbox_inches='tight');plt.close(fig)
    svg=output/'mixed_prefill_pilot.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    (output/'inputs.json').write_text(json.dumps({'proof':str(proof_path),
        'proof_sha256':hashlib.sha256(proof_path.read_bytes()).hexdigest(),
        'manifest_sha256':proof['source_SHA_manifest_sha256'],'application_raw_sha256':proof['application_raw_sha256'],
        'plot_points':points,'single_request_page_upper_bound':bound,'qualifying_time_seconds':at,
        'formal_quality_or_performance_or_maximum_capacity_certified':False},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof',type=Path,required=True);parser.add_argument('--sha-manifest',type=Path,required=True)
    parser.add_argument('--observations',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();render(args.proof,args.sha_manifest,args.observations,args.out)
