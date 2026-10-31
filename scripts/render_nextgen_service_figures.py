"""Plot independently audited production observations with source hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def render(paths: list[Path],output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    records=[];sources=[]
    for path in paths:
        raw=path.read_bytes();audit=json.loads(raw)
        if (audit.get('schema')!='kv-nextgen-independent-production-performance-audit-v1'
                or audit.get('actual_model_and_Graph_and_raw_service_performance_verified') is not True
                or audit.get('all_offered_requests_completed') is not True):
            raise ValueError('Only independently verified complete raw model observations may be plotted')
        records.append(audit);sources.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()})
    arms=['bf16','fp8_e4m3','nvfp4'];labels=['BF16 / FA3+XQA','FP8 / FA2','NVFP4 / FA2']
    services={json.dumps(row['request_identity']['service'],sort_keys=True) for row in records}
    if len(services)!=1:raise ValueError('Plot distinct scheduling profiles separately')
    service=records[0]['request_identity']['service']
    cases=list(dict.fromkeys(row['case'] for row in records));rows={}
    for case in cases:
        subset=[r for r in records if r['case']==case]
        if len(subset)!=3 or {r['arm'] for r in subset}!=set(arms):
            raise ValueError('Each plotted workload must contain exactly one complete observation of all three formats')
        rows[case]={r['arm']:r for r in subset}
    colors=['#637482','#287f9d','#aa743e']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'axes.axisbelow':True})
    fig,axes=plt.subplots(1,4,figsize=(15,4.4),layout='constrained')
    metrics=[('request_rps','Successful requests / s',1),('output_tps','Output tokens / s\n(natural response lengths)',1),
             ('ttft_p95_ms','TTFT p95 / s',.001),('itl_p95_ms','Request-mean ITL\np95 / ms',1)]
    positions=np.arange(len(cases));width=.24
    plotted=[]
    for axis,(metric,title,scale) in zip(axes,metrics):
        for index,(arm,label,color) in enumerate(zip(arms,labels,colors)):
            values=[rows[case][arm]['profile'][metric]*scale for case in cases]
            axis.bar(positions+(index-1)*width,values,width,label=label,color=color)
            for x,value,case in zip(positions+(index-1)*width,values,cases):
                top=max(rows[case][other]['profile'][metric]*scale for other in arms)
                axis.annotate(f'{value:.3g}',(x,top),xytext=(0,3+index*10),textcoords='offset points',ha='center',fontsize=8)
                plotted.append({'case':case,'arm':arm,'metric':metric,'source_value':rows[case][arm]['profile'][metric],
                                'display_value':value,'display_multiplier':scale})
        axis.set_title(title);axis.set_xticks(positions,[case.replace('long-unique-c8','Long docs\nC8')
                                                       .replace('long-prefix-c16','Repeated docs\nC16')
                                                       .replace('chat-c16','Chat\nC16') for case in cases],fontsize=9)
        axis.set_ylim(bottom=0,top=axis.get_ylim()[1]*1.13);axis.grid(axis='x',visible=False)
    handles,names=axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Qwen3.8-27B on one H800: audited real-text service observations\n'
                 f"Batch budget {service['max_tokens_per_batch']}; chunk {service.get('max_tokens_per_chunk_for_prefill',service['max_tokens_per_batch'])}; "
                 'one observation per arm; external warmup excluded',fontsize=12)
    fig.savefig(output/'service_observations.png',dpi=180,bbox_inches='tight',pad_inches=.12)
    fig.savefig(output/'service_observations.svg',bbox_inches='tight',pad_inches=.12);plt.close(fig)
    svg=output/'service_observations.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    (output/'inputs.json').write_text(json.dumps({'sources':sources,'plotted_values':plotted,
        'error_bars':'No repeated observations available; no statistical confidence interval is claimed',
        'production_quality_or_SLO_accepted':False},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audits',nargs='+',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();render(args.audits,args.out)
