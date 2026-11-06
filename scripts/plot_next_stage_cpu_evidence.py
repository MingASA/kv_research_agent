"""Plot measured historical ranges and the actual 32-task agent screening bound."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def main() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root=Path(__file__).resolve().parents[1]
    output=root/'reports/experiments/mxfp4-fp8-20261105/cpu-figures-b1'
    output.mkdir(parents=True,exist_ok=False)
    operations=root/'reports/operations/mxfp4-fp8-20261105a'
    source=json.loads((operations/'FP8_RANGE_PRIOR-a1.json').read_bytes())
    observed=json.loads((root/source['source']).read_bytes())
    rows=[]
    for layer in range(3,64,4):
        rows.append({'layer':layer,**{kind:max(r['absmax'] for r in observed['rows']
                         if r['layer_id']==layer and r['kind']==kind) for kind in ('key','value')}})
    with (output/'historical-ranges.csv').open('x') as stream:
        writer=csv.DictWriter(stream,fieldnames=['layer','key','value']);writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,kind,limit in zip(axes,('key','value'),(56,28)):
        ax.plot([r['layer'] for r in rows],[r[kind] for r in rows],marker='o',label='Observed absolute maximum')
        ax.axhline(limit,color='#c44e52',linestyle='--',label=f'Original FP8 range: {limit}')
        if kind=='value':ax.axhline(224,color='#55a868',linestyle=':',label='Candidate range: 224')
        ax.set(xlabel='Full-attention layer',ylabel='Absolute value',title=kind.upper())
        ax.legend(fontsize=8)
    fig.suptitle('Historical independent BF16 calibration: range risk only',fontsize=12)
    fig.text(.5,.015,'Older executor 849bcbf9; not a current clipping-rate or quality result.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.93))
    for suffix in ('png','svg'):fig.savefig(output/('historical-ranges.'+suffix),dpi=180)
    plt.close(fig)
    screen=json.loads((operations/'AGENTIC_REAL_TOOLS_CPU_SCREEN-a1.json').read_bytes())
    prompts=[r['prompt_tokens'] for r in screen['rows']]
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    axes[0].hist(prompts,bins=20,color='#4c72b0',edgecolor='white')
    axes[0].set(xlabel='Official tokenizer / server prompt tokens',ylabel='Model calls',title='Actual 32-task tool loop\n352 model calls')
    axes[1].bar(['32 sessions at\nindividual maxima','BF16 usable\ncapacity'],
                [screen['sum_task_max_pages128'],screen['BF16_usable_pages']],color=['#55a868','#8172b2'])
    for i,v in enumerate([screen['sum_task_max_pages128'],screen['BF16_usable_pages']]):
        axes[1].text(i,v+12,str(v),ha='center')
    axes[1].set(ylabel='KV pages (128 tokens/page)',title='Conservative simultaneous\nlive-cache bound',ylim=(0,1500))
    fig.text(.5,.015,'Original recorded generations; does not bound longer replies from an untested candidate.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1))
    for suffix in ('png','svg'):fig.savefig(output/('agentic-capacity-screen.'+suffix),dpi=180)
    plt.close(fig)
    (output/'PROVENANCE.json').write_text(json.dumps({'range_source':source['source'],
        'range_source_sha256':source['source_sha256'],'screen_source':'reports/operations/mxfp4-fp8-20261105a/AGENTIC_REAL_TOOLS_CPU_SCREEN-a1.json',
        'historical_ranges_are_not_current_clipping_rate':True,'new_GPU_service_results_present':False},indent=2)+'\n')
    print(output)


if __name__=='__main__':main()
