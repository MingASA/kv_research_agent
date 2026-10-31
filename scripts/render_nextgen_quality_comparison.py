"""Render paired quality changes from independently verified native KV scores."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fp8-comparison',type=Path,required=True)
    parser.add_argument('--nvfp4-comparison',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--scheduling-label',help='Explicit configuration caption for an exported comparison')
    args=parser.parse_args()
    inputs=[('FP8 FA2',args.fp8_comparison),('NVFP4',args.nvfp4_comparison)]
    rows=[];reference_sha=None
    for name,path in inputs:
        raw=path.read_bytes();data=json.loads(raw)
        if reference_sha is None:reference_sha=data['reference_summary_sha256']
        assert data['reference_summary_sha256']==reference_sha
        assert data['bootstrap_repeats']==4000 and not data['production_quality_accepted']
        gsm=data['task_differences']['gsm8k']
        rows.append({'format':name,'source':str(path.resolve()),'source_sha256':hashlib.sha256(raw).hexdigest(),
                     'GSM8K_flexible_EM_delta_pp':gsm['delta_percentage_points'],
                     'GSM8K_paired_CI_pp':gsm['paired_95pct_context_cluster_CI_pp'],
                     'LongBench_macro_delta_pp':data['LongBench_core_macro_delta_pp'],
                     'LongBench_paired_CI_pp':data['LongBench_macro_paired_95pct_context_cluster_CI_pp']})
    args.out.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.6 if args.scheduling_label else 3.3),sharey=True)
    specs=[('GSM8K flexible EM','GSM8K_flexible_EM_delta_pp','GSM8K_paired_CI_pp'),
           ('LongBench core macro','LongBench_macro_delta_pp','LongBench_paired_CI_pp')]
    for axis,(title,metric,interval) in zip(axes,specs):
        axis.axvline(0,color='0.55',linewidth=1,linestyle='--')
        for y,row in enumerate(rows):
            value=row[metric];low,high=row[interval]
            assert low<=value<=high
            axis.errorbar(value,y,xerr=[[value-low],[high-value]],fmt='o',capsize=4,
                          color=('#3b6e9b' if y==0 else '#855b93'),markersize=6)
        axis.set_title(title,fontsize=11)
        axis.set_xlabel('Change from BF16 (percentage points)',fontsize=9)
        axis.set_yticks(range(len(rows)),[r['format'] for r in rows])
        axis.set_ylim(-.55,1.55);axis.invert_yaxis()
        axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='x',alpha=.15)
    title='Qwen3.8-27B native KV: fixed core quality cohort'
    if args.scheduling_label:title+='\n'+args.scheduling_label
    fig.suptitle(title,fontsize=11 if args.scheduling_label else 12)
    fig.text(.5,.02,'95% paired bootstrap intervals; quality acceptance and service performance remain separate.',
             ha='center',fontsize=8,color='0.3')
    fig.tight_layout(rect=(0,.09,1,.86 if args.scheduling_label else .92))
    for suffix in ('png','svg'):fig.savefig(args.out/('paired_quality_change.'+suffix),dpi=180)
    svg=args.out/'paired_quality_change.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    (args.out/'inputs.json').write_text(json.dumps({'reference_summary_sha256':reference_sha,
        'scheduling_label':args.scheduling_label,'rows':rows},indent=2)+'\n')


if __name__=='__main__':main()
