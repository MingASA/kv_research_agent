"""Render the frozen numerical gate; never infer serving or scored quality."""

import argparse
import hashlib
import json
from pathlib import Path


def render(source: Path,output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    raw=source.read_bytes();data=json.loads(raw)
    if (data.get('schema')!='FP8_independent_same_base_BF16_calibration_v1'
            or data.get('requests')!=136 or data.get('formal_quality_answers_used') is not False
            or data.get('frozen_candidate')!={'K':.125,'V':.5,'Q':1.}):
        raise ValueError('Require the exact independent frozen calibration')
    output.mkdir()
    keys=('value_current','value_candidate');rows=[]
    for key in keys:
        row=data['numeric_summary'][key]
        rows.append(dict(row,recipe=key,MSE=row['squared_error']/row['finite'],
            clipped_percent=100*row['clipped']/row['finite'],
            normalized_MSE=row['squared_error']/row['squared_signal']))
    public={'schema':'FP8_independent_calibration_plot_data_v1','requests':136,
        'run_id':'upstream-fast-20261106a-fp8-numeric-0003','model':'Qwen3.8-27B',
        'python_source_commit':'260b5fab204436af5a6f5f6a4897f3076c72df19',
        'compiled_source_commit':'da7c6446ff4361c8060633b04d6bfb77e6f737ff',
        'source_acceptance_sha256':hashlib.sha256(raw).hexdigest(),'rows':rows,
        'MSE_reduction_percent':100*(1-rows[1]['MSE']/rows[0]['MSE']),
        'source_lifecycle_audit':'104 files SHA verified; original COMPLETE/released',
        'quality_acceptance':False,'performance_acceptance':False}
    (output/'DATA.json').open('x').write(json.dumps(public,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','savefig.dpi':180})
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.5),layout='constrained')
    labels=['V scale 0.0625','V scale 0.5\n(frozen candidate)']
    for ax,metric,title in zip(axes,('MSE','clipped_percent'),('V quantization mean squared error','Clipped V activation elements (%)')):
        values=[r[metric] for r in rows]
        ax.bar(labels,values,color=['#3971a8','#d38736'],width=.55)
        ax.set_title(title);ax.set_ylim(0,max(values)*1.25)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for i,value in enumerate(values):ax.annotate(f'{value:.9g}',(i,value),xytext=(0,4),textcoords='offset points',ha='center')
    fig.suptitle('Independent BF16 activation calibration: 136 requests')
    fig.supxlabel('Numerical gate only; task quality and serving performance require separate tests.',fontsize=9)
    for suffix in ('png','svg'):fig.savefig(output/('fp8-numerical-gate.'+suffix))
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--acceptance',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();render(a.acceptance,a.out)
