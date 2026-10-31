"""Plot audited sequential application observations; no GPU/remote operations."""
import argparse
import hashlib
import json
from pathlib import Path


def render(paths,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    audits=[json.loads(path.read_bytes()) for path in paths]
    present={row['arm'] for row in audits}
    assert len(audits)==len(present) and present in ({'bf16','fp8_e4m3'},{'bf16','fp8_e4m3','nvfp4'})
    assert len({json.dumps(row['request_identity']['service'],sort_keys=True) for row in audits})==1
    for row in audits:
        assert row['schema']=='actual-mixed-pressure-residency-audit-v1'
        assert row['all32_actual_prompt_counts_match_official_JSON_tokenizer'] is True
        assert row['sample_points'] and row['application_audit']['raw_HTML_metric_values_independently_reparsed']>0
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');table=[]
    for arm,label,color in [('bf16','BF16 / FA3+XQA','#637482'),('fp8_e4m3','FP8 / FA2','#287f9d'),('nvfp4','NVFP4 / FA2','#aa743e')]:
        if arm not in present:continue
        row=next(item for item in audits if item['arm']==arm);points=row['sample_points']
        x=[point['elapsed_since_measured_worker_start_seconds'] for point in points]
        used=[point['values'].get('num_used_blocks',np.nan) for point in points]
        running=[point['values'].get('num_running_requests',np.nan) for point in points]
        pages=row['physical_FullKV_pages'];fraction=[100*value/pages for value in used]
        axes[0].plot(x,fraction,color=color,label=f'{label}: {pages:,} pages',linewidth=1.6)
        axes[1].step(x,running,where='post',color=color,label=label,linewidth=1.6)
        selected=row['qualifying_same_sample']
        table.append({'arm':arm,'physical_FullKV_pages':pages,
            'max_effective_used_FullKV_pages':row['max_effective_used_FullKV_pages'],
            'max_running_requests_independent_sample_peak':row['max_running_requests_independent_sample_peak'],
            'same_sample_running':None if selected is None else selected['values']['num_running_requests'],
            'same_sample_used_pages':None if selected is None else selected['values']['num_used_blocks'],
            'observed_more_used_pages_than_BF16_physical_pool':row['observed_used_pages_exceed_same_budget_BF16_physical3868_pages'],
            'exact_counter_span':row['application_audit']['counter_span_covers_exact_offered_prompt_total']})
    axes[0].set_title('Effective Full-KV used pages / physical pages')
    axes[0].set_ylabel('Physical pool occupancy (%)');axes[0].set_ylim(0,105)
    axes[1].set_title('Observed running requests');axes[1].set_ylabel('Application gauge');axes[1].set_ylim(bottom=0)
    for axis in axes:
        axis.set_xlabel('Seconds since measured client invocation');axis.legend(frameon=False,fontsize=8,loc='best')
    missing_note='; NVFP4 app observations unavailable (SSH reply lost)' if 'nvfp4' not in present else ''
    fig.suptitle('Qwen3.8-27B / H800: same real 16-document, 32-request pressure case\n'
                 'Batch8192 / chunk2048; sequential2s samples'+missing_note,fontsize=11)
    fig.savefig(output/'resident_pressure.png',dpi=180,bbox_inches='tight')
    fig.savefig(output/'resident_pressure.svg',bbox_inches='tight');plt.close(fig)
    svg=output/'resident_pressure.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    result={'sources':[{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
        'renderer_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'summary':table,
        'all_observed_frames_kept_missing_values_are_plot_gaps':True,
        'missing_arms_are_unknown_not_zero':sorted({'bf16','fp8_e4m3','nvfp4'}-present),
        'GPU_or_remote_calls':False,'max_admission_or_HBM_bandwidth_or_SLO_claim':False,
        'figures':{name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ['resident_pressure.png','resident_pressure.svg']}}
    with (output/'inputs.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit',type=Path,action='append',required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();render(args.audit,args.out)
