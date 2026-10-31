"""Keep complete and failed pressure conditions distinct in tables and plots."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.audit_nextgen_validation import _mirror_paths

OP=Path(__file__).parent
SPECS=[('bf16','mixed-BF16-WC-independent-AIPerf-audit-a1.json'),
       ('fp8_e4m3','mixed-FP8-XC-independent-AIPerf-audit-a1.json'),
       ('nvfp4','mixed-NVFP4-YC-independent-failed-load-audit-a1.json')]


def load(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output):
    rows=[];sources=[];identities=[]
    for arm,name in SPECS:
        path=OP/name;a=load(path);failed=arm=='nvfp4';run=a['run_id']
        manifest_path=ROOT/'results/mirrors'/(run+'-complete-a1')/'SHA_MANIFEST.json'
        manifest=load(manifest_path);_,root=_mirror_paths(manifest);ticket=load(root/'ticket.json');request=ticket['request'];terminal=load(root/'result.json')
        assert sha(manifest_path)==a['source_SHA_manifest_sha256'] and terminal['gpu_release_verified'] is True
        assert request['performance_evaluation']['case']=='long-prefix-c16' and request['performance_evaluation']['concurrency']==16
        assert request['scheduling_profile']=='mixed8192_chunk2048_v1'
        identities.append({k:request[k] for k in ['model_id','model_config_sha256','image_id','binary_sha256','server_source_commit','python_source_commit','gpu_uuids','service','graph_mode','prefix_cache']})
        if failed:
            assert a['schema']=='independent-original-failed-load-audit-v1' and a['original_status']==terminal['status']=='FAILED'
            assert a['all32_denominator_retained'] and a['successful_requests']==31 and a['failed_requests']==1
            count=a['successful_requests'];duration=a['full_AIPerf_observation_seconds'];outputs=a['verified_successful_output_tokens']
            rps=a['successful_completions_per_full_observation_second'];tps=a['verified_complete_output_tokens_per_full_observation_second']
            conditional=a['successful_response_only_conditional_quantiles'];profile=None
        else:
            assert a['schema']=='kv-nextgen-independent-production-performance-audit-v1' and terminal['status']=='COMPLETE'
            assert a['actual_model_and_Graph_and_raw_service_performance_verified'] and a['all_offered_requests_completed']
            profile=a['profile'];count=a['successful_requests'];duration=profile['observation_seconds'];outputs=profile['actual_output_tokens']
            rps=profile['request_rps'];tps=profile['output_tps'];conditional=None;assert count==32
        assert math.isclose(rps,count/duration,rel_tol=1e-8) and math.isclose(tps,outputs/duration,rel_tol=1e-8)
        lease=sum((part['end_ns']-part['start_ns'])/1e9 for part in terminal['allocation_intervals_ns'])
        row={'run_id':run,'arm':arm,'original_status':terminal['status'],'offered_requests':32,
            'successful_requests':count,'failed_requests':32-count,'success_fraction':count/32,
            'full_measured_window_seconds':duration,'successful_completions_per_full_window_second':rps,
            'verified_complete_output_tokens':outputs,'complete_output_tokens_per_full_window_second':tps,
            'successful_response_mean_output_tokens':outputs/count,'full_GPU_allocation_seconds':lease,
            'GPU_allocation_seconds_per_successful_completion':lease/count,
            'conditional_tails_are_not_full_cohort_tails':failed,
            'FullKV_physical_token_slots':a['native_allocation']['token_capacity'] if failed else a['lifecycle']['Full_KV_physical_token_slots'],
            'operators':json.dumps(request['operators'],sort_keys=True)}
        for label,key in [('ttft','ttft'),('request_mean_itl','itl'),('e2e','e2e')]:
            for pct in ['p50','p95','p99']:
                row[label+'_'+pct+'_full_success_cohort_ms']=None if failed else profile[key+'_'+pct+'_ms']
                row[label+'_'+pct+'_conditional_success_only_ms']=conditional[label+'_ms'][pct] if failed else None
        rows.append(row);sources.append({'audit':str(path),'audit_sha256':sha(path),'manifest':str(manifest_path),'manifest_sha256':sha(manifest_path)})
    assert identities[0]==identities[1]==identities[2]
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    with (output/'pressure_observations.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    result={'schema':'mixed-original-pressure-complete-and-failed-observations-v1','rows':rows,'sources':sources,
        'all32_offered_requests_remain_in_each_denominator':True,'NVFP4_original_failed_not_relabelled':True,
        'failed_partial_output_never_credited_as_success':True,'full_failed_cohort_latency_tails_not_invented':True,
        'different_kernels_not_dtype_only_causal_comparison':True,'quality_or_SLO_acceptance':False,
        'source_sha256':sha(Path(__file__))}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,4,figsize=(15,4.8),layout='constrained');labels=['BF16\nFA3+XQA','FP8\nFA2','NVFP4\nFA2 / FAILED'];colors=['#637482','#287f9d','#aa743e']
    panels=[('successful_completions_per_full_window_second','Successful complete requests / s',1),
            ('complete_output_tokens_per_full_window_second','Complete-response output tokens / s',1),
            ('ttft_p95_full_success_cohort_ms','TTFT p95 / s\n(fully successful cohorts)',.001),
            ('request_mean_itl_p95_full_success_cohort_ms','Request-mean ITL p95 / ms\n(fully successful cohorts)',1)]
    for axis,(key,title,scale) in zip(axes,panels):
        values=[np.nan if row[key] is None else row[key]*scale for row in rows]
        bars=axis.bar(range(3),values,color=colors,width=.62)
        bars[2].set_hatch('///')
        finite=[v for v in values if not np.isnan(v)];top=max(finite)
        for i,(value,row) in enumerate(zip(values,rows)):
            if np.isnan(value):axis.text(i,top*.4,'1 timeout\nfull tail N/A',ha='center',fontsize=9)
            else:axis.annotate(f'{value:.3g}\n{row["successful_requests"]}/32',(i,value),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
        axis.set_title(title,fontsize=10);axis.set_xticks(range(3),labels,fontsize=9);axis.set_ylim(0,top*1.35);axis.grid(axis='y',alpha=.15);axis.set_axisbelow(True)
    fig.suptitle('Qwen3.8-27B / one H800: same real C16, 32-request pressure case\n'
        'Batch8192 / chunk2048; full observation window; NVFP4 retains one600s timeout',fontsize=12)
    fig.savefig(output/'pressure_outcomes.png',dpi=180,bbox_inches='tight');fig.savefig(output/'pressure_outcomes.svg',bbox_inches='tight');plt.close(fig)
    svg=output/'pressure_outcomes.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    result['figures']={name:sha(output/name) for name in ['pressure_outcomes.png','pressure_outcomes.svg']}
    with (output/'inputs.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'out':str(output),'statuses':[row['original_status'] for row in rows],'successful_requests':[row['successful_requests'] for row in rows]}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();build(args.out)
