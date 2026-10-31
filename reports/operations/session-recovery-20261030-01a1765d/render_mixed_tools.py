"""Plot all32 original native-tool tasks, preserving incomplete failures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.compare_nextgen_tools import bound


def render(paths,manifests,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    assert len(paths)==len(manifests)==3
    verified=[bound(path,manifest) for path,manifest in zip(paths,manifests)]
    entries=[item[0] for item in verified];requests=[item[1] for item in verified]
    assert all(r.get('scheduling_profile')=='mixed8192_chunk2048_v1' for r in requests)
    assert len({json.dumps(r['service'],sort_keys=True) for r in requests})==1
    assert all(r['service']['max_tokens_per_batch']==8192 and r['service']['max_tokens_per_chunk_for_prefill']==2048 for r in requests)
    arms=['bf16','fp8_e4m3','nvfp4'];labels=['BF16 / FA3+XQA','FP8 / FA2','NVFP4 / FA2']
    assert len(entries)==3 and {row['arm'] for row in entries}==set(arms)
    assert all(row['actual_model_provenance_verified'] is True for row in entries)
    assert len({row['quality']['source_tasks_sha256'] for row in entries})==1
    assert len({row['quality']['reference_sha256'] for row in entries})==1
    values=[]
    for arm in arms:
        row=next(entry for entry in entries if entry['arm']==arm);quality=row['quality'];scores=quality['scores']
        assert quality['all_original_tasks_retained_in_denominator'] and quality['all_actual_histories_SSE_usage_and_tool_replays_verified']
        assert quality['tasks']==len(scores)==32 and len({s['task_id'] for s in scores})==32
        correct=sum(s['task_correct'] for s in scores);incomplete=sum(s['budget_or_incomplete_conversation'] for s in scores)
        assert correct==quality['correct'] and quality['task_success_percent']==100*correct/32
        assert not any(s['task_correct'] and s['budget_or_incomplete_conversation'] for s in scores)
        values.append({'arm':arm,'correct':correct,'failed_complete':32-correct-incomplete,'incomplete_or_budget_failed':incomplete,
            'completed_user_turns':sum(s['completed_user_turns'] for s in scores),
            'expected_user_turns':sum(s['original_user_turns'] for s in scores)})
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axis=plt.subplots(figsize=(8.5,5),layout='constrained');positions=range(3)
    left=[row['correct'] for row in values];failed=[row['failed_complete'] for row in values];incomplete=[row['incomplete_or_budget_failed'] for row in values]
    axis.bar(positions,left,color='#287f9d',label='State/response correct')
    axis.bar(positions,failed,bottom=left,color='#aeb9c1',label='Failed completed conversation')
    axis.bar(positions,incomplete,bottom=[a+b for a,b in zip(left,failed)],color='#aa743e',label='Incomplete / step budget (score0)')
    for x,row in zip(positions,values):
        axis.annotate(f"{row['correct']}/32 ({100*row['correct']/32:.2f}%)",(x,32),xytext=(0,6),textcoords='offset points',ha='center')
    axis.set_xticks(list(positions),labels);axis.set_ylim(0,36);axis.set_ylabel('Original tasks (fixed denominator32)')
    axis.set_title('Actual Qwen3.8-27B native multi-turn tools on H800\nBatch8192 / chunk2048;32source tasks /112expected turns',fontsize=12)
    axis.legend(loc='lower center',bbox_to_anchor=(.5,-.32),frameon=False,fontsize=9)
    fig.savefig(output/'native_tools32.png',dpi=180,bbox_inches='tight');fig.savefig(output/'native_tools32.svg',bbox_inches='tight');plt.close(fig)
    svg=output/'native_tools32.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    result={'sources':[{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'sha_manifest':str(manifest),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
        for path,manifest in zip(paths,manifests)],
        'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'plotted':values,
        'not_full_BFCL_or_final_natural_language_quality_or_performance':True,
        'all32_original_tasks_kept_in_denominator':True,'GPU_or_remote_calls':False}
    with (output/'inputs.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit',type=Path,action='append',required=True)
    parser.add_argument('--sha-manifest',type=Path,action='append',required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();render(args.audit,args.sha_manifest,args.out)
