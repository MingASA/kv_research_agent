"""Main-agent local metadata after an actual once ACK, ticket and ledger row."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.controller.durable import digest
from scripts.observe_nextgen_long_metrics import require_original_ack

RUNS={
 'xc':'nextgen-fp8fa2-qwen38-mixed8192-prefix32-r0-20261030xc',
 'yc':'nextgen-nvfp4-qwen38-mixed8192-prefix32-r0-20261030yc',
 'zc':'nextgen-bf16-qwen38-mixed8192-tools32-r0-20261030zc',
 'za':'nextgen-fp8fa2-qwen38-mixed8192-tools32-r0-20261030za',
 'zb':'nextgen-nvfp4-qwen38-mixed8192-tools32-r0-20261030zb',
}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--suffix',choices=RUNS,required=True)
    parser.add_argument('--receipt',type=Path,required=True);args=parser.parse_args()
    run=RUNS[args.suffix];require_original_ack(run);receipt=args.receipt.resolve(strict=True)
    ack=json.loads(receipt.read_text());assert ack['ok'] is True and ack['status']=='DISPATCHED'
    ticket=json.loads((ROOT/'state/controller/tickets'/(run+'.json')).read_text());r=ticket['request']
    assert r['run_id']==run and r['scheduling_profile']=='mixed8192_chunk2048_v1'
    ledger=sqlite3.connect('file:'+str(ROOT/'state/controller/ledger.sqlite3')+'?mode=ro',uri=True)
    ledger.row_factory=sqlite3.Row
    row=dict(ledger.execute('select * from runs where run_id=?',(run,)).fetchone());ledger.close()
    assert row['state']=='RUNNING' and row['request_sha256']==ticket['request_sha256']==digest(r)
    assert row['reservation_id']==ticket['reservation_id']
    once=json.loads((ROOT/'state/controller/local_launch_once'/(run+'.json')).read_text())
    assert Path(once['receipt']).resolve(strict=True)==receipt
    package=json.loads(Path(once['package']).read_text());tools='tool_evaluation' in r
    config=r['tool_evaluation'] if tools else r['performance_evaluation']
    source=config['tasks_sha256'] if tools else config['input_manifest_sha256']
    workload='actual_native_tools32_112turns_C4' if tools else 'real_long-prefix-c16_AIPerf32_sessionstop_v2'
    now=datetime.now(timezone.utc).isoformat();op=Path(__file__).parent;relative=receipt.relative_to(ROOT).as_posix()
    index=ROOT/'results/INDEX.csv';rows=list(csv.DictReader(index.open()));fields=list(rows[0])
    assert not any(item['run_id']==run for item in rows),'Never repeat first INDEX insertion'
    entry={key:'' for key in fields};entry.update(run_id=run,status='ORIGINAL_DISPATCH_ACK_TICKET_LEDGER_RUNNING',
        model=r['model_id'],backend=r['arms'][0]+'_mixed8192_chunk2048_'+workload,
        git_sha=r['python_source_commit'],binary_sha256=r['binary_sha256'],dataset_sha256=source,
        gpu_uuid=r['gpu_uuids'][0],concurrency=str(config['client_concurrency'] if tools else config['concurrency']),
        manifest_path=r['remote_output']+'/manifest.json',metric_path=relative,
        validity='originalACK_only_not_actual_workload_or_quality_acceptance',
        notes='First INDEX after actual originalACK/ticket/RUNNINGledger; '+str(len(package['assets']))+
            'frozenassets; newprofile full1720 quality and S realpilot bound; noSLOacceptance. '+
            ('Same original32tasks/112turns/10steps/natural1024cap; no gold on serving path.' if tools else
             'CPU194/sessionstop v2 and actualBF16WC complete32/16x2 prerequisite independently verified.'))
    rows.append(entry)
    with index.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    path=ROOT/'state/STATE.json';state=json.loads(path.read_text());phase=args.suffix.upper()+'_ORIGINAL_ACK_'+workload.upper()+'_RUNNING'
    state.update(updated_utc=now,phase=phase,active_run_id=run,active_run_ids=[run],active_gpu_uuids=r['gpu_uuids'])
    m=state['mixed_prefill_work'];m.update(state=phase,current_run_id=run,current_original_receipt=relative,
        current_reservation_id=row['reservation_id'],current_original_supervisor_pid=ack['identity']['pid'],
        current_own_CID=None,current_own_CUDA_PID=None,current_model_loaded=False,
        current_stage='original_ACK_before_actual_model_status',current_workload=workload,
        current_original_collection=None,current_measured_requests_observed=0,current_deadline_utc=ticket['deadline_utc'])
    m[args.suffix.upper()+'_original_receipt']=relative
    state['pending_GPU_proposal']={'state':'NONE_WHILE_ORIGINAL_RUN_ACTIVE','run_ids':[]}
    state['monitoring'].update(active_run_id=run,current_status=phase,resource_status_anchor_run_id=run)
    path.write_text(json.dumps(state,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    event={'utc':now,'action':args.suffix.upper()+'_ORIGINAL_ACK_FIRST_INDEX','classification':'confirmed',
        'run_id':run,'reservation_id':row['reservation_id'],'supervisor_pid':ack['identity']['pid'],
        'deadline_utc':ticket['deadline_utc'],'GPU_success_claim':False}
    with (ROOT/'state/EVENTS.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
    line='Latest '+now+': '+args.suffix.upper()+'原once ACK/ticket/RUNNINGledger后首INDEX；'+workload+'，supervisor'+str(ack['identity']['pid'])+'，完整原raw及评分/性能仍待完成；不重发。main活跃/Luna至'+state['monitoring']['until_utc']+'。\n\n'
    for name in ['plans/NEXT_ACTIONS.md','state/NEXTGEN_ACTIVE_RECOVERY.md']:
        path=ROOT/name;path.write_text(line+path.read_text())
    print(json.dumps(event))


if __name__=='__main__':main()
