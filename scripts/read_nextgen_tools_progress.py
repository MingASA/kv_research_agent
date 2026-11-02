"""Bounded read-only status of the original owned native tool experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.controller.cli import _ticket
from scripts.controller.durable import write_new_json
from scripts.controller.transport import PinnedSSH


ALLOWED=('nextgen-bf16-qwen38-native-tools32-r0-20261030r',
         'nextgen-nvfp4-qwen38-native-tools32-r0-20261030m',
         'nextgen-fp8fa2-qwen38-native-tools32-r0-20261030n',
         'upstream-A-tools32-r0-20261102b',
         'upstream-C-fp8-tools32-r0-20261102b',
         'upstream-C-fp8-tools32-r0-20261102c',
         'upstream-C-nvfp4-tools32-r0-20261102b')

SOURCE=r'''
import datetime,hashlib,json,sys
from pathlib import Path
p=json.load(sys.stdin);t=p['ticket'];lib={'__name__':'owned_tools_readonly'}
exec(p['rpc_source'],lib);root=lib['check_ticket'](t,False);lib['existing'](root,t)
assert t['request']['run_id'] in p['allowed']
s=lib['status'](root,t);arm=t['request']['arms'][0];out=root/'output'/arm;files={}
for name in ('validation_result.json','tool_API_admission_raw.json','tool_API_admission.json',
             'tool_generations/generation_summary.json'):
 f=out/name
 if f.exists():
  assert f.stat().st_size<256*1024;b=lib['regular'](f);v=json.loads(b)
  files[name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),
   'value':v if name!='tool_API_admission_raw.json' else
    {'status':v.get('status'),'error':v.get('error'),'raw_SSE_lines':len(v.get('raw_SSE',[]))}}
raw=lib['regular'](root/'input/tools32/tasks.jsonl')
assert hashlib.sha256(raw).hexdigest()==t['request']['tool_evaluation']['tasks_sha256']
ids=[json.loads(line)['id'] for line in raw.splitlines() if line];assert len(ids)==len(set(ids))==32
tasks=[]
for identifier in ids:
 assert identifier.startswith('multi_turn_base_') and identifier.rsplit('_',1)[1].isdigit()
 f=out/'tool_generations'/identifier/'generation_result.json'
 if not f.exists():continue
 assert f.stat().st_size<8192;v=json.loads(lib['regular'](f));assert v['task_id']==identifier
 tasks.append({k:v.get(k) for k in ('task_id','state','finished_user_turns','model_API_requests',
                                  'tool_calls','failure_stage','error')})
print(json.dumps({'ok':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'run_id':t['request']['run_id'],'request_sha256':t['request_sha256'],'status':s['status'],
 'supervisor_alive':s['supervisor_alive'],'live_owned_containers':s.get('live_owned_containers'),
 'terminal_result':s.get('files',{}).get('result.json'),
 'files':files,'generation_results':tasks,'GPU_snapshot':s.get('gpu_snapshot'),
 'model_task_correctness_scored':False,'remote_writes':False}))
'''


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',choices=ALLOWED,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Preserve the original read-only observation')
    if args.run_id.startswith('upstream-'):
        from scripts.observe_nextgen_long_metrics import require_original_ack
        require_original_ack(args.run_id)
    ticket=_ticket(args.run_id)
    transport=PinnedSSH(ticket['approval'],args.out.parent/'tools-progress-pins'/ticket['reservation_id'])
    result=transport.rpc(SOURCE,{'ticket':ticket,'rpc_source':Path('scripts/controller/remote_rpc.py').read_text(),
                                'allowed':ALLOWED},mutation=False,timeout=60)
    write_new_json(args.out,result)
    print(json.dumps({'run_id':args.run_id,'status':result['status'],'files':list(result['files']),
        'completed_generation_results':len(result['generation_results']),
        'states':{state:sum(row['state']==state for row in result['generation_results'])
                  for state in {row['state'] for row in result['generation_results']}}}))
