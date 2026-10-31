"""Read application cache gauges in an existing owned long-text container.

No model request, GPU operation, remote file write, start/stop/retry or settings
change. Docker exec runs one stdlib HTTP reader in the exact admitted container;
its output is returned to a new local receipt. Original experiments are intact.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from scripts.controller.cli import _ticket
from scripts.controller.durable import write_new_json,write_new_bytes
from scripts.controller.transport import PinnedSSH

ROOT=Path(__file__).resolve().parents[1]


ALLOWED=('nextgen-bf16-qwen38-long16-c8-r0-20261030f',
         'nextgen-nvfp4-qwen38-long16-c8-r0-20261030g',
         'nextgen-fp8fa2-qwen38-long16-c8-r0-20261030h',
         'nextgen-bf16-qwen38-prefix32-c16-r0-20261030o',
         'nextgen-nvfp4-qwen38-prefix32-c16-r0-20261030p',
         'nextgen-fp8fa2-qwen38-prefix32-c16-r0-20261030q',
         'nextgen-bf16-qwen38-mixed8192-calibration-r0-20261030s',
         'nextgen-bf16-qwen38-mixed8192-prefix32-r0-20261030w',
         'nextgen-fp8fa2-qwen38-mixed8192-prefix32-r0-20261030x',
         'nextgen-nvfp4-qwen38-mixed8192-prefix32-r0-20261030y',
         'nextgen-bf16-qwen38-mixed8192-prefix32-r1-20261030wc',
         'nextgen-fp8fa2-qwen38-mixed8192-prefix32-r0-20261030xc',
         'nextgen-nvfp4-qwen38-mixed8192-prefix32-r0-20261030yc')

INNER=r'''
import datetime,json,os,sys,time,urllib.request
p=json.load(sys.stdin)
assert os.environ['CUDA_VISIBLE_DEVICES']==''
lib={'__name__':'owned_readonly_bvar_parser'};exec(p['parser_source'],lib)
start=time.monotonic();failures=0;total=0
pilot=p['run_id']=='nextgen-bf16-qwen38-mixed8192-calibration-r0-20261030s'
max_seconds=1800 if pilot else 900;max_bytes=(96 if pilot else 48)*2**20
print(json.dumps({'schema':'nextgen-owned-long-bvar-observation-v1','run_id':p['run_id'],
 'request_sha256':p['request_sha256'],'interval_seconds':2,'max_seconds':max_seconds,'max_output_bytes':max_bytes,
 'HTTP_scope':'private_127.0.0.1_vars_only','GPU_or_model_calls':False,
 'limitation':'Sequential sampled application gauges;not atomic or hardware traffic counters'}),flush=True)
while time.monotonic()-start<max_seconds:
 began=time.monotonic();row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'monotonic_ns':time.monotonic_ns(),'values':{},'raw':{},'errors':{}}
 for name in lib['METRICS']:
  try:
   with urllib.request.urlopen('http://127.0.0.1:19001/vars/'+name,timeout=.2) as response:
    raw=response.read(128*1024+1)
   if len(raw)>128*1024:raise ValueError('Excessive application response')
   body=raw.decode();row['raw'][name]=body;row['values'][name]=lib['parse_metric'](name,body)
  except (OSError,ValueError) as error:row['errors'][name]=type(error).__name__+': '+str(error)[:180]
 row['elapsed_seconds']=time.monotonic()-began;line=json.dumps(row,allow_nan=False)
 total+=len(line.encode())+1
 if total>max_bytes:break
 print(line,flush=True);failures=failures+1 if not row['values'] else 0
 if failures>=3:break
 time.sleep(max(0,2-(time.monotonic()-began)))
'''

OUTER=r'''
import base64,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);t=p['ticket'];lib={'__name__':'readonly_owned_rpc_library'}
exec(p['rpc_source'],lib);root=lib['check_ticket'](t,False);lib['existing'](root,t)
assert t['request']['run_id'] in p['allowed']
pilot=t['request']['run_id']=='nextgen-bf16-qwen38-mixed8192-calibration-r0-20261030s'
if pilot:
 assert t['request'].get('scheduling_profile')=='mixed8192_chunk2048_v1' and t['request']['arms']==['bf16']
 assert t['request']['workload']['kind']=='native_hybrid_format_service_validation'
 assert t['request']['range_observation'] is True and t['request']['calibration_count']==136
else:
 assert t['request']['performance_evaluation']['case'] in ('long-unique-c8','long-prefix-c16')
 if 'mixed8192' in t['request']['run_id']:
  assert t['request'].get('scheduling_profile')=='mixed8192_chunk2048_v1'
  assert t['request']['performance_evaluation']['case']=='long-prefix-c16'
  assert t['request']['range_observation'] is False
  assert t['request']['service']['max_tokens_per_batch']==8192
  assert t['request']['service']['max_tokens_per_chunk_for_prefill']==2048
arm=t['request']['arms'][0];output=root/'output'/arm
# A new formal service may spend900s in the operator gate and another900s in
# model startup. Cover those existing bounds plus idle/warmup without changing
# the GPU job or the900s/48MiB actual sampling envelope.
mixed_formal='mixed8192' in t['request']['run_id'] and not pilot
wait_seconds=2100 if mixed_formal else 900;until=time.monotonic()+wait_seconds
signal=output/('calibration_raw.jsonl' if pilot else 'measured/command.json')
while not signal.exists():
 if (root/'result.json').exists():raise ValueError('Original run terminal before observation;no restart')
 if time.monotonic()>until:raise ValueError('Original measurement not observed in'+str(wait_seconds)+'s;no recovery action')
 time.sleep(5)
identity=json.loads(lib['regular'](output/'container_identity.json'))
lease=json.loads(lib['regular'](output/'lease.json'));assert lease['request_sha256']==t['request_sha256']
sys.path.insert(0,str(root/'runtime'))
from kv_controller.supervisor import DockerBackend
backend=DockerBackend(t,root);container=backend._inspect(identity['id'])
assert container['State']['Running'] and container['Created']==identity['created']
assert lib['alive'](identity['pid_identity'])
parser_path=root/'runtime/kv_controller/service_metrics.py'
parser=lib['regular'](parser_path).decode();assert hashlib.sha256(parser.encode()).hexdigest()==t['assets_sha256']['runtime/kv_controller/service_metrics.py']
payload={'parser_source':parser,'run_id':t['request']['run_id'],'request_sha256':t['request_sha256']}
uid=t['approval']['approved_remote_identity']['uid'];gid=t['approval']['approved_remote_identity']['gid']
command=['docker','exec','-i','--user',str(uid)+':'+str(gid),'-e','CUDA_VISIBLE_DEVICES=',
 '-e','PYTHONDONTWRITEBYTECODE=1',identity['id'],'python3','-B','-c',p['inner']]
result=subprocess.run(command,input=json.dumps(payload).encode(),capture_output=True,timeout=1830 if pilot else 930)
assert len(result.stdout)<=(97 if pilot else 49)*2**20 and len(result.stderr)<65536
print(json.dumps({'ok':True,'run_id':t['request']['run_id'],'container_id':identity['id'],
 'request_sha256':t['request_sha256'],'parser_sha256':hashlib.sha256(parser.encode()).hexdigest(),
 'container_exec_exit':result.returncode,'stdout_base64':base64.b64encode(result.stdout).decode(),
 'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),'stderr':result.stderr.decode(errors='replace'),
 'remote_file_writes':False,'GPU_or_model_calls':False,'container_start_or_stop_or_retry':False,
 'observation_max_seconds':1800 if pilot else 900,'observation_max_bytes':(96 if pilot else 48)*2**20,
 'wait_for_measurement_max_seconds':wait_seconds}))
'''


def require_original_ack(run_id: str) -> None:
    """Never enter the remote reader before the original once-launch has ACKed."""
    directory=ROOT/'state/controller/local_launch_once'
    finished=json.loads((directory/(run_id+'.finished.json')).read_text())
    intent=json.loads((directory/(run_id+'.json')).read_text())
    if (finished.get('run_id')!=run_id or intent.get('run_id')!=run_id
            or finished.get('local_CLI_exit_code')!=0):
        raise ValueError('Original once-launch has no successful local ACK; no remote observation')
    receipt=Path(intent['receipt']).resolve(strict=True)
    recorded=Path(finished['receipt'])
    if not recorded.is_absolute():recorded=ROOT/recorded
    response=json.loads(receipt.read_text())
    if recorded.resolve(strict=True)!=receipt or response.get('ok') is not True or response.get('status')!='DISPATCHED':
        raise ValueError('Original once-launch receipt is not a successful dispatch ACK')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',choices=ALLOWED,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();require_original_ack(args.run_id)
    args.out.mkdir(mode=0o700,parents=True,exist_ok=False)
    ticket=_ticket(args.run_id)
    source=Path(__file__).read_bytes()
    write_new_json(args.out/'LOCAL_INTENT.json',{'run_id':args.run_id,'request_sha256':ticket['request_sha256'],
        'source_sha256':hashlib.sha256(source).hexdigest(),
        'outer_source_sha256':hashlib.sha256(OUTER.encode()).hexdigest(),
        'inner_source_sha256':hashlib.sha256(INNER.encode()).hexdigest(),
        'GPU_or_model_calls':False,'remote_file_writes':False,'repeat_permitted':False})
    write_new_bytes(args.out/'OBSERVER_SOURCE.py',source)
    write_new_bytes(args.out/'OUTER_SOURCE.py',OUTER.encode())
    write_new_bytes(args.out/'INNER_SOURCE.py',INNER.encode())
    transport=PinnedSSH(ticket['approval'],args.out/'pin')
    # No shared read-only mux: a finite independent connection carries this one
    # observation. The remote program never restarts an original experiment.
    result=transport.rpc(OUTER,{'ticket':ticket,'rpc_source':Path('scripts/controller/remote_rpc.py').read_text(),
        'allowed':ALLOWED,'inner':INNER},mutation=True,
        timeout=3090 if 'mixed8192' in args.run_id and args.run_id!='nextgen-bf16-qwen38-mixed8192-calibration-r0-20261030s'
                else 2760 if args.run_id=='nextgen-bf16-qwen38-mixed8192-calibration-r0-20261030s' else 1860)
    write_new_json(args.out/'ORIGINAL_RESPONSE.json',result)
    raw=base64.b64decode(result['stdout_base64'],validate=True)
    if hashlib.sha256(raw).hexdigest()!=result['stdout_sha256']:raise ValueError('Observation SHA mismatch')
    write_new_bytes(args.out/'application_metrics.jsonl',raw)
    print(json.dumps({key:value for key,value in result.items() if key!='stdout_base64'}))


if __name__=='__main__':main()
