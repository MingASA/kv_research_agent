"""Read an original ACKed upstream run, including fixed bounded operator metadata."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.controller.transport import PinnedSSH
from scripts.observe_nextgen_long_metrics import require_original_ack

SOURCE = '''import hashlib,json,sys
from pathlib import Path
p=json.load(sys.stdin)
lib={'__name__':'upstream_readonly_status_library'}
exec(p['rpc_source'],lib)
t=p['ticket'];root=lib['check_ticket'](t,False);lib['existing'](root,t)
response=lib['status'](root,t)
response['operator_files']={}
names=[]
if t['request']['workload']['kind']=='upstream_native_C_operator_qualification':
 names=['operators/'+case+'/'+name for case in
  ('fp8_e4m3-fa3-xqa','fp8_e4m3-fa2-flashinfer','nvfp4-fa2-flashinfer')
  for name in ('qualification.json','failure.json','timings.json','prefill_timings.json')]
 names.append('operators/acceptance.json')
if t['request']['workload']['kind']=='upstream_FP8_direct_library_attribution':
 names=['diagnosis/'+name for name in ('prefill.json','chunked_prefill.json','changed_input_XQA_graph.json','diagnosis.json')]
if t['request']['workload']['kind']=='upstream_native_VMM_compatibility':
 names=[profile+'/'+name for profile in ('bf16-vmm','fp8-vmm','nvfp4-vmm')
  for name in ('validation_result.json','trace_summary.json','cross_request_prefix_probe.json','server.log')]
 names.append('VMM_compatibility_result.json')
if t['request']['workload']['kind']=='upstream_AC_fixed_production_service':
 names=[case+'/'+phase+'/'+name for case in ('sharegpt-c32','sharegpt-c64','govreport-c32')
  for phase in ('warmup','measured') for name in ('command.json','independent_worker_audit.json')]
 names+=['service_result.json']+[case+'/server.log' for case in ('sharegpt-c32','sharegpt-c64','govreport-c32')]
for relative in names:
  path=root/'output'/t['request']['arms'][0]/relative
  if not path.exists():
   response['operator_files'][relative]={'exists':False};continue
  if relative.endswith('/server.log') and t['request']['workload']['kind']=='upstream_AC_fixed_production_service':
   lib['require'](path.resolve()==path and path.is_file(),'Invalid service log')
   with path.open('rb') as handle:
    handle.seek(max(0,path.stat().st_size-16000));tail=handle.read(16000)
   response['operator_files'][relative]={'exists':True,'bytes':path.stat().st_size,'tail':tail.decode('utf-8',errors='replace')};continue
  lib['require'](path.stat().st_size<=1048576,'Oversized operator metadata')
  raw=lib['regular'](path)
  if relative.endswith('.log'):
   response['operator_files'][relative]={'exists':True,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'tail':raw[-16000:].decode('utf-8',errors='replace')};continue
  value=json.loads(raw)
  summary={k:value[k] for k in ('status','candidate','classification','actual_model_graph_replay','actual_cuda_graph_launches','known_answers_checked','repeat_hit_tokens','GPU_operator_qualification','all_candidates_pass','qualified_candidates','direct_library_conformance_pass','all_original_dense_checks_pass','conclusion','records','requests','successes','official_AIPerf_complete','all_official_tokenizer_and_server_prompt_counts_match','profiling_records','request_rps','output_tps','actual_output_tokens','observation_seconds','ttft_p95_ms','itl_p95_ms','e2e_p95_ms') if isinstance(value,dict) and k in value}
  if isinstance(value,dict) and 'error' in value:summary['error']=value['error'][:4096]
  response['operator_files'][relative]={'exists':True,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'summary':summary}
response['ok']=True
response['remote_writes']=0
print(json.dumps(response))
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require_original_ack(args.run_id)
    ticket = json.loads((Path('state/controller/tickets')/(args.run_id+'.json')).read_bytes())
    assert ticket['request']['run_id'] == args.run_id
    if args.out.exists():
        raise FileExistsError('Preserve the original read-only receipt')
    result = PinnedSSH(ticket['approval'], args.out.parent/'pin'/args.run_id).rpc(SOURCE,
        {'ticket': ticket, 'rpc_source': Path('scripts/controller/remote_rpc.py').read_text()})
    args.out.open('x').write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'run_id': args.run_id, 'status': result.get('status'),
        'supervisor_alive': result.get('supervisor_alive'),
        'response_sha256': hashlib.sha256(args.out.read_bytes()).hexdigest(),
        'operator_files': result['operator_files'], 'remote_writes': 0}))
