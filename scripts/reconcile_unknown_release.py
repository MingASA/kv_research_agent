"""Settle a stopped UNKNOWN run from two read-only driver-release witnesses.

The original remote UNKNOWN result is never changed. Operational failure is
retained; GPU accounting uses the last witness as a conservative upper bound.
This certificate never grants admission or repeats an experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from scripts.controller.approval import ApprovalError
from scripts.controller.durable import digest, write_new_json
from scripts.controller.quota import QuotaLedger
from scripts.controller.transport import PinnedSSH
from scripts.reconcile_absent_dispatch import _local_launcher_absent


ROOT=Path(__file__).resolve().parents[1]
SOURCE=r'''
import datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);t=p['ticket'];lib={'__name__':'owned_release_readonly_library'}
exec(p['rpc_source'],lib)
root=lib['check_ticket'](t,False);lib['existing'](root,t)
raw=lib['regular'](root/'result.json');result=json.loads(raw)
registry=json.loads(lib['regular'](root/'owned_containers.json'))
lib['require'](registry['reservation_id']==t['reservation_id'] and registry['request_sha256']==t['request_sha256'],'Unbound owned registry')
supervisor=json.loads(lib['regular'](root/'SUPERVISOR_STARTED.json'))['identity']
supervisor_live=False
try:
 observed=lib['identity'](supervisor['pid']);line=Path('/proc',str(supervisor['pid']),'stat').read_text();tail=line[line.rfind(')')+2:].split()
 supervisor_live=all(observed[k]==supervisor[k] for k in ('pid','uid','start_ticks','boot_id')) and tail[0]!='Z'
except FileNotFoundError:pass
identifiers=set(item['container_id'] for item in registry['containers'])
for label in ('kvra.reservation_id='+t['reservation_id'],'kvra.run_id='+t['request']['run_id']):
 identifiers.update(subprocess.check_output(['docker','ps','-aq','--no-trunc','--filter','label='+label],text=True,timeout=15).split())
containers=[]
for identifier in sorted(identifiers):
 value=json.loads(subprocess.check_output(['docker','inspect',identifier],text=True,timeout=15))[0];labels=value['Config'].get('Labels') or {}
 lib['require'](value['Id']==identifier and labels.get('kvra.run_id')==t['request']['run_id'] and labels.get('kvra.reservation_id')==t['reservation_id'] and labels.get('kvra.request_sha256')==t['request_sha256'],'Ambiguous owned container')
 containers.append({'id':identifier,'running':value['State']['Running'],'pid':value['State']['Pid'],'status':value['State']['Status'],'OOMKilled':value['State']['OOMKilled'],'finished_at':value['State']['FinishedAt']})
rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
gpus={}
for line in rows.splitlines():
 u,m,v=[x.strip() for x in line.split(',')]
 if u in t['request']['gpu_uuids']:gpus[u]={'memory_mib':int(m),'util_pct':int(v),'pids':[]}
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15)
for line in apps.splitlines():
 u,n=[x.strip() for x in line.split(',')]
 if u in gpus:gpus[u]['pids'].append(int(n))
workers=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdecimal() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=os.getuid():continue
  args=(proc/'cmdline').read_bytes().split(b'\0')
  if any(x in args for x in (b'kv_controller.supervisor',b'kv_controller.worker')) and any(('/runs/'+t['request']['run_id']+'/').encode() in x for x in args):workers.append(int(proc.name))
 except FileNotFoundError:pass
total=0
for directory,folders,files in os.walk(root,followlinks=False):
 folders[:]=[name for name in folders if not (Path(directory)/name).is_symlink()]
 for name in files:total+=(Path(directory)/name).lstat().st_size
print(json.dumps({'ok':True,'schema':'kvra-unknown-release-witness-v1','remote_writes':False,
 'run_id':t['request']['run_id'],'reservation_id':t['reservation_id'],'request_sha256':t['request_sha256'],
 'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'allocation_boot_id':supervisor['boot_id'],'monotonic_ns':time.monotonic_ns(),
 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'result':result,'result_sha256':hashlib.sha256(raw).hexdigest(),
 'supervisor_live':supervisor_live,'containers':containers,'selected':gpus,'run_workers':workers,'new_output_bytes':total}))
'''


def terminal_from_witnesses(ticket: dict, witnesses: list[dict]) -> dict:
    if len(witnesses)!=2:raise ApprovalError('Two release witnesses required')
    first,last=witnesses
    for witness in witnesses:
        if witness.get('schema')!='kvra-unknown-release-witness-v1' or witness.get('ok') is not True or witness.get('remote_writes') is not False:
            raise ApprovalError('Unverified read-only release witness')
        if witness.get('boot_id')!=witness.get('allocation_boot_id'):
            raise ApprovalError('Remote reboot invalidated monotonic allocation accounting')
        for key,expected in (('run_id',ticket['request']['run_id']),('reservation_id',ticket['reservation_id']),('request_sha256',ticket['request_sha256'])):
            if witness.get(key)!=expected:raise ApprovalError('Release witness identity differs')
        result=witness['result']
        if (result.get('status')!='UNKNOWN' or result.get('gpu_release_verified') is not False
                or result.get('arm_results')!=[] or result.get('request_sha256')!=ticket['request_sha256']
                or result.get('run_id')!=ticket['request']['run_id'] or result.get('reservation_id')!=ticket['reservation_id']):
            raise ApprovalError('Not the exact incomplete original UNKNOWN run')
        if witness.get('supervisor_live') is not False or witness.get('run_workers')!=[]:
            raise ApprovalError('Original supervisor or worker may still be running')
        containers=witness.get('containers')
        if (not isinstance(containers,list) or not containers
                or any(c.get('running') is not False or c.get('pid')!=0 or c.get('status') not in ('exited','dead') for c in containers)
                or not set(result['live_owned_containers']).issubset({c['id'] for c in containers})):
            raise ApprovalError('Owned container release remains uncertain')
        selected=witness.get('selected',{})
        if set(selected)!=set(ticket['request']['gpu_uuids']) or any(g.get('memory_mib')!=0 or g.get('pids')!=[] or g.get('util_pct')!=0 for g in selected.values()):
            raise ApprovalError('Driver release not proven; busy or unknown GPU retained')
        if type(witness.get('monotonic_ns')) is not int or type(witness.get('new_output_bytes')) is not int:
            raise ApprovalError('Missing conservative accounting witness')
    if (not first.get('boot_id') or first['boot_id']!=last.get('boot_id')
            or first['result_sha256']!=last['result_sha256'] or first['result']!=last['result']
            or last['monotonic_ns']-first['monotonic_ns']<30_000_000_000):
        raise ApprovalError('Release/result not stable on the same boot for30seconds')
    result=first['result'];intervals=result.get('allocation_intervals_ns')
    if (type(result.get('allocation_started_monotonic_ns')) is not int or not isinstance(intervals,list)
            or len(intervals)!=1 or intervals[0]['start_ns']!=result['allocation_started_monotonic_ns']
            or intervals[0]['end_ns'] is not None or result['allocation_ended_monotonic_ns'] is not None
            or last['monotonic_ns']<=result['allocation_started_monotonic_ns']):
        raise ApprovalError('Unknown allocation envelope; do not guess GPU cost')
    return {'run_id':ticket['request']['run_id'],'reservation_id':ticket['reservation_id'],
        'request_sha256':ticket['request_sha256'],'status':'FAILED','gpu_release_verified':True,'live_owned_containers':[],
        'allocation_started_monotonic_ns':result['allocation_started_monotonic_ns'],
        'allocation_ended_monotonic_ns':last['monotonic_ns'],
        'allocation_intervals_ns':[dict(intervals[0],end_ns=last['monotonic_ns'])],
        'new_output_bytes':max(result['new_output_bytes'],*(w['new_output_bytes'] for w in witnesses)),
        'evidence_origin':'independent_two_readonly_driver_release_witnesses_original_UNKNOWN_unchanged',
        'original_result_status':'UNKNOWN','original_result_sha256':first['result_sha256'],
        'original_result_overwritten':False,'accounting_scope':'conservative_upper_bound_until_last_release_witness_not_GPU_active_time',
        'quality_or_performance_accepted':False,'new_GPU_admission_authorized':False,'readonly_witnesses':witnesses}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
    parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}',args.run_id) is None:raise ApprovalError('Invalid run identity')
    directory=ROOT/'state/controller';ticket=json.loads((directory/'tickets'/(args.run_id+'.json')).read_text())
    if args.out.exists() or not _local_launcher_absent(args.run_id):raise ApprovalError('Existing certificate or live launcher; inspect original')
    with QuotaLedger(directory) as ledger:
        row=ledger.get(args.run_id)
        if row['state']!='UNKNOWN' or row['reservation_id']!=ticket['reservation_id'] or row['request_sha256']!=digest(ticket['request']):
            raise ApprovalError('Only the exact original UNKNOWN reservation may be recovered')
    transport=PinnedSSH(ticket['approval'],directory/'pins'/ticket['reservation_id'])
    payload={'ticket':ticket,'rpc_source':(ROOT/'scripts/controller/remote_rpc.py').read_text()}
    witnesses=[transport.rpc(SOURCE,payload,mutation=False,timeout=90)]
    write_new_json(args.out.with_name(args.out.name+'.first.json'),witnesses[0]);time.sleep(30)
    witnesses.append(transport.rpc(SOURCE,payload,mutation=False,timeout=90))
    write_new_json(args.out.with_name(args.out.name+'.observations.json'),{'witnesses':witnesses})
    evidence=terminal_from_witnesses(ticket,witnesses)
    if not _local_launcher_absent(args.run_id):raise ApprovalError('Original launcher reappeared')
    write_new_json(args.out,evidence)
    with QuotaLedger(directory) as ledger:
        if ledger.get(args.run_id)['state']!='UNKNOWN':raise ApprovalError('Concurrent ledger change; do not double settle')
        row=ledger.settle(args.run_id,evidence)
    print(json.dumps({'run_id':args.run_id,'state':row['state'],'original_remote_result':'UNKNOWN_UNCHANGED',
        'recovered_classification':'FAILED_RELEASE_VERIFIED','accounting':'conservative_upper_bound','charged_gpu_us':row['actual_gpu_us'],
        'GPU_launched':False,'remote_writes':False,'new_GPU_admission_authorized':False}))


if __name__=='__main__':main()
