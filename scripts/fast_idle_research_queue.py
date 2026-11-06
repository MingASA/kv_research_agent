"""Five-second read-only GPU stream and serial, exactly-once research dispatch.

Only an idle nomination can start a fresh guarded run. The remote supervisor
rechecks before create and stops only our allocation on a conflict. UNKNOWN
never permits another launch until its original allocation is reconciled.
Stdout contains transitions and at most one routine heartbeat per20minutes.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time

from scripts.controller.approval import TRUSTED_APPROVAL,load_approval_snapshot,validate_scope
from scripts.controller.cli import validate_frozen_policy
from scripts.controller.dispatch import Dispatcher
from scripts.controller.durable import atomic_json,digest,fsync_directory,secure_directory,write_new_json
from scripts.controller.idle_admission import FAST_ADMISSION
from scripts.controller.policy import validate_execution_authority,validate_request
from scripts.controller.processes import process_identity,signal_exact
from scripts.controller.quota import QuotaLedger
from scripts.controller.transport import PinnedSSH


ROOT=Path(__file__).resolve().parents[1]
STREAM=r'''
import datetime as dt,json,os,socket,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);a=p['approval'];e=a['approved_remote_identity']
if (os.getuid(),os.getgid(),socket.gethostname())!=(e['uid'],e['gid'],e['hostname']):raise RuntimeError('Identity changed')
base=Path(a['approved_workspace'])
if base.resolve(strict=True)!=base or base.stat().st_uid!=os.getuid():raise RuntimeError('Project owner changed')
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
if boot!=p['expected_boot_id']:raise RuntimeError('Boot changed')
end=dt.datetime.fromisoformat(p['not_after_utc']);seq=0;previous_cpu=None
while dt.datetime.now(dt.timezone.utc)<end:
 start=time.monotonic();seq+=1
 try:
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=2)
  g={}
  for row in rows.splitlines():
   u,m,v=[s.strip() for s in row.split(',')];g[u]={'memory_mib':int(m),'util_pct':int(v),'pids':[]}
  rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=2)
  for row in rows.splitlines():
   u,n=[s.strip() for s in row.split(',')]
   if u not in g:raise RuntimeError('Unknown GPU process mapping')
   g[u]['pids'].append(int(n))
  if not set(a['allowed_gpu_uuids']).issubset(g):raise RuntimeError('GPU UUID disappeared')
  cpu=[int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
  idle=None
  if previous_cpu is not None:
   delta=[x-y for x,y in zip(cpu,previous_cpu)]
   if sum(delta)>0:idle=delta[3]/sum(delta)
  previous_cpu=cpu
  out={'ok':True,'boot_id':boot,'seq':seq,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'gpus':g,'cpu_idle_fraction':idle,'remote_writes':False}
 except Exception as error:
  out={'ok':False,'seq':seq,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'error':type(error).__name__+': '+str(error)[:180],'remote_writes':False}
 try:print(json.dumps(out,separators=(',',':')),flush=True)
 except BrokenPipeError:break
 time.sleep(max(0,5-(time.monotonic()-start)))
'''


def now() -> dt.datetime:return dt.datetime.now(dt.timezone.utc)
def sha(path: Path) -> str:return hashlib.sha256(path.read_bytes()).hexdigest()


def coordinator_epoch() -> dict:
    return {p.name:sha(p) for p in sorted((ROOT/'scripts/controller').glob('*.py'))}


def require_coordinator_epoch(manifest: dict) -> None:
    if manifest.get('coordinator_runtime_sha256')!=coordinator_epoch():
        raise RuntimeError('Controller runtime changed; create a fresh queue handoff before new admission')


def idle_candidates(sample: dict,allowed: list[str],boot: str,at: dt.datetime) -> list[str]:
    if sample.get('ok') is not True or sample.get('remote_writes') is not False or sample.get('boot_id')!=boot:
        return []
    try:age=(at-dt.datetime.fromisoformat(sample['utc'])).total_seconds()
    except (ValueError,TypeError,KeyError):return []
    if not -5<=age<=10:return []
    result=[]
    for uuid in allowed:
        gpu=sample.get('gpus',{}).get(uuid,{})
        if (type(gpu.get('memory_mib')) is int and gpu['memory_mib']==0
                and type(gpu.get('util_pct')) is int and 0<=gpu['util_pct']<=5
                and gpu.get('pids')==[]):result.append(uuid)
    return result


def resource_retryable(result: dict) -> bool:
    if result.get('gpu_release_verified') is not True or result.get('live_owned_containers')!=[]:
        return False
    if result.get('status')=='CONFLICT':return True
    return (result.get('status')=='NOT_LAUNCHED' and result.get('allocation_intervals_ns')==[]
            and result.get('arm_results')==[] and result.get('error') in (
                'TimeoutError: Budget expired while waiting for 0s genuinely idle GPUs',
                'UnsafeState: GPU state changed between stable idle and worker create',
                'TimeoutError: No runtime budget remains after preflight'))


class Probe:
    """A bounded latest-value reader keeps polling during slow dispatch RPCs."""
    def __init__(self,transport: PinnedSSH,manifest: dict,directory: Path,number: int):
        self.latest={};self.received=0.;self.error='';self.lock=threading.Lock()
        self.started=time.monotonic()
        self.stderr=(directory/f'ssh-{number:04}.stderr').open('xb')
        self.process=subprocess.Popen(transport.argv+['python3 -B -u -c '+shlex.quote(STREAM)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,start_new_session=True)
        self.identity=process_identity(self.process.pid)
        payload={'approval':manifest['_approval'],'expected_boot_id':manifest['expected_boot_id'],
                 'not_after_utc':manifest['not_after_utc']}
        self.process.stdin.write(json.dumps(payload).encode());self.process.stdin.close()
        self.thread=threading.Thread(target=self._read,daemon=True);self.thread.start()

    def _read(self):
        try:
            while True:
                line=self.process.stdout.readline(16385)
                if not line:break
                if len(line)>16384 or not line.endswith(b'\n'):raise ValueError('Oversized/incomplete stream frame')
                value=json.loads(line)
                if not isinstance(value,dict):raise ValueError('Invalid stream frame')
                with self.lock:self.latest=value;self.received=time.monotonic()
        except Exception as error:self.error=type(error).__name__+': '+str(error)[:200]
        finally:
            if not self.error:self.error='Read-only stream ended'

    def sample(self):
        with self.lock:return dict(self.latest),self.received

    def close(self):
        if self.process.poll() is None:signal_exact(self.identity,signal.SIGTERM)
        try:self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            signal_exact(self.identity,signal.SIGKILL);self.process.wait(timeout=5)
        self.thread.join(timeout=2);self.stderr.close()


def index_run(request: dict,status: str) -> None:
    """Update only this run's row, preserving every older byte and line ending."""
    path=ROOT/'results/INDEX.csv';lines=path.read_bytes().splitlines(keepends=True)
    fields=next(csv.reader([lines[0].decode().strip()]))
    positions=[i for i,line in enumerate(lines[1:],1) if line.startswith((request['run_id']+',').encode())]
    if len(positions)>1:raise ValueError('Duplicate run index')
    entry=dict(zip(fields,next(csv.reader([lines[positions[0]].decode().strip()])))) if positions else {k:'' for k in fields}
    entry.update(run_id=request['run_id'],status=status,model=request['model_id'],
        backend=json.dumps(request['operators'],sort_keys=True),git_sha=request['python_source_commit'],
        binary_sha256=request['binary_sha256'],dataset_sha256=request.get('calibration_sha256',request['dataset_sha256']),
        gpu_uuid='|'.join(request['gpu_uuids']),manifest_path=request['remote_output']+'/manifest.json',
        validity='raw_prerequisite_only_no_production_acceptance',notes='User5s idle queue; exact original run; '+status)
    out=io.StringIO(newline='');csv.DictWriter(out,fieldnames=fields,lineterminator='\n').writerow(entry)
    if positions:lines[positions[0]]=out.getvalue().encode()
    else:lines.append(out.getvalue().encode())
    temporary=path.with_name('.INDEX.fast-idle-'+str(time.time_ns())+'.tmp')
    with temporary.open('x') as f:f.write(b''.join(lines).decode());f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
    fsync_directory(path.parent)


def prepare_attempt(manifest: dict,stage: dict,uuid: str,attempt: int) -> Path:
    original=Path(stage['package'])
    if sha(original)!=stage['package_sha256']:raise ValueError('Frozen stage template drifted')
    package=json.loads(original.read_bytes());request=dict(package['request'])
    run='upstream-fast-'+manifest['campaign']+'-'+stage['id']+f'-{attempt:04}'
    directory=ROOT/'plans/experiments'/run;directory.mkdir();(directory/'input').mkdir()
    remote=request['approved_workspace']+'/agent_work/runs/'+run
    request.update(run_id=run,remote_output=remote,python_source_repo=remote+'/source_repo',
        dataset_path=remote+'/input/validation.json',gpu_uuids=[uuid],resource_admission=dict(FAST_ADMISSION),
        operational_attempt=attempt,operational_retry_group=stage['package_sha256'],
        experiment_key=digest({'campaign':manifest['campaign'],'stage':stage['id'],'attempt':attempt,
                               'original_request':package['request_sha256']}),replicate=0)
    # Templates already contain the tested fast supervisor/runtime snapshot.
    # Retries reuse its exact SHA-bound bytes instead of reading mutable source.
    assets=dict(package['assets'])
    approval,_=load_approval_snapshot(TRUSTED_APPROVAL);validate_request(request,approval)
    result=dict(package,request=request,request_sha256=digest(request),assets=assets)
    result.pop('remote_asset_reuse',None)
    path=directory/'package.json';write_new_json(path,result);write_new_json(directory/'request.json',request)
    validate_frozen_policy(path,request,approval)
    return path


class Runner:
    def __init__(self,manifest: dict,directory: Path,state: dict):
        self.manifest,self.directory,self.state=manifest,directory,state
        self.probe=None;self.launcher=None;self.handles=[];self.stop=False;self.next_status=0.
        self.last_heartbeat=0.;self.last_signature=None;self.connections=0;self.next_connect=0.;self.backoff=5
        self.approval,self.approval_sha=load_approval_snapshot(TRUSTED_APPROVAL)
        self.transport=PinnedSSH(self.approval,directory/'pin')

    def event(self,status: str,**values):
        changed=status!=self.state.get('status')
        self.state.update(status=status,updated_utc=now().isoformat(),**values)
        atomic_json(self.directory/'STATE.json',self.state)
        if changed or values.get('important'):
            row={'utc':self.state['updated_utc'],'status':status,**values}
            with (self.directory/'EVENTS.jsonl').open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
            print(json.dumps({k:v for k,v in row.items() if k not in ('error_detail',)},separators=(',',':')),flush=True)
            with QuotaLedger(ROOT/'state/controller') as ledger:
                active=[dict(r) for r in ledger.connection.execute(
                    "SELECT run_id,request_json FROM runs WHERE state IN ('RESERVED','DISPATCHING','RUNNING','UNKNOWN')")]
            shared=ROOT/'state/STATE.json';document=json.loads(shared.read_bytes())
            document.update(main_agent_active=True,current_task_status='in_progress_user_fast_idle_queue',
                current_phase=status,active_run_ids=[r['run_id'] for r in active],
                active_run_id=self.state.get('active_run_id'),
                reserved_gpu_uuids=[gpu for r in active for gpu in json.loads(r['request_json'])['gpu_uuids']],
                updated_utc=self.state['updated_utc'])
            document['automatic_gpu_queue']={'state_file':str(self.directory/'STATE.json'),
                'status':status,'not_after_utc':self.manifest['not_after_utc'],'poll_seconds':5,'main_review_seconds':1200}
            document['active_gpu_uuids']=[gpu for r in active if r['run_id']==self.state.get('active_run_id')
                and self.state.get('physical_container') for gpu in json.loads(r['request_json'])['gpu_uuids']]
            if 'current_work' in document:document['current_work'].update(main_agent_active=True,status='in_progress_user_fast_idle_queue')
            atomic_json(shared,document)
            with (ROOT/'state/EVENTS.jsonl').open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())

    def launch(self,uuid: str):
        with QuotaLedger(ROOT/'state/controller') as ledger:
            usage=ledger.usage(self.approval)
        if usage['active_gpu_uuids'] or usage['uncertain_runs']:
            self.event('WAIT_EXISTING_OR_UNKNOWN_ALLOCATION');return
        stage=self.manifest['stages'][self.state['cursor']]
        wall=json.loads(Path(stage['package']).read_bytes())['request']['max_wall_seconds']
        if (dt.datetime.fromisoformat(self.manifest['not_after_utc'])-now()).total_seconds()<wall+180:
            self.event('WATCH_ONLY_INSUFFICIENT_COMPLETE_JOB_WINDOW');return
        require_coordinator_epoch(self.manifest)
        attempt=self.state.get('attempt',0)+1;self.state['attempt']=attempt
        path=prepare_attempt(self.manifest,stage,uuid,attempt)
        package=json.loads(path.read_bytes());request=package['request']
        validate_scope(self.approval,request,usage,now())
        run=request['run_id'];run_dir=self.directory/run;run_dir.mkdir()
        self.state.update(active_run_id=run,package=str(path),run_directory=str(run_dir),physical_container=False)
        self.state.pop('result',None)
        self.event('ORIGINAL_DISPATCH_INTENT',gpu=uuid,run_id=run,important=True)
        write_new_json(run_dir/'QUEUE_DISPATCH_INTENT.json',{'run_id':run,'package_sha256':sha(path),'do_not_repeat':True})
        out=(run_dir/'wrapper.stdout').open('x');err=(run_dir/'wrapper.stderr').open('x');self.handles=[out,err]
        args=[sys.executable,'-B','-m','scripts.launch_guarded_once','--package',str(path),
              '--receipt',str(run_dir/'dispatch.json'),'--stderr',str(run_dir/'dispatch.stderr')]
        self.launcher=subprocess.Popen(args,cwd=ROOT,stdout=out,stderr=err,start_new_session=True)
        write_new_json(run_dir/'LOCAL_DISPATCH_PROCESS.json',process_identity(self.launcher.pid))
        self.event('ORIGINAL_DISPATCH_IN_FLIGHT',run_id=run)

    def active(self):
        run=self.state['active_run_id'];run_dir=Path(self.state['run_directory'])
        if self.launcher is not None:
            if self.launcher.poll() is None:return
            for handle in self.handles:handle.close()
            self.handles=[];self.launcher=None
        ticket_path=ROOT/'state/controller/tickets'/(run+'.json')
        finished=ROOT/'state/controller/local_launch_once'/(run+'.finished.json')
        if not ticket_path.exists() or not finished.exists():
            self.event('HALTED_AMBIGUOUS_OR_REFUSED_ORIGINAL_DISPATCH',run_id=run);self.stop=True;return
        ticket=json.loads(ticket_path.read_bytes())
        if time.monotonic()<self.next_status:return
        self.next_status=time.monotonic()+30
        transport=PinnedSSH(ticket['approval'],run_dir/'pin')
        with QuotaLedger(ROOT/'state/controller') as ledger:
            dispatch=Dispatcher(ledger,transport,(ROOT/'scripts/controller/remote_rpc.py').read_text())
            response=dispatch.reconcile(ticket);row=ledger.get(run)
        atomic_json(run_dir/'LATEST_STATUS.json',response)
        self.state['physical_container']=bool(response.get('live_owned_containers'))
        if row['state']=='SETTLED':
            result=json.loads(row['evidence_json']);status=result['status']
            write_new_json(run_dir/'TERMINAL.json',result) if not (run_dir/'TERMINAL.json').exists() else None
            index_run(ticket['request'],status)
            if resource_retryable(result):
                self.state['last_conflict_gpu']=ticket['request']['gpu_uuids'][0]
                self.state.update(active_run_id=None,physical_container=False)
                self.event('ABANDONED_RESOURCE_ATTEMPT_RETURN_TO_WATCH',run_id=run,result=status,important=True);return
            if status!='COMPLETE':
                self.event('HALTED_EXPERIMENT_FAILURE_NEEDS_ATTRIBUTION',run_id=run,result=status,important=True);self.stop=True;return
            self.state['cursor']+=1;self.state['attempt']=0;self.state.update(active_run_id=None,physical_container=False)
            self.event('STAGE_COMPLETE_RAW_HANDOFF',run_id=run,stage=self.manifest['stages'][self.state['cursor']-1]['id'],important=True);return
        if row['state']=='UNKNOWN':
            result=response.get('files',{}).get('result.json',{}).get('value',{})
            if (result.get('error')=='UnsafeState: Foreign GPU activity/reservation; stop this run only'
                    and response.get('supervisor_alive') is False and response.get('live_owned_containers')==[]):
                self.event('CONFLICT_RELEASE_RECONCILIATION',run_id=run)
                proof=run_dir/('CONFLICT_RELEASE-'+str(time.time_ns())+'.json')
                with proof.with_suffix('.stdout').open('x') as out,proof.with_suffix('.stderr').open('x') as err:
                    code=subprocess.run([sys.executable,'-B','-m','scripts.reconcile_conflict_release',
                        '--run-id',run,'--out',str(proof)],cwd=ROOT,stdout=out,stderr=err,timeout=330).returncode
                if code==0:self.next_status=0;return
            self.event('WAIT_ORIGINAL_UNKNOWN_NO_NEW_ADMISSION',run_id=run);return
        index_run(ticket['request'],'RUNNING')
        self.event('ORIGINAL_RUN_ACTIVE',run_id=run,physical_container=bool(response.get('live_owned_containers')))

    def accept_next_stages(self):
        generation=len(self.state.get('accepted_stage_batches',[]))+1
        path=self.directory/f'NEXT_STAGES-{generation:04}.json'
        if not path.exists():return
        require_coordinator_epoch(self.manifest)
        batch=json.loads(path.read_bytes())
        if batch.get('parent_manifest_sha256')!=self.state['manifest_sha256']:
            raise ValueError('Next-stage handoff belongs to another queue')
        stages=batch.get('stages')
        if not isinstance(stages,list) or not stages:raise ValueError('Empty next-stage handoff')
        seen={s['id'] for s in self.manifest['stages']}
        for stage in stages:
            validate_stage(stage,self.manifest['runtime_contract'])
            if stage['id'] in seen:raise ValueError('Duplicate stage handoff')
            seen.add(stage['id'])
        self.state.setdefault('accepted_stage_batches',[]).append({'path':str(path),'sha256':sha(path)})
        self.manifest['stages'].extend(stages)
        self.event('ACCEPTED_FROZEN_NEXT_STAGES',stages=[s['id'] for s in stages],important=True)

    def run(self):
        try:
            self.event('READONLY_WATCH_NO_GPU_RESERVATION')
            while not self.stop and now()<dt.datetime.fromisoformat(self.manifest['not_after_utc']):
                approval,current_sha=load_approval_snapshot(TRUSTED_APPROVAL)
                if current_sha!=self.approval_sha:raise RuntimeError('Authority changed; reconcile original before any continuation')
                validate_execution_authority(approval,now())
                if self.probe is None and time.monotonic()>=self.next_connect:
                    self.connections+=1
                    m=dict(self.manifest,_approval=approval)
                    self.probe=Probe(self.transport,m,self.directory,self.connections)
                if self.probe is not None:
                    sample,received=self.probe.sample()
                    stale=time.monotonic()-(received or self.probe.started)>30
                    if sample and sample.get('ok') is not True:
                        self.event('GPU_QUERY_UNKNOWN_NO_ADMISSION',error_detail=sample.get('error','')[:200])
                    if sample and time.monotonic()-received<=10:
                        atomic_json(self.directory/'LATEST_GPU.json',sample)
                        free=idle_candidates(sample,approval['allowed_gpu_uuids'],self.manifest['expected_boot_id'],now())
                        signature=tuple(free)
                        if signature!=self.last_signature:
                            self.last_signature=signature;self.event('GPU_AVAILABILITY_CHANGED',idle_gpu_uuids=free,important=True)
                        self.backoff=5
                        if not self.state.get('active_run_id'):
                            if self.state['cursor']>=len(self.manifest['stages']):
                                self.accept_next_stages()
                            if self.state['cursor']>=len(self.manifest['stages']):
                                self.event('WATCHING_AWAIT_MAIN_NEXT_QUALIFIED_PHASE')
                            elif free:
                                stage=self.manifest['stages'][self.state['cursor']]
                                targets=[u for u in free if u in stage.get('allowed_gpu_uuids',approval['allowed_gpu_uuids'])]
                                targets.sort(key=lambda u:u==self.state.get('last_conflict_gpu'))
                                if targets:self.launch(targets[0])
                    if stale or self.probe.error or self.probe.process.poll() is not None:
                        detail=self.probe.error;self.probe.close();self.probe=None
                        self.event('READONLY_STREAM_RECOVERY_NO_STALE_ADMISSION',error_detail=detail)
                        self.next_connect=time.monotonic()+self.backoff;self.backoff=min(60,self.backoff*2)
                if self.state.get('active_run_id'):
                    try:self.active()
                    except Exception as error:
                        self.event('READONLY_RECONCILE_ERROR_NO_NEW_ADMISSION',error_detail=str(error)[-400:])
                        self.next_status=time.monotonic()+30
                if time.monotonic()-self.last_heartbeat>=1200:
                    self.last_heartbeat=time.monotonic()
                    self.event(self.state['status'],heartbeat=True,cursor=self.state['cursor'],run_id=self.state.get('active_run_id'),important=True)
                time.sleep(1)
        except Exception as error:
            self.event('HALTED_FATAL_NO_NEW_ADMISSION',error_detail=type(error).__name__+': '+str(error)[-400:],important=True)
            raise
        finally:
            if self.probe is not None:self.probe.close()
            exit_path=self.directory/'EXIT.json'
            if exit_path.exists():exit_path=self.directory/('EXIT-'+str(time.time_ns())+'.json')
            write_new_json(exit_path,{'utc':now().isoformat(),'active_run_id':self.state.get('active_run_id'),
                'original_run_must_be_reconciled':bool(self.state.get('active_run_id')),'GPU_mutations_replayed':False})


def validate_stage(stage: dict,runtime_contract: dict | None=None):
    path=Path(stage['package'])
    if (re.fullmatch('[a-z0-9-]{1,24}',stage['id']) is None
            or path.resolve(strict=True)!=path or not path.is_relative_to(ROOT)
            or sha(path)!=stage['package_sha256']):raise ValueError('Invalid/changed frozen stage')
    if runtime_contract is not None:
        assets=json.loads(path.read_bytes())['assets']
        if any(assets.get(name,{}).get('sha256')!=expected for name,expected in runtime_contract.items()):
            raise ValueError('Stage lacks the tested fast-admission runtime')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--state-dir',type=Path,required=True);p.add_argument('--resume',action='store_true');args=p.parse_args()
    manifest=json.loads(args.manifest.read_bytes());approval,_=load_approval_snapshot(TRUSTED_APPROVAL)
    end=dt.datetime.fromisoformat(manifest['not_after_utc'])
    if (manifest.get('schema')!='fast-idle-research-queue-v1' or manifest.get('poll_seconds')!=5
            or manifest.get('main_review_seconds')!=1200 or not now()<end<=now()+dt.timedelta(hours=24)
            or end>dt.datetime.fromisoformat(approval['valid_until_utc'])
            or re.fullmatch('[a-z0-9-]{1,32}',manifest.get('campaign','')) is None):raise ValueError('Invalid finite user queue')
    if manifest.get('runner_sha256')!=sha(Path(__file__)):
        raise ValueError('Reviewed queue runner changed')
    require_coordinator_epoch(manifest)
    runtime_contract={'runtime/kv_controller/'+name:sha(ROOT/'scripts/controller'/name)
                      for name in ('idle_admission.py','supervisor.py','nextgen.py')}
    if manifest.get('runtime_contract')!=runtime_contract:
        raise ValueError('Tested controller runtime changed')
    if not isinstance(manifest.get('stages'),list) or not manifest['stages']:raise ValueError('No frozen stages')
    for stage in manifest['stages']:validate_stage(stage,runtime_contract)
    if len({s['id'] for s in manifest['stages']})!=len(manifest['stages']):raise ValueError('Duplicate stage')
    directory=args.state_dir.resolve();secure_directory(directory)
    lock=os.open(ROOT/'state/controller/fast_idle_queue.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_file=directory/'STATE.json'
        if state_file.exists():
            if not args.resume:raise ValueError('Original queue exists; explicit read-only recovery required')
            state=json.loads(state_file.read_bytes())
            if state['manifest_sha256']!=sha(args.manifest):raise ValueError('Queue manifest changed')
            for batch in state.get('accepted_stage_batches',[]):
                path=Path(batch['path'])
                if sha(path)!=batch['sha256']:raise ValueError('Accepted stage handoff changed')
                manifest['stages'].extend(json.loads(path.read_bytes())['stages'])
            if state.get('active_run_id') and not (ROOT/'state/controller/tickets'/(state['active_run_id']+'.json')).exists():
                raise ValueError('Ambiguous local original dispatch; do not relaunch')
        else:
            if args.resume:raise ValueError('No original queue to resume')
            state={'manifest_sha256':sha(args.manifest),'cursor':0,'attempt':0,'active_run_id':None}
        runner=Runner(manifest,directory,state)
        signal.signal(signal.SIGTERM,lambda *_:setattr(runner,'stop',True))
        write_new_json(directory/('START-'+str(time.time_ns())+'.json'),process_identity(os.getpid()))
        runner.run()
    finally:os.close(lock)


if __name__=='__main__':main()
