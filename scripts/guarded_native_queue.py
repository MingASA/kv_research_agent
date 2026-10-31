"""Finite local sequential queue over the existing guarded experiment CLI.

GPU snapshots only nominate a candidate. The remote supervisor retains full
120s admission, ownership checks, conflict watchdog and per-job deadline.
Mutations are never repeated. A failure or ambiguity stops further admission.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time

from scripts.controller.approval import TRUSTED_APPROVAL, load_approval_snapshot, validate_scope
from scripts.controller.cli import _local_package, validate_frozen_policy
from scripts.controller.durable import atomic_json, fsync_directory, secure_directory, write_new_json
from scripts.controller.owned_result_mirror import mirror_terminal
from scripts.controller.policy import validate_execution_authority, validate_request
from scripts.controller.quota import QuotaLedger
from scripts.controller.transport import PinnedSSH


ROOT=Path(__file__).resolve().parents[1]
GPU_SOURCE=r'''
import json,os,socket,subprocess,sys
from pathlib import Path
p=json.load(sys.stdin);a=p['approval'];e=a['approved_remote_identity']
if (os.getuid(),os.getgid(),socket.gethostname())!=(e['uid'],e['gid'],e['hostname']):raise ValueError('Remote identity differs')
base=Path(a['approved_workspace'])
if base.resolve()!=base or base.stat().st_uid!=os.getuid():raise ValueError('Project BASE changed')
rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
gpus={}
for line in rows.splitlines():
 u,m,v=[x.strip() for x in line.split(',')];gpus[u]={'memory_mib':int(m),'util_pct':int(v)}
raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15)
apps=[]
for line in raw.splitlines():
 u,n=[x.strip() for x in line.split(',')];proc=Path('/proc')/n;uid=None;comm='unknown'
 try:uid=proc.stat().st_uid;comm=(proc/'comm').read_text().strip()
 except FileNotFoundError:pass
 apps.append({'gpu_uuid':u,'pid':int(n),'uid':uid,'comm':comm})
print(json.dumps({'ok':True,'gpus':gpus,'apps':apps,'remote_writes':False,
 'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))
'''


class QueueAlreadyRunning(RuntimeError):
    """A second runner must not modify the first runner's state."""


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _utc(value: str) -> datetime.datetime:
    parsed=datetime.datetime.fromisoformat(value.replace('Z','+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset()!=datetime.timedelta(0):
        raise ValueError('Queue deadlines must explicitly use UTC')
    return parsed


def _sha(path: Path) -> str:
    hasher=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(8*2**20),b''):hasher.update(data)
    return hasher.hexdigest()


def coarse_gpu_gate(snapshot: dict, requested: list[str], expected_boot: str) -> tuple[bool,str]:
    if (snapshot.get('ok') is not True or snapshot.get('remote_writes') is not False
            or snapshot.get('boot_id')!=expected_boot):
        raise ValueError('Unverified snapshot or changed remote boot; reconcile before admission')
    if any(row.get('uid')==0 and row.get('comm','').startswith('VLLM::Worker')
           for row in snapshot.get('apps',[])):
        return False,'FOREIGN_ROOT_VLLM_ALLOCATION_STILL_PRESENT'
    for uuid in requested:
        gpu=snapshot.get('gpus',{}).get(uuid)
        if gpu is None or gpu.get('memory_mib')!=0 or gpu.get('util_pct')!=0:
            return False,'SELECTED_GPU_NOT_ZERO_IDLE'
        if any(row.get('gpu_uuid')==uuid for row in snapshot.get('apps',[])):
            return False,'SELECTED_GPU_HAS_COMPUTE_PID'
    return True,'COARSE_CANDIDATE_FULL_120S_ADMISSION_STILL_REQUIRED'


def action_for_existing(state: str | None) -> str:
    if state in ('RUNNING','DISPATCHING'):return 'RECONCILE_ONLY'
    if state=='SETTLED':return 'POSTPROCESS_TERMINAL'
    return 'HALT_UNKNOWN_OR_UNDISPATCHED_NO_RELAUNCH'


def validate_queue_manifest(manifest: dict) -> None:
    if (manifest.get('schema')!='kv-guarded-sequential-queue-v1'
            or not isinstance(manifest.get('jobs'),list) or len(manifest['jobs'])!=2):
        raise ValueError('Expected exactly the two reviewed core-quality experiments')
    if len({job['run_id'] for job in manifest['jobs']})!=len(manifest['jobs']):
        raise ValueError('Duplicate queue run identity')
    if type(manifest.get('poll_seconds')) is not int or not 60<=manifest['poll_seconds']<=1800:
        raise ValueError('Queue poll must be a bounded 60-1800 seconds')
    now=_now();end=_utc(manifest['not_after_utc'])
    if not now<end<=now+datetime.timedelta(hours=48):
        raise ValueError('Queue admission window must be finite and within48h')
    approval,_=load_approval_snapshot(TRUSTED_APPROVAL)
    validate_execution_authority(approval,now)
    if end>_utc(approval['valid_until_utc']):raise ValueError('Queue extends beyond authority')
    if not manifest.get('expected_boot_id'):raise ValueError('Missing observed host boot identity')
    formats=[]
    for job in manifest['jobs']:
        path=Path(job['package'])
        if path.resolve(strict=True)!=path or _sha(path)!=job['package_sha256']:
            raise ValueError('Frozen queue package changed')
        request,_,_=_local_package(path)
        if (request['run_id']!=job['run_id'] or request['workload']['kind']!='native_hybrid_core_quality'
                or request['arms'] not in (['fp8_e4m3'],['nvfp4']) or len(request['gpu_uuids'])!=1):
            raise ValueError('Only the reviewed single-GPU native core-quality formats are admitted')
        formats.append(request['arms'][0]);validate_request(request,approval)
        validate_frozen_policy(path,request,approval)
    if formats!=['fp8_e4m3','nvfp4']:raise ValueError('Expected FP8 followed by NVFP4')
    baseline=Path(manifest['baseline_scores'])/'summary.json'
    if _sha(baseline)!=manifest['baseline_summary_sha256']:
        raise ValueError('Verified BF16 baseline scores changed')
    for item in manifest['pinned_local_files']:
        path=Path(item['path'])
        if path.resolve(strict=True)!=path or _sha(path)!=item['sha256']:
            raise ValueError('Queue analysis source or independent input changed')
    executable=Path(manifest['analysis_python'])
    if not executable.is_absolute() or not executable.is_file() or not os.access(executable,os.X_OK):
        raise ValueError('Analysis Python is unavailable')


def _replace_text(path: Path, content: str) -> None:
    if path.is_symlink():raise ValueError('Refusing to replace a symlink')
    fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            stream.write(content);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path);fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def _publish(directory: Path, state: dict, previous_status: str | None) -> None:
    """Publish actual ledger identities only; never index unlaunched proposals."""
    with QuotaLedger(ROOT/'state/controller') as ledger:
        active=[dict(row) for row in ledger.connection.execute(
            "SELECT run_id,state,request_json FROM runs WHERE state IN ('RESERVED','DISPATCHING','RUNNING','UNKNOWN')")]
        current=state.get('current_run_id')
        row=ledger.connection.execute('SELECT * FROM runs WHERE run_id=?',(current,)).fetchone() if current else None
        row=dict(row) if row else None
    document=json.loads((ROOT/'state/STATE.json').read_text())
    document['automatic_queue']={'directory':str(directory),'status':state['status'],'cursor':state['cursor'],
        'current_run_id':current,'updated_utc':state['updated_utc'],'not_after_utc':state.get('not_after_utc')}
    document['active_run_ids']=[r['run_id'] for r in active]
    document['active_run_id']=document['active_run_ids'][0] if active else None
    document['active_gpu_uuids']=list(dict.fromkeys(g for r in active for g in json.loads(r['request_json'])['gpu_uuids']))
    document['local_controller_running']=not state['status'].startswith(('HALTED','COMPLETE_ALL'))
    document['updated_at_utc']=state['updated_utc']
    atomic_json(ROOT/'state/STATE.json',document)
    if previous_status!=state['status']:
        with (ROOT/'state/EVENTS.jsonl').open('a') as stream:
            stream.write(json.dumps({'utc':state['updated_utc'],'type':'guarded_sequential_queue',
                'status':state['status'],'run_id':current,'queue_directory':str(directory)},ensure_ascii=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
        plan=ROOT/'plans/NEXT_ACTIONS.md';text=plan.read_text();start='<!-- GUARDED_QUEUE_BEGIN -->';end='<!-- GUARDED_QUEUE_END -->'
        block=(start+'\n自动顺序队列：`'+str(directory)+'`；实际状态 `'+state['status']+'`，cursor='+str(state['cursor'])
               +'。以该目录STATE/EVENTS与原run票据为准；禁止手动重复launch；失败/未知停止后续。\n'+end+'\n\n')
        if start in text:
            before,rest=text.split(start,1);_,after=rest.split(end,1);text=before+block+after.lstrip('\n')
        else:text=block+text
        _replace_text(plan,text)
    if row is None:return
    request=json.loads(row['request_json']);path=ROOT/'results/INDEX.csv'
    with path.open(newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    matches=[r for r in rows if r['run_id']==current]
    if len(matches)>1:raise ValueError('Duplicate existing run index')
    entry=matches[0] if matches else {key:'' for key in fields}
    indexed_status=('COMPLETE_RAW_SCORED' if state['status']=='ONE_EXPERIMENT_RAW_AND_SCORES_VERIFIED' else
                    'QUEUE_HALTED_RECONCILE_ORIGINAL' if state['status'].startswith('HALTED') else row['state'])
    entry.update(run_id=current,status=indexed_status,model=request['model_id'],
        backend=json.dumps(request['operators'],sort_keys=True),git_sha=request['python_source_commit'],
        binary_sha256=request['binary_sha256'],dataset_sha256=request['quality_evaluation']['requests_sha256'],
        gpu_uuid='|'.join(request['gpu_uuids']),concurrency=str(request['quality_evaluation']['client_concurrency']),
        manifest_path=request['remote_output']+'/manifest.json',metric_path=state.get('independent_scores',''),
        validity='quality_raw_pending_or_scored_no_automatic_acceptance',
        notes='Sequential guarded quality only; no AIPerf claim. '+state['status'])
    if not matches:rows.append(entry)
    output=io.StringIO(newline='');writer=csv.DictWriter(output,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    _replace_text(path,output.getvalue())


def _record(directory: Path, state: dict, status: str, **details) -> None:
    previous=state.get('status');state.update(status=status,updated_utc=_now().isoformat(),**details)
    atomic_json(directory/'STATE.json',state)
    with (directory/'EVENTS.jsonl').open('a') as stream:
        stream.write(json.dumps({'utc':state['updated_utc'],'status':status,**details},ensure_ascii=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    _publish(directory,state,previous)


def _process_identity(pid: int) -> dict:
    root=Path('/proc')/str(pid);text=(root/'stat').read_text();tail=text[text.rfind(')')+2:].split()
    return {'pid':pid,'uid':root.stat().st_uid,'start_ticks':int(tail[19]),'process_state':tail[0],
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'argv':(root/'cmdline').read_bytes().split(b'\0')[:-1]}


def _child_alive(state: dict) -> bool:
    child=state.get('CLI_child')
    if not child:return False
    try:observed=_process_identity(child['pid'])
    except FileNotFoundError:return False
    if observed['boot_id']!=child['boot_id'] or observed['start_ticks']!=child['start_ticks']:return False
    if observed['uid']!=child['uid']:raise ValueError('Local CLI child ownership differs')
    if observed['process_state']=='Z':return False
    if [part.decode() for part in observed['argv']]!=child['argv']:
        raise ValueError('Original local CLI process changed identity')
    return True


def _cli(directory: Path, label: str, arguments: list[str], state: dict) -> tuple[int,dict]:
    output=directory/(label+'.json');errors=directory/(label+'.stderr.log')
    if not arguments or arguments[0] not in ('launch','preflight','reconcile'):
        raise ValueError('Unreviewed queue CLI operation')
    if arguments[0]=='launch':
        if len(arguments)!=3 or arguments[1]!='--package':raise ValueError('Unexpected queue launch arguments')
        # The wrapper owns the original CLI receipts. Never pre-open or truncate
        # those files here, even if the wrapper refuses an existing run intent.
        stdout_path=directory/(label+'.wrapper.json')
        stderr_path=directory/(label+'.wrapper.stderr.log')
        argv=[sys.executable,'-B','-m','scripts.launch_guarded_once','--package',arguments[2],
              '--receipt',str(output),'--stderr',str(errors)]
    else:
        stdout_path,stderr_path=output,errors
        argv=[sys.executable,'-B','-m','scripts.controller.cli',*arguments]
    if any(path.exists() or path.is_symlink() for path in (output,errors,stdout_path,stderr_path)):
        raise ValueError('Existing CLI attempt; never repeat a mutation')
    with stdout_path.open('x') as stdout,stderr_path.open('x') as stderr:
        child=subprocess.Popen(argv,cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True)
        identity=_process_identity(child.pid);identity['argv']=argv
        _record(directory,state,'CLI_PROCESS_RUNNING',CLI_child=identity,CLI_operation=arguments[0],CLI_output=str(output))
        # Timeout preserves the child and its durable identity. Never kill it or
        # assume a timed-out prepare/start did not reach the remote server.
        code=child.wait(timeout=2100 if arguments[0]=='launch' else 600)
    state['CLI_child']=None
    try:body=json.loads(output.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        body={'error':'CLI returned no complete JSON; inspect wrapper/original intent, do not repeat mutation'}
    return code,body


def _postprocess(manifest: dict, directory: Path, job: dict, ticket: dict, transport) -> dict:
    rpc_source=(ROOT/'scripts/controller/remote_rpc.py').read_text()
    mirror=directory/'mirrors'/job['run_id']
    sha_manifest=mirror_terminal(transport,ticket,rpc_source,mirror)
    scores=directory/'scores'/job['run_id']
    if not (scores/'summary.json').exists():
        subprocess.run([manifest['analysis_python'],'-B','-m','scripts.score_nextgen_core_quality',
            '--sha-manifest',str(sha_manifest),'--cohort',manifest['quality_cohort'],'--out',str(scores)],
            cwd=ROOT,check=True,timeout=600)
    comparison=directory/('paired-'+job['run_id']+'.json')
    if not comparison.exists():
        subprocess.run([manifest['analysis_python'],'-B','-m','scripts.compare_nextgen_quality',
            '--reference',manifest['baseline_scores'],'--candidate',str(scores),'--out',str(comparison)],
            cwd=ROOT,check=True,timeout=600)
    summary=json.loads((scores/'summary.json').read_text())
    if summary['run_id']!=job['run_id'] or summary['all_raw_API_and_usage_contracts_independently_verified'] is not True:
        raise ValueError('Independent quality score verification missing')
    return {'raw_sha_manifest':str(sha_manifest),'independent_scores':str(scores/'summary.json'),
            'paired_comparison':str(comparison),'production_quality_accepted':False}


def _cycle(manifest: dict, directory: Path, state: dict) -> bool:
    """Return True only when the queue must stop; at most one launch per cycle."""
    if _child_alive(state):
        _record(directory,state,'WAIT_ORIGINAL_LOCAL_CLI_PROCESS');return False
    job=manifest['jobs'][state['cursor']]
    ticket_path=ROOT/'state/controller/tickets'/(job['run_id']+'.json')
    with QuotaLedger(ROOT/'state/controller') as ledger:
        row=ledger.connection.execute('SELECT state FROM runs WHERE run_id=?',(job['run_id'],)).fetchone()
        approval,_=load_approval_snapshot(TRUSTED_APPROVAL);usage=ledger.usage(approval)
        active=list(ledger.connection.execute("SELECT run_id,state FROM runs WHERE state IN ('RESERVED','DISPATCHING','RUNNING','UNKNOWN')"))
    if state.get('current_run_id') is not None or row is not None or ticket_path.exists():
        if row is None or not ticket_path.exists():
            _record(directory,state,'HALTED_UNCERTAIN_ORIGINAL_LAUNCH_INTENT');return True
        if action_for_existing(row['state']).startswith('HALT'):
            _record(directory,state,'HALTED_UNKNOWN_NO_RELAUNCH',ledger_state=row['state']);return True
        state['current_run_id']=job['run_id']
        count=state.get('reconcile_count',0)+1;state['reconcile_count']=count
        code,status=_cli(directory,f'reconcile-{state["cursor"]}-{count}', ['reconcile','--run-id',job['run_id']],state)
        result=status.get('files',{}).get('result.json',{}).get('value',{})
        if (not code and status.get('status')=='COMPLETE' and result.get('gpu_release_verified') is True
                and status.get('live_owned_containers')==[]):
            ticket=json.loads(ticket_path.read_text());transport=PinnedSSH(ticket['approval'],ROOT/'state/controller/pins'/ticket['reservation_id'])
            _record(directory,state,'POSTPROCESSING_COMPLETE_RAW')
            details=_postprocess(manifest,directory,job,ticket,transport)
            if not any(x['run_id']==job['run_id'] for x in state.get('completed',[])):
                state.setdefault('completed',[]).append({'run_id':job['run_id'],**details})
            _record(directory,state,'ONE_EXPERIMENT_RAW_AND_SCORES_VERIFIED',**details)
            state['cursor']+=1;state['current_run_id']=None;state['independent_scores']=''
            _record(directory,state,'READY_FOR_NEXT_EXPERIMENT')
            return False
        if (code or status.get('status') in ('FAILED','CONFLICT','TIMED_OUT','DISK_LIMIT','ABSENT','UNKNOWN')
                or status.get('error') or status.get('supervisor_alive') is not True):
            _record(directory,state,'HALTED_EXPERIMENT_FAILURE_OR_UNKNOWN',original_status=status.get('status'),error=status.get('error'))
            return True
        _record(directory,state,'RUNNING_ORIGINAL_EXPERIMENT');return False
    if _now()>=_utc(manifest['not_after_utc']):
        _record(directory,state,'HALTED_FINITE_ADMISSION_WINDOW_EXPIRED');return True
    validate_execution_authority(approval,_now())
    if active or usage['uncertain_runs']:
        _record(directory,state,'WAIT_OTHER_OWN_OR_UNCERTAIN_JOB',active_run_ids=[x['run_id'] for x in active]);return False
    validate_queue_manifest(manifest)
    request,_,_=_local_package(Path(job['package']));validate_scope(approval,request,usage,_now())
    transport=PinnedSSH(approval,directory/'pin')
    snapshot=transport.rpc(GPU_SOURCE,{'approval':approval},mutation=False,timeout=60)
    state['sample']+=1;write_new_json(directory/f'GPU-S0-{state["sample"]:06d}.json',snapshot)
    allowed,reason=coarse_gpu_gate(snapshot,request['gpu_uuids'],manifest['expected_boot_id'])
    if not allowed:
        _record(directory,state,'WAIT_FOREIGN_GPU_OR_ALLOCATION',reason=reason);return False
    code,preflight=_cli(directory,f'preflight-{state["cursor"]}-{state["sample"]}', ['preflight','--package',job['package']],state)
    if code or not preflight.get('static_scope_valid'):
        _record(directory,state,'HALTED_FRESH_CLI_PREFLIGHT_REFUSAL',error=preflight.get('error'));return True
    _record(directory,state,'LAUNCH_INTENT_DURABLE',current_run_id=job['run_id'])
    code,ack=_cli(directory,f'launch-{state["cursor"]}', ['launch','--package',job['package']],state)
    if code or ack.get('status')!='DISPATCHED':
        _record(directory,state,'HALTED_LAUNCH_REPLY_UNCERTAIN_NO_RETRY',error=ack.get('error'));return True
    _record(directory,state,'DISPATCHED_ONE_EXPERIMENT',supervisor=ack['identity']);return False


def run(manifest_path: Path, directory: Path, *, once: bool = False) -> dict:
    secure_directory(directory)
    lock=os.open(directory/'runner.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        if not stat.S_ISREG(os.fstat(lock).st_mode) or os.fstat(lock).st_uid!=os.getuid():
            raise ValueError('Unsafe local runner lock')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise QueueAlreadyRunning('Existing queue runner retains lock; no state changed') from error
        manifest=json.loads(manifest_path.read_text())
        saved=directory/'STATE.json'
        state=json.loads(saved.read_text()) if saved.exists() else {'cursor':0,'current_run_id':None,'sample':0,
            'queue_manifest_sha256':_sha(manifest_path),'not_after_utc':manifest['not_after_utc']}
        if state.get('status','').startswith(('HALTED','COMPLETE_ALL')):return state
        try:
            if state['queue_manifest_sha256']!=_sha(manifest_path):raise ValueError('Queue manifest changed during recovery')
            validate_queue_manifest(manifest)
            identity=_process_identity(os.getpid());identity['argv']=[x.decode() for x in identity['argv']]
            _record(directory,state,'LOCAL_QUEUE_RUNNING',runner=identity)
            while state['cursor']<len(manifest['jobs']):
                if _cycle(manifest,directory,state) or once:return state
                time.sleep(manifest['poll_seconds'])
            _record(directory,state,'COMPLETE_ALL_QUEUED_RAW_GENERATIONS_AND_SCORES_NO_AUTOMATIC_ACCEPTANCE')
            return state
        except Exception as error:
            _record(directory,state,'HALTED_LOCAL_ERROR_NO_GPU_RETRY',error=type(error).__name__+': '+str(error))
            raise
    finally:os.close(lock)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if args.validate_only:
        validate_queue_manifest(json.loads(args.queue.read_text()))
        print(json.dumps({'static_queue_valid':True,'GPU_launched':False}));return
    try:result=run(args.queue,args.directory,once=args.once)
    except QueueAlreadyRunning as error:
        print(json.dumps({'already_running':True,'state_changed':False,'reason':str(error)}));raise SystemExit(3)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
