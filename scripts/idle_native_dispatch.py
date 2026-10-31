"""Admit exactly one frozen missing-quality run after genuine resource availability.

Before admission this reserves no GPU and writes nothing remotely. The existing
unique launcher/ledger and remote120s/watchdog gates remain authoritative. Any
launch uncertainty stops this dispatcher; it never retries a dispatched run.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from scripts.controller.approval import TRUSTED_APPROVAL,load_approval_snapshot,validate_scope
from scripts.controller.cli import _local_package,validate_frozen_policy
from scripts.controller.durable import atomic_json,secure_directory,write_new_json
from scripts.controller.policy import validate_execution_authority,validate_request
from scripts.controller.quota import QuotaLedger
from scripts.controller.transport import PinnedSSH
from scripts.guarded_native_queue import GPU_SOURCE,coarse_gpu_gate,_process_identity


ROOT=Path(__file__).resolve().parents[1]


def publish(directory: Path,state: dict,status: str,**extra) -> None:
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    state.update(status=status,updated_utc=now,**extra);atomic_json(directory/'STATE.json',state)
    with (directory/'EVENTS.jsonl').open('a') as stream:
        stream.write(json.dumps({'utc':now,'status':status,**extra},ensure_ascii=False)+'\n')
        stream.flush();os.fsync(stream.fileno())


def has_original_intent(run: str) -> bool:
    location=ROOT/'state/controller'
    if (location/'local_launch_once'/(run+'.json')).exists() or (location/'tickets'/(run+'.json')).exists():return True
    if (location/'idle_admission_once'/(run+'.json')).exists():return True
    with QuotaLedger(location) as ledger:
        return bool(ledger.connection.execute('SELECT 1 FROM runs WHERE run_id=?',(run,)).fetchone()
                    or ledger.connection.execute('SELECT 1 FROM history WHERE run_id=?',(run,)).fetchone())


def cycle(manifest: dict,directory: Path,state: dict) -> bool:
    package=Path(manifest['package'])
    if hashlib.sha256(package.read_bytes()).hexdigest()!=manifest['package_sha256']:
        raise ValueError('Frozen admission package changed')
    request,_,_=_local_package(package);run=request['run_id']
    if run!=manifest['run_id']:raise ValueError('Admission manifest run ID differs from frozen package')
    if has_original_intent(run):
        publish(directory,state,'HALTED_EXISTING_ORIGINAL_INTENT_RECONCILE_ONLY');return True
    now=datetime.datetime.now(datetime.timezone.utc)
    if now>=datetime.datetime.fromisoformat(manifest['not_after_utc'].replace('Z','+00:00')):
        publish(directory,state,'HALTED_FINITE_ADMISSION_WINDOW_EXPIRED');return True
    approval,_=load_approval_snapshot(TRUSTED_APPROVAL)
    validate_execution_authority(approval,now);validate_request(request,approval);validate_frozen_policy(package,request,approval)
    if request['workload']['kind']!='native_hybrid_core_quality' or not request.get('quality_completion'):
        raise ValueError('This one-shot dispatcher admits only the reviewed missing-quality continuation')
    with QuotaLedger(ROOT/'state/controller') as ledger:
        active=[dict(row) for row in ledger.connection.execute(
            "SELECT run_id,state FROM runs WHERE state IN ('RESERVED','DISPATCHING','RUNNING','UNKNOWN')")]
        usage=ledger.usage(approval)
    if active or usage['uncertain_runs']:
        publish(directory,state,'WAIT_OTHER_OWN_OR_UNCERTAIN_RUN',active=active);return False
    validate_scope(approval,request,usage,now)
    transport=PinnedSSH(approval,directory/'pin')
    try:sample=transport.rpc(GPU_SOURCE,{'approval':approval},mutation=False,timeout=45)
    except Exception as error:
        publish(directory,state,'WAIT_READONLY_NETWORK_RECOVERY_COOLDOWN',error=str(error)[-1000:]);return False
    state['sample_index']+=1
    sample_path=directory/f'GPU-S0-{state["sample_index"]:06d}.json';write_new_json(sample_path,sample)
    candidate,reason=coarse_gpu_gate(sample,request['gpu_uuids'],manifest['expected_boot_id'])
    if not candidate:
        publish(directory,state,'WAIT_FOREIGN_GPU_OR_RESERVATION',reason=reason,latest_snapshot=str(sample_path));return False
    # A snapshot is only nomination. The actual CLI and remote stable-idle,
    # ownership, reservation, driver, budget and independent-watchdog gates run.
    claims=ROOT/'state/controller/idle_admission_once';secure_directory(claims)
    write_new_json(claims/(run+'.json'),{'run_id':run,'dispatcher_directory':str(directory),
                    'package_sha256':manifest['package_sha256'],'do_not_reenter':True})
    write_new_json(directory/'ADMISSION_INTENT.json',{'run_id':run,'package_sha256':manifest['package_sha256'],
                    'snapshot':str(sample_path),'do_not_repeat':True})
    argv=[sys.executable,'-B','-m','scripts.launch_guarded_once','--package',str(package),
          '--receipt',str(directory/'ORIGINAL_DISPATCH_ACK.json'),'--stderr',str(directory/'ORIGINAL_CLI.stderr.log')]
    with (directory/'LAUNCH_WRAPPER.stdout.json').open('x') as stdout,(directory/'LAUNCH_WRAPPER.stderr.log').open('x') as stderr:
        child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
        identity=_process_identity(child.pid);identity['argv']=argv
        write_new_json(directory/'LOCAL_LAUNCH_PROCESS.json',identity)
        publish(directory,state,'ORIGINAL_LOCAL_LAUNCH_IN_FLIGHT',CLI_child=identity)
        try:code=child.wait(timeout=2100)
        except subprocess.TimeoutExpired:
            publish(directory,state,'HALTED_WAIT_ORIGINAL_LOCAL_LAUNCH_PROCESS',CLI_child=identity,
                    original_receipt=str(directory/'ORIGINAL_DISPATCH_ACK.json'),relaunch_allowed=False)
            return True  # Preserve the still-running original child; no signal/retry.
    ack_path=directory/'ORIGINAL_DISPATCH_ACK.json'
    ack=json.loads(ack_path.read_text()) if ack_path.exists() and ack_path.stat().st_size else {}
    if code or ack.get('status')!='DISPATCHED':
        publish(directory,state,'HALTED_ORIGINAL_LAUNCH_REPLY_REFUSED_OR_UNKNOWN',local_exit_code=code,
                original_reply_status=ack.get('status'));return True
    with QuotaLedger(ROOT/'state/controller') as ledger:row=ledger.get(run)
    if row['state']!='RUNNING':raise ValueError('Dispatch ACK lacks its exact durable running ledger row')
    # The main agent owns shared project STATE.json. Keep this handoff durable
    # here; the monitor/main reads the exact ticket/ledger before publishing it,
    # avoiding competing read/modify/write updates to the main project state.
    publish(directory,state,'DISPATCHED_HANDOFF_TO_MAIN_AND_READONLY_MONITOR',supervisor=ack['identity'],
            ledger_state=row['state'],ticket_path=str(ROOT/'state/controller/tickets'/(run+'.json')))
    return True


def run(manifest_path: Path,directory: Path) -> None:
    secure_directory(directory)
    with (directory/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (directory/'STATE.json').exists():raise ValueError('Original dispatcher exists;inspect its state,do not restart')
        manifest=json.loads(manifest_path.read_text())
        if (manifest.get('schema')!='native-one-shot-idle-admission-v1' or manifest.get('poll_seconds')!=300
                or not Path(manifest['package']).is_absolute()):raise ValueError('Unreviewed finite idle dispatcher')
        if manifest.get('runner_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError('The reviewed idle-dispatch runner changed before startup')
        end=datetime.datetime.fromisoformat(manifest['not_after_utc'].replace('Z','+00:00'))
        now=datetime.datetime.now(datetime.timezone.utc)
        if end.tzinfo is None or end.utcoffset()!=datetime.timedelta(0) or not now<end<=now+datetime.timedelta(hours=48):
            raise ValueError('Admission deadline must be explicit finite UTC within48h')
        run_id=manifest.get('run_id','')
        if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}',run_id) is None:raise ValueError('Unsafe dispatcher run ID')
        locks=ROOT/'state/controller/idle_dispatch_locks';secure_directory(locks)
        global_fd=os.open(locks/(run_id+'.lock'),os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        try:
            fcntl.flock(global_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            identity=_process_identity(os.getpid());identity['argv']=[value.decode() for value in identity['argv']]
            write_new_json(directory/'LOCAL_STARTED.json',identity)
            state={'run_id':manifest['run_id'],'sample_index':0,'manifest':str(manifest_path)}
            publish(directory,state,'READONLY_WAIT_NO_GPU_RESERVATION')
            while True:
                started=time.monotonic()
                if cycle(manifest,directory,state):return
                time.sleep(max(1,300-(time.monotonic()-started)))
        finally:os.close(global_fd)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--state-dir',type=Path,required=True);args=parser.parse_args()
    run(args.manifest,args.state_dir)


if __name__=='__main__':main()
