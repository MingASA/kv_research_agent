"""Clock-driven, read-only original-ticket monitoring. Stdout only; no launches.

The main agent prepares the exact SSH argv/pin and owns state/stop-file writes.
This reader never creates receipts, changes ledger/state, or writes remotely.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.controller.transport import wire_request

ALLOWED={'nextgen-bf16-qwen38-mixed8192-tools32-r0-20261030zc',
         'nextgen-fp8fa2-qwen38-mixed8192-tools32-r0-20261030za',
         'nextgen-nvfp4-qwen38-mixed8192-tools32-r0-20261030zb'}


def emit(value):print(json.dumps(value,ensure_ascii=False),flush=True)
def utc():return datetime.now(timezone.utc).isoformat()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['ssh-config','stop-file']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--first-check-utc',required=True);p.add_argument('--until-utc',required=True);a=p.parse_args()
    first=datetime.fromisoformat(a.first_check_utc.replace('Z','+00:00')).timestamp()
    deadline=datetime.fromisoformat(a.until_utc.replace('Z','+00:00')).timestamp()
    assert 0<deadline-time.time()<=4*3600
    config=json.loads(a.ssh_config.read_text());source=(ROOT/'scripts/controller/remote_rpc.py').read_text()
    assert hashlib.sha256(source.encode()).hexdigest()==config['remote_status_source_sha256']
    argv=config['ssh_argv'];assert argv[0]=='/usr/bin/ssh' and 'StrictHostKeyChecking=yes' in argv
    next_check=time.monotonic()+max(0,first-time.time());count=0
    emit({'event':'MONITOR_STARTED','utc':utc(),'until_utc':a.until_utc,'period_seconds':300,'no_local_or_remote_writes':True})
    while time.time()<deadline and count<48:
        if a.stop_file.exists():emit({'event':'MAIN_STOP_RECEIPT_SEEN','utc':utc(),'stop_receipt':json.loads(a.stop_file.read_text())});return
        if time.monotonic()<next_check:
            time.sleep(min(30,next_check-time.monotonic(),max(.1,deadline-time.time())));continue
        began=time.monotonic();started=utc();state=json.loads((ROOT/'state/STATE.json').read_text())
        run=state.get('active_run_id') or state['monitoring']['resource_status_anchor_run_id'];assert run in ALLOWED
        ticket=json.loads((ROOT/'state/controller/tickets'/(run+'.json')).read_text())
        command,wire=wire_request(source,{'operation':'status','ticket':ticket});response=None
        for attempt in range(3):
            try:
                r=subprocess.run(argv+[command],input=wire,capture_output=True,timeout=60)
                if r.returncode:raise RuntimeError('SSH exit'+str(r.returncode)+': '+r.stderr.decode(errors='replace')[-800:])
                assert len(r.stdout)<=2*2**20;response=json.loads(r.stdout);assert response.get('ok') is True
                break
            except (OSError,ValueError,AssertionError,RuntimeError,subprocess.SubprocessError) as error:
                emit({'event':'READONLY_SSH_RECOVERY','utc':utc(),'run_id':run,'attempt':attempt+1,'error':str(error)[-1000:]})
                if attempt==2:emit({'event':'READONLY_RECOVERY_EXHAUSTED','utc':utc(),'run_id':run});raise SystemExit(1)
                time.sleep(15*(2**attempt))
        progress={}
        for arm,values in response.get('nextgen_progress',{}).items():
            v=values.get('validation_result.json',{}).get('value',{})
            progress[arm]={'files':list(values),'known_answers_checked':v.get('known_answers_checked'),
                'Graph':v.get('actual_model_Graph_replay_logged')}
            for key in ['tool_API_admission.json','tool_generations/generation_summary.json']:
                if key in values:progress[arm][key]=values[key]
        emit({'event':'ORIGINAL_READONLY_S0','started_utc':started,'finished_utc':utc(),'run_id':run,
            'status':response.get('status'),'supervisor_alive':response.get('supervisor_alive'),
            'live_owned_containers':response.get('live_owned_containers'),
            'gpu_snapshot':response.get('gpu_snapshot'),'progress':progress,
            'response_sha256':hashlib.sha256(r.stdout).hexdigest(),'elapsed_seconds':time.monotonic()-began})
        count+=1;next_check=began+300
    emit({'event':'MONITOR_DEADLINE_OR_COUNT_REACHED','utc':utc(),'checks':count})


if __name__=='__main__':main()
