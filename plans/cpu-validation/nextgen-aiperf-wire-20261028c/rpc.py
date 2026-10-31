"""Pinned, GPU-invisible CPU transport validation in a new private scratch root."""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def _require(condition: bool, message: str) -> None:
    if not condition:raise ValueError(message)


def _owned(path: Path) -> None:
    _require(path.resolve(strict=True)==path and path.stat().st_uid==os.getuid(),'Noncanonical/foreign project path')


def _new(path: Path, data: bytes) -> None:
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())


def _json(path: Path, value: dict) -> None:
    _new(path,(json.dumps(value,sort_keys=True,indent=2)+'\n').encode())


def run(payload: dict) -> dict:
    a,m,operation=payload['approval'],payload['manifest'],payload['operation']
    expected=a['approved_remote_identity']
    _require((os.getuid(),os.getgid(),socket.gethostname())==(expected['uid'],expected['gid'],expected['hostname']),
             'Remote identity differs')
    base=Path(a['approved_workspace']);_owned(base)
    _require(m['id'] in ('nextgen-aiperf-wire-20261028a','nextgen-aiperf-wire-20261028b',
                        'nextgen-aiperf-wire-20261028c')
             and operation in ('inspect','prepare','start','status'),
             'Unknown CPU wire operation')
    root=base/'agent_work/scratch'/m['id'];name='kvra-cpu-'+m['id']
    if operation=='status':
        _owned(root);result={'root':str(root),'files':{},'remote_writes':False}
        for filename in ('manifest.json','PREPARED.json','START_INTENT.json','CPU_ADMISSION.json','wire_acceptance.json','CPU_ERROR.json'):
            path=root/filename
            if path.exists():_owned(path);result['files'][filename]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'value':json.loads(path.read_text())}
        inspected=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=15)
        if inspected.returncode==0:
            value=json.loads(inspected.stdout)[0]
            _require(value['Config']['Labels'].get('kvra.cpu_wire_id')==m['id'],'Unknown CPU container')
            result['container']={key:value['State'][key] for key in ('Status','Pid','Running','ExitCode','OOMKilled')}
            result['logs']=subprocess.check_output(['docker','logs','--tail','30',value['Id']],stderr=subprocess.STDOUT,text=True,timeout=15)[-6000:]
        return result
    _require(a['enabled'] and a['allow_remote_user_workspace_writes'],'No project-write authority')
    expiry=datetime.datetime.fromisoformat(a['valid_until_utc'].replace('Z','+00:00'))
    _require(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=900)<expiry,'CPU wire would exceed authority expiry')
    def ticks():return list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    first=ticks();time.sleep(2);last=ticks();delta=[b-c for b,c in zip(last,first)];idle=delta[3]/sum(delta)
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    _require(idle>=.25 and available>=256*2**30,'Shared CPU/RAM headroom insufficient; leave foreign vLLM untouched')
    image=json.loads(subprocess.check_output(['docker','image','inspect',a['approved_image_id']],text=True,timeout=15))[0]
    _require(image['Id']==a['approved_image_id'],'Existing pinned CPU image is unavailable; do not pull/install')
    if operation=='inspect':
        _require(not root.exists(),'Existing CPU wire must be reconciled')
        return {'can_prepare':True,'remote_writes':False,'GPU_requested':False,'CPU_idle_fraction':idle,
                'memory_available_bytes':available}
    if operation=='prepare':
        _require(not root.exists(),'Never repeat CPU prepare')
        _owned(base/'agent_work')
        if not root.parent.exists():root.parent.mkdir(mode=0o700)
        _owned(root.parent);root.mkdir(mode=0o700)
        _json(root/'manifest.json',m)
        total=0
        for relative,asset in payload['assets'].items():
            _require(relative in ('wire.py','driver.py','warmup_chat16.json','chat128.json','warmup_long2.json',
                                  'long_unique16.json','long_prefix16x2.json'),'Unexpected CPU wire input')
            data=base64.b64decode(asset['base64'],validate=True);total+=len(data)
            _require(total<=32*2**20 and hashlib.sha256(data).hexdigest()==m['asset_sha256'][relative],
                     'CPU wire asset size/hash mismatch')
            _new(root/relative,data)
        _require(set(payload['assets'])==set(m['asset_sha256']),'CPU wire assets missing')
        inner='/workspace/agent_work/scratch/'+m['id']
        command=['docker','create','--name',name,'--label','kvra.cpu_wire_id='+m['id'],
                 '--runtime','nvidia','--user',str(os.getuid())+':'+str(os.getgid()),
                 '--network','none','--ipc','private','--shm-size','128m','--cpus','2','--memory','8g','--memory-swap','8g',
                 '--pids-limit','512','--cap-drop','ALL','--security-opt','no-new-privileges',
                 '--security-opt','seccomp=unconfined','--read-only','--tmpfs','/tmp:rw,nosuid,nodev,size=256m,mode=1777',
                 '--mount','type=bind,src='+str(base)+',dst=/workspace,readonly',
                 '--mount','type=bind,src='+str(base)+',dst='+str(base)+',readonly',
                 '--mount','type=bind,src='+str(root)+',dst='+inner,
                 '-e','HOME=/tmp','-e','NVIDIA_VISIBLE_DEVICES=none','-e','CUDA_VISIBLE_DEVICES=',
                 '-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1',
                 '-e','OPENBLAS_NUM_THREADS=1','-e','NUMEXPR_NUM_THREADS=1',
                 '--entrypoint','/workspace/.venv-aiperf/bin/python',a['approved_image_id'],'-B',inner+'/driver.py',inner]
        if m['id']=='nextgen-aiperf-wire-20261028c':
            model=Path(a['read_only_model_path'])
            _require(model.resolve(strict=True)==model and model.is_dir()
                     and model.name=='Qwen3.8-27B','CPU CLI tokenizer must use the approved existing model read-only')
            position=command.index('--entrypoint')
            command[position:position]=['--mount','type=bind,src='+str(model)+',dst=/models/Qwen3.8-27B,readonly',
                                       '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
                                       '-e','TOKENIZERS_PARALLELISM=false']
        _json(root/'docker_command.json',{'argv':command})
        identifier=subprocess.check_output(command,text=True,timeout=30).strip()
        _json(root/'PREPARED.json',{'container_id':identifier,'GPU_requested':False})
        return {'status':'PREPARED_CPU_WIRE_NO_GPU','container_id':identifier}
    _owned(root)
    prepared=json.loads((root/'PREPARED.json').read_text())
    value=json.loads(subprocess.check_output(['docker','inspect',prepared['container_id']],text=True,timeout=15))[0]
    _require(value['State']['Status']=='created' and value['Config']['Labels'].get('kvra.cpu_wire_id')==m['id']
             and not value['HostConfig'].get('DeviceRequests') and not value['HostConfig'].get('Devices')
             and not value['HostConfig']['Privileged'],'CPU container already started or has GPU/privileged devices')
    _json(root/'START_INTENT.json',{'operation':'start_exact_CPU_wire_once','GPU_requested':False})
    subprocess.check_output(['docker','start',value['Id']],text=True,timeout=30)
    return {'status':'STARTED_CPU_WIRE_NO_GPU','container_id':value['Id'],'CPU':2,'memory_GiB':8}


if __name__=='__main__':
    try:result={'ok':True,**run(json.load(sys.stdin))}
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:result={'ok':False,'error':str(error)[-2500:]}
    sys.stdout.write(json.dumps(result)+'\n')
