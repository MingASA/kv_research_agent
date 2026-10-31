"""One isolated CPU-only tool-runtime validation, using an existing pinned image."""

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


def require(condition,message):
    if not condition:raise ValueError(message)


def owned(path):
    require(path.resolve(strict=True)==path and path.stat().st_uid==os.getuid(),'Unowned/noncanonical CPU validation path')


def write_new(path,raw):
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())


def save(path,value):write_new(path,(json.dumps(value,sort_keys=True,indent=2)+'\n').encode())


def run(payload):
    approval,manifest,operation=payload['approval'],payload['manifest'],payload['operation']
    identity=approval['approved_remote_identity'];base=Path(approval['approved_workspace']);owned(base)
    require((os.getuid(),os.getgid(),socket.gethostname())==(identity['uid'],identity['gid'],identity['hostname']),
            'CPU validation host identity changed')
    require(manifest['id'] in ('nextgen-native-tools-cpu-20261028a','nextgen-native-tools-cpu-20261028b')
            and manifest.get('GPU_requested') is False and manifest.get('max_seconds')==600
            and manifest.get('image_id')==approval['approved_image_id']
            and operation in ('inspect','prepare','start','status'),'Unknown CPU-only validation request')
    root=base/'agent_work/scratch'/manifest['id'];name='kvra-cpu-'+manifest['id'];label='kvra.native_tools_cpu_id'
    if operation=='status':
        result={'root_exists':root.exists(),'files':{},'remote_writes':False,'GPU_requested':False}
        if not root.exists():return result
        owned(root)
        for filename in ('manifest.json','PREPARED.json','START_INTENT.json','CPU_ADMISSION.json','CPU_acceptance.json','CPU_ERROR.json'):
            path=root/filename
            if path.exists():
                owned(path);raw=path.read_bytes()
                require(len(raw)<=2**20,'Excessive CPU status metadata')
                result['files'][filename]={'sha256':hashlib.sha256(raw).hexdigest(),'value':json.loads(raw)}
        query=subprocess.run(['docker','inspect',name],text=True,capture_output=True,timeout=15)
        if not query.returncode:
            value=json.loads(query.stdout)[0]
            require(value['Config']['Labels'].get(label)==manifest['id'],'Unknown CPU validation container')
            result['container_id']=value['Id']
            result['container']={key:value['State'][key] for key in ('Status','Pid','Running','ExitCode','OOMKilled')}
            result['host_config']={key:value['HostConfig'].get(key) for key in
                                   ('Runtime','NetworkMode','IpcMode','DeviceRequests','Devices','Privileged','ReadonlyRootfs','NanoCpus','Memory')}
            result['logs']=subprocess.check_output(['docker','logs','--tail','24',value['Id']],
                                                   stderr=subprocess.STDOUT,text=True,timeout=15)[-5000:]
        return result
    require(approval['enabled'] and approval['allow_remote_user_workspace_writes'],'No project-write authority')
    expiry=datetime.datetime.fromisoformat(approval['valid_until_utc'].replace('Z','+00:00'))
    require(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=900)<expiry,'CPU validation exceeds authority window')
    def ticks():return list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    first=ticks();time.sleep(2);last=ticks();delta=[a-b for a,b in zip(last,first)];idle=delta[3]/sum(delta)
    available=int(next(row.split()[1] for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith('MemAvailable:')))*1024
    require(idle>=.25 and available>=256*2**30,'Shared CPU/RAM headroom insufficient')
    disk=os.statvfs(base)
    require(disk.f_bavail*disk.f_frsize>=2**30,'Project disk headroom insufficient for bounded CPU artifacts')
    image=json.loads(subprocess.check_output(['docker','image','inspect',approval['approved_image_id']],text=True,timeout=15))[0]
    require(image['Id']==approval['approved_image_id'],'Pinned existing image missing; no pull/install')
    runtimes=json.loads(subprocess.check_output(['docker','info','--format','{{json .Runtimes}}'],text=True,timeout=15))
    require('runc' in runtimes,'GPU-free runc runtime unavailable; do not modify Docker')
    if operation=='inspect':
        require(not root.exists(),'Existing CPU validation must be reconciled')
        return {'can_prepare':True,'remote_writes':False,'GPU_requested':False,'CPU_idle_fraction':idle,
                'memory_available_bytes':available,'runc_available':True}
    if operation=='prepare':
        require(not root.exists(),'CPU prepare is one-time only')
        owned(base/'agent_work');owned(root.parent);root.mkdir(mode=0o700)
        save(root/'manifest.json',manifest);total=0
        require(set(payload['assets'])==set(manifest['asset_sha256']),'Missing frozen CPU asset')
        for relative,asset in payload['assets'].items():
            path=Path(relative)
            require(not path.is_absolute() and '..' not in path.parts and str(path)==relative
                    and (relative=='driver.py' or path.parts[0] in ('pkg','input')),'Unexpected CPU asset path')
            data=base64.b64decode(asset['base64'],validate=True);total+=len(data)
            require(len(data)<=4*2**20 and total<=16*2**20
                    and hashlib.sha256(data).hexdigest()==manifest['asset_sha256'][relative],'CPU asset size/SHA differs')
            target=root/path;target.parent.mkdir(mode=0o700,parents=True,exist_ok=True);owned(target.parent)
            write_new(target,data)
        model=Path(approval['read_only_model_path'])
        require(model.resolve(strict=True)==model and model.is_dir() and model.name=='Qwen3.8-27B','Unexpected tokenizer model path')
        command=['docker','create','--name',name,'--label',label+'='+manifest['id'],'--runtime','runc',
                 '--user',str(os.getuid())+':'+str(os.getgid()),'--network','none','--ipc','private','--shm-size','128m',
                 '--cpus','2','--memory','8g','--memory-swap','8g','--pids-limit','512','--cap-drop','ALL',
                 '--security-opt','no-new-privileges','--security-opt','seccomp=unconfined','--read-only',
                 '--tmpfs','/tmp:rw,nosuid,nodev,size=256m,mode=1777',
                 '--mount','type=bind,src='+str(root)+',dst=/cpu_work',
                 '--mount','type=bind,src='+str(model)+',dst=/models/Qwen3.8-27B,readonly',
                 '-e','HOME=/tmp','-e','NVIDIA_VISIBLE_DEVICES=void','-e','CUDA_VISIBLE_DEVICES=',
                 '-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1',
                 '-e','OPENBLAS_NUM_THREADS=1','-e','TOKENIZERS_PARALLELISM=false','-e','RAYON_NUM_THREADS=2',
                 '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','PYTHONPATH=/cpu_work/pkg',
                 '--entrypoint','python3',approval['approved_image_id'],'-B','/cpu_work/driver.py','/cpu_work']
        save(root/'docker_command.json',{'argv':command})
        identifier=subprocess.check_output(command,text=True,timeout=30).strip()
        save(root/'PREPARED.json',{'container_id':identifier,'GPU_requested':False})
        return {'status':'PREPARED_NATIVE_TOOLS_CPU_NO_GPU','container_id':identifier}
    owned(root);require(json.loads((root/'manifest.json').read_text())==manifest,'Original CPU manifest differs')
    prepared=json.loads((root/'PREPARED.json').read_text())
    value=json.loads(subprocess.check_output(['docker','inspect',prepared['container_id']],text=True,timeout=15))[0]
    require(value['State']['Status']=='created' and value['Config']['Labels'].get(label)==manifest['id']
            and value['HostConfig']['Runtime']=='runc' and value['HostConfig']['NetworkMode']=='none'
            and not value['HostConfig'].get('DeviceRequests') and not value['HostConfig'].get('Devices')
            and not value['HostConfig']['Privileged'] and value['HostConfig']['ReadonlyRootfs']
            and value['HostConfig']['NanoCpus']==2_000_000_000 and value['HostConfig']['Memory']==8*2**30,
            'CPU container already started or has unexpected resource access')
    save(root/'START_INTENT.json',{'operation':'start_exact_native_tools_CPU_once','GPU_requested':False})
    subprocess.check_output(['docker','start',value['Id']],text=True,timeout=30)
    return {'status':'STARTED_NATIVE_TOOLS_CPU_NO_GPU','container_id':value['Id'],'CPU':2,'memory_GiB':8}


if __name__=='__main__':
    try:result={'ok':True,**run(json.load(sys.stdin))}
    except (OSError,ValueError,RuntimeError,KeyError,subprocess.SubprocessError) as error:
        result={'ok':False,'error':type(error).__name__+': '+str(error)[-2500:]}
    print(json.dumps(result))
