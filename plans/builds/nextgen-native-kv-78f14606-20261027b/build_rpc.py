"""Pinned SSH endpoint for one isolated, GPU-invisible CPU build."""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys


def _require(condition: bool, message: str) -> None:
    if not condition:raise ValueError(message)


def _new(path: Path, data: bytes) -> None:
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())


def _owned(path: Path) -> None:
    _require(path.resolve(strict=True)==path and path.stat().st_uid==os.getuid(),'Noncanonical/foreign project path')


def run(payload: dict) -> dict:
    approval=payload['approval'];manifest=payload['manifest'];operation=payload['operation']
    expected=approval['approved_remote_identity']
    _require((os.getuid(),os.getgid(),socket.gethostname())==(expected['uid'],expected['gid'],expected['hostname']),
             'Remote identity differs')
    base=Path(approval['approved_workspace']);_owned(base)
    build_id=manifest['build_id']
    _require(re.fullmatch(r'nextgen-native-kv-[0-9a-f]{8}-20261027[a-z]',build_id) is not None,'Invalid build ID')
    root=base/'agent_work/builds'/build_id
    _require(manifest['host_root']==str(root) and manifest['uid']==os.getuid(),'Build root/UID mismatch')
    _require(manifest['image_id']==approval['approved_image_id'],'Unpinned build image')
    _require(re.fullmatch(r'[0-9a-f]{40}',manifest['source_commit']) is not None,'Invalid source identity')
    name='kvra-cpu-'+build_id
    if operation=='status':
        _owned(root);result={'root':str(root),'files':{}}
        for filename in ('CPU_ONLY_ADMISSION.json','build_result.json','PREPARED.json'):
            path=root/filename
            if path.exists():_owned(path);result['files'][filename]=json.loads(path.read_text())
        inspected=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=15)
        if inspected.returncode==0:
            value=json.loads(inspected.stdout)[0]
            _require(value['Config']['Labels'].get('kvra.cpu_build_id')==build_id,'Unknown CPU container')
            result['container']={k:value['State'][k] for k in ('Status','Running','ExitCode','OOMKilled','Pid')}
        for filename in ('configure.log','compile.log'):
            path=root/filename
            if path.exists():
                _owned(path)
                with path.open('rb') as stream:
                    stream.seek(max(0,path.stat().st_size-5000));result[filename]=stream.read().decode(errors='replace')
        return result
    _require(operation in ('inspect','prepare','start'),'Unknown build operation')
    _require(approval.get('enabled') is True and approval.get('allow_remote_user_workspace_writes') is True
             and approval.get('execution_contract')=='kv-controller-v1','No workspace-write authority')
    expiry=datetime.datetime.fromisoformat(approval['valid_until_utc'].replace('Z','+00:00'))
    _require(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=6000)<expiry,
             'CPU build would exceed standing authority expiry')
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    _require(available>=256*2**30 and os.getloadavg()[0] <= (os.cpu_count() or 1)*0.6,'Shared CPU/RAM is busy')
    _require(os.statvfs(base).f_bavail*os.statvfs(base).f_frsize>=30*2**30,'Insufficient new-build disk headroom')
    image=json.loads(subprocess.check_output(['docker','image','inspect',manifest['image_id']],text=True,timeout=15))[0]
    _require(image['Id']==manifest['image_id'],'Pinned image changed')
    source=Path(manifest['baseline_source']);_owned(source)
    _require(str(source).startswith(str(base)+'/'),'Source escaped project BASE')
    if operation=='inspect':
        _require(not root.exists(),'Existing build must be reconciled, never prepared twice')
        return {'read_only':True,'remote_writes':0,'can_prepare':True,'GPU_requested':False,
                'memory_available_bytes':available,'loadavg':os.getloadavg(),'host_cores':os.cpu_count()}
    if operation=='prepare':
        _require(not root.exists(),'Build already exists; reconcile only')
        _owned(base/'agent_work')
        parent=root.parent
        if not parent.exists():parent.mkdir(mode=0o700)
        _owned(parent);root.mkdir(mode=0o700)
        _new(root/'manifest.json',(json.dumps(manifest,sort_keys=True)+'\n').encode())
        for name_,entry in payload['assets'].items():
            _require(name_ in ('source.bundle','build_driver.py'),'Unexpected build asset')
            data=base64.b64decode(entry['base64'],validate=True)
            _require(len(data)<=32*2**20 and hashlib.sha256(data).hexdigest()==entry['sha256']
                     ==manifest['assets_sha256'].get(name_),'Build asset identity/size differs')
            _new(root/name_,data)
        env=dict(os.environ,GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL='/dev/null',GIT_TERMINAL_PROMPT='0',GIT_NO_LAZY_FETCH='1')
        common=['git','-c','core.hooksPath=/dev/null','-c','init.templateDir=']
        target=root/'source_repo'
        subprocess.run(common+['clone','--shared','--no-checkout',str(source),str(target)],env=env,capture_output=True,check=True,timeout=90)
        subprocess.run(common+['-C',str(target),'fetch','--no-tags',str(root/'source.bundle'),manifest['source_commit']],env=env,capture_output=True,check=True,timeout=60)
        subprocess.run(common+['-C',str(target),'checkout','--detach',manifest['source_commit']],env=env,capture_output=True,check=True,timeout=90)
        modules=subprocess.check_output(['git','-C',str(target),'config','--file','.gitmodules','--get-regexp','path'],env=env,text=True).splitlines()
        _require(len(modules)==14,'Submodule set changed')
        for row in modules:
            relative=row.split(None,1)[1];_require(relative.startswith('third_party/') and '..' not in Path(relative).parts,'Unsafe submodule path')
            original=source/relative;_owned(original)
            commit=subprocess.check_output(['git','-C',str(target),'ls-tree','HEAD',relative],env=env,text=True).split()[2]
            actual=subprocess.check_output(['git','-C',str(original),'rev-parse','HEAD'],env=env,text=True).strip()
            _require(actual==commit,'Existing submodule does not match new source gitlink')
            subprocess.run(common+['clone','--shared','--no-checkout',str(original),str(target/relative)],env=env,capture_output=True,check=True,timeout=90)
            subprocess.run(common+['-C',str(target/relative),'checkout','--detach',commit],env=env,capture_output=True,check=True,timeout=90)
        inner='/workspace/agent_work/builds/'+build_id
        command=['docker','create','--name',name,'--label','kvra.cpu_build_id='+build_id,
                 '--label','kvra.uid='+str(os.getuid()),'--runtime','nvidia','--user',str(os.getuid())+':'+str(os.getgid()),
                 '--network','none','--ipc','private','--shm-size','8g','--cpus','4','--memory','64g','--memory-swap','64g',
                 '--pids-limit','1024','--cap-drop','ALL','--security-opt','no-new-privileges',
                 '--security-opt','seccomp=unconfined','--read-only','--tmpfs','/tmp:rw,nosuid,nodev,size=8g,mode=1777',
                 '--mount','type=bind,src='+str(base)+',dst=/workspace,readonly',
                 '--mount','type=bind,src='+str(base)+',dst='+str(base)+',readonly',
                 '--mount','type=bind,src='+str(root)+',dst='+inner,
                 '-e','HOME=/tmp','-e','NVIDIA_VISIBLE_DEVICES=none','-e','CUDA_VISIBLE_DEVICES=',
                 '-e','PYTHONDONTWRITEBYTECODE=1','--entrypoint','/workspace/.venv/bin/python',manifest['image_id'],
                 '-B',inner+'/build_driver.py','--root',inner]
        _new(root/'docker_command.json',(json.dumps(command)+'\n').encode())
        identifier=subprocess.check_output(command,env=env,text=True,timeout=30).strip()
        _new(root/'PREPARED.json',(json.dumps({'container_id':identifier,'build_id':build_id})+'\n').encode())
        return {'status':'PREPARED_CPU_ONLY','root':str(root),'container_id':identifier}
    _owned(root)
    value=json.loads(subprocess.check_output(['docker','inspect',name],text=True,timeout=15))[0]
    _require(value['State']['Status']=='created' and value['Config']['Labels'].get('kvra.cpu_build_id')==build_id,'Do not restart existing CPU build')
    _require(not value['HostConfig'].get('DeviceRequests') and not value['HostConfig'].get('Devices') and not value['HostConfig']['Privileged'],
             'CPU build must have no GPU device request or privileged device access')
    _new(root/'START_INTENT.json',b'{"operation":"start_original_CPU_container"}\n')
    subprocess.check_output(['docker','start',value['Id']],text=True,timeout=30)
    return {'status':'STARTED_CPU_ONLY','root':str(root),'container_id':value['Id'],'GPU_requested':False}


if __name__=='__main__':
    try:result={'ok':True,**run(json.load(sys.stdin))}
    except Exception as error:result={'ok':False,'error':type(error).__name__+': '+str(error)}
    sys.stdout.write(json.dumps(result,sort_keys=True)+'\n')
