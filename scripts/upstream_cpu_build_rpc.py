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
import time


def _require(condition: bool, message: str) -> None:
    if not condition:raise ValueError(message)


def _new(path: Path, data: bytes) -> None:
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())


def _owned(path: Path) -> None:
    _require(path.resolve(strict=True)==path and path.stat().st_uid==os.getuid(),'Noncanonical/foreign project path')


def _incremental_parent(base: Path, manifest: dict) -> Path | None:
    """Only reuse our exited, unsuccessful build, preserving its source/results."""
    specification=manifest.get('incremental_reuse')
    if specification is None:return None
    _require(isinstance(specification,dict),'Invalid incremental build specification')
    previous_id=specification['build_id']
    _require(re.fullmatch(r'upstream-ac-[0-9a-f]{8}-202611(?:02|05)[a-z]',previous_id) is not None
             and previous_id!=manifest['build_id'],'Invalid incremental parent ID')
    previous=base/'agent_work/builds'/previous_id
    for path in (previous,previous/'source_repo',previous/'build'):_owned(path)
    for name,expected in specification['immutable_files_sha256'].items():
        _require(name in ('manifest.json','build_result.json','CPU_ONLY_ADMISSION.json'),
                 'Unexpected incremental parent evidence')
        path=previous/name;_owned(path)
        _require(hashlib.sha256(path.read_bytes()).hexdigest()==expected,'Parent failure evidence changed')
    _require(set(specification['immutable_files_sha256'])==
             {'manifest.json','build_result.json','CPU_ONLY_ADMISSION.json'},'Missing parent evidence')
    old=json.loads((previous/'manifest.json').read_text())
    failed=json.loads((previous/'build_result.json').read_text())
    admitted=json.loads((previous/'CPU_ONLY_ADMISSION.json').read_text())
    _require(old['source_commit']==specification['source_commit']==failed['source_commit']
             and failed['success'] is False and failed['GPU_job'] is False
             and admitted['GPU_device_nodes']==[],'Parent is not an unsuccessful CPU-only build')
    value=json.loads(subprocess.check_output(['docker','inspect',specification['container_id']],
                                            text=True,timeout=15))[0]
    _require(value['Config']['Labels'].get('kvra.cpu_build_id')==previous_id
             and value['State']['Status']=='exited' and value['State']['Pid']==0
             and not value['State']['Running'] and not value['State']['OOMKilled'],
             'Incremental parent is not safely exited')
    source=previous/'source_repo'
    actual=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True,timeout=15).strip()
    clean=subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=normal'],
                                  text=True,timeout=30)
    _require(actual==specification['source_commit'] and not clean.strip(),'Parent source changed')
    _require(not any((previous/relative).exists() for relative in ('build/bin/xllm','build/xllm/xllm')),
             'Do not overwrite a previously linked service binary')
    return previous


def _shared_headroom() -> dict:
    """Use measured CPU availability; load average also includes blocked tasks."""
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith('MemAvailable:')))*1024
    def ticks() -> list[int]:
        return list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    first=ticks();time.sleep(2);second=ticks();delta=[b-a for a,b in zip(first,second)];total=sum(delta)
    _require(total>0 and min(delta)>=0,'CPU sampling is invalid')
    idle=delta[3]/total;iowait=delta[4]/total;cores=os.cpu_count() or 1;load=os.getloadavg()
    pressure={}
    for kind in ('io','memory'):
        rows=Path('/proc/pressure',kind).read_text().splitlines()
        full=next(row for row in rows if row.startswith('full '))
        pressure[kind]=float(dict(field.split('=') for field in full.split()[1:])['avg10'])
    clear=(available>=256*2**30 and idle>=.25 and iowait<=.1
           and load[0]<=cores*1.25 and pressure['io']<=5 and pressure['memory']<=1)
    return {'can_prepare':clear,'memory_available_bytes':available,'host_cores':cores,'loadavg':load,
            'CPU_idle_fraction':idle,'CPU_iowait_fraction':iowait,'pressure_full_avg10_pct':pressure,
            'sample_seconds':2,'required_idle_fraction':.25,'policy':'bounded4CPU64GiB_actual_shared_headroom_v2'}


def _snapshot_parent(base: Path, manifest: dict) -> dict | None:
    """Copy successful A objects into C's new tree; never write A artifacts."""
    specification = manifest.get('build_snapshot')
    if specification is None:
        return None
    _require(manifest.get('incremental_reuse') is None, 'Snapshot cannot mutate a parent build')
    for key in ('build_id', 'object_build_id'):
        _require(re.fullmatch(r'upstream-ac-[0-9a-f]{8}-202611(?:02|05)[a-z]', specification[key]) is not None,
                 'Invalid snapshot parent identity')
    parent = base/'agent_work/builds'/specification['build_id']
    objects = base/'agent_work/builds'/specification['object_build_id']
    for path in (parent, parent/'source_repo', objects/'build'):
        _owned(path)
    old_files = {}
    for name in ('manifest.json', 'build_result.json', 'CPU_ONLY_ADMISSION.json'):
        path = parent/name; _owned(path)
        data = path.read_bytes()
        _require(hashlib.sha256(data).hexdigest() == specification['immutable_files_sha256'][name],
                 'Successful baseline snapshot evidence changed')
        old_files[name] = json.loads(data)
    result = old_files['build_result.json']
    cpu_context_recovery = specification.get('CPU_context_recovery') is True
    if cpu_context_recovery:
        _require(result.get('success') is False and result.get('error') == 'RuntimeError: kv_cache_test exited -6',
                 'Only the preserved GPU-context CPU test failure may enter this qualification recovery')
        _require('Failed to get device properties for device 0' in (parent/'kv_cache_test.log').read_text()
                 and '[117/117] Linking CXX executable xllm/xllm' in (parent/'compile.log').read_text(),
                 'Missing completed-link and GPU-context failure evidence')
    _require((cpu_context_recovery or (result.get('success') is True and result.get('CPU_tests_executed') is True))
             and result.get('GPU_job') is False
             and result['source_commit'] == specification['source_commit']
             and old_files['CPU_ONLY_ADMISSION.json']['GPU_device_nodes'] == [],
             'Snapshot parent is not a successful CPU-only baseline')
    old_manifest = old_files['manifest.json']
    actual_objects = old_manifest.get('object_build_id', old_manifest.get('incremental_reuse', {}).get('build_id'))
    _require(actual_objects == specification['object_build_id'],
             'Snapshot objects differ from the successful baseline build')
    inspected = json.loads(subprocess.check_output(
        ['docker', 'inspect', 'kvra-cpu-'+specification['build_id']], text=True, timeout=15))[0]
    _require(inspected['State']['Status'] == 'exited' and inspected['State']['Pid'] == 0
             and inspected['State']['ExitCode'] == (1 if cpu_context_recovery else 0)
             and inspected['Config']['Labels'].get('kvra.cpu_build_id') == specification['build_id']
             and not inspected['State']['OOMKilled'],
             'Successful baseline build is not terminal')
    binary = objects/'build/xllm/xllm' if cpu_context_recovery else parent/'artifacts/xllm'
    _owned(binary)
    hasher = hashlib.sha256()
    with binary.open('rb') as stream:
        for block in iter(lambda: stream.read(4*2**20), b''):
            hasher.update(block)
    _require(hasher.hexdigest() == specification['binary_sha256']
             and (cpu_context_recovery or hasher.hexdigest() == result['binary_sha256']),
             'Successful baseline artifact changed')
    clean = subprocess.check_output(['git', '-C', str(parent/'source_repo'), 'status', '--porcelain'],
                                    text=True, timeout=30)
    head = subprocess.check_output(['git', '-C', str(parent/'source_repo'), 'rev-parse', 'HEAD'],
                                   text=True, timeout=15).strip()
    _require(not clean.strip() and head == specification['source_commit'], 'Baseline source changed')
    return {'source': parent/'source_repo', 'objects': objects/'build',
            'virtual_build_id': specification.get('virtual_build_id', specification['object_build_id'])}


def run(payload: dict) -> dict:
    approval=payload['approval'];manifest=payload['manifest'];operation=payload['operation']
    expected=approval['approved_remote_identity']
    _require((os.getuid(),os.getgid(),socket.gethostname())==(expected['uid'],expected['gid'],expected['hostname']),
             'Remote identity differs')
    base=Path(approval['approved_workspace']);_owned(base)
    build_id=manifest['build_id']
    legacy_id = re.fullmatch(r'upstream-ac-[0-9a-f]{8}-202611(?:02|05)[a-z]',build_id) is not None
    p1_id = (re.fullmatch(r'upstream-ac-[0-9a-f]{8}-20261109[a-z]',build_id) is not None
             and manifest.get('build_purpose') == 'MXFP4_SM90a_CPU_JIT_P0_P1'
             and isinstance(manifest.get('native_jit_source_manifest'),dict)
             and build_id[12:20] == manifest.get('source_commit','')[:8])
    _require(legacy_id or p1_id,'Invalid build ID or unbound P1 JIT build purpose')
    stable_cache=manifest.get('stable_JIT_cache_mount')
    _require(stable_cache is None or p1_id and stable_cache=='/run_cache',
             'Stable cache alias is only allowed for the isolated P1 JIT build')
    root=base/'agent_work/builds'/build_id
    _require(manifest['host_root']==str(root) and manifest['uid']==os.getuid(),'Build root/UID mismatch')
    _require(manifest['image_id']==approval['approved_image_id'],'Unpinned build image')
    _require(re.fullmatch(r'[0-9a-f]{40}',manifest['source_commit']) is not None,'Invalid source identity')
    _require(manifest.get('target_platform') == 'cuda', 'Only the frozen CUDA build is supported')
    generated=manifest.get('source_generated_write_subdirs',[])
    _require(generated in ([],['third_party/brpc/src/butil']),
             'Only the confirmed private brpc generated-config subtree may be writable')
    _require(not generated or manifest.get('incremental_reuse') is not None
             or manifest.get('build_snapshot') is not None,
             'Generated config overlay is only for an isolated incremental snapshot')
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
    _require(operation in ('inspect','prepare','resume','start'),'Unknown build operation')
    _require(approval.get('enabled') is True and approval.get('allow_remote_user_workspace_writes') is True
             and approval.get('execution_contract')=='kv-controller-v1','No workspace-write authority')
    expiry=datetime.datetime.fromisoformat(approval['valid_until_utc'].replace('Z','+00:00'))
    _require(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=6000)<expiry,
             'CPU build would exceed standing authority expiry')
    headroom=_shared_headroom()
    _require(headroom['can_prepare'],'Shared CPU/RAM/IO headroom is insufficient: '+json.dumps(headroom,sort_keys=True))
    _require(os.statvfs(base).f_bavail*os.statvfs(base).f_frsize>=30*2**30,'Insufficient new-build disk headroom')
    image=json.loads(subprocess.check_output(['docker','image','inspect',manifest['image_id']],text=True,timeout=15))[0]
    _require(image['Id']==manifest['image_id'],'Pinned image changed')
    source=Path(manifest['baseline_source']);_owned(source)
    _require(str(source).startswith(str(base)+'/'),'Source escaped project BASE')
    previous=_incremental_parent(base,manifest) if operation!='start' else None
    snapshot = _snapshot_parent(base, manifest) if operation != 'start' else None
    dependency = manifest.get('zeromq_reuse')
    if dependency:
        dependency_parent = (manifest.get('incremental_reuse', {}).get('build_id')
                             or manifest.get('build_snapshot', {}).get('virtual_build_id')
                             or manifest.get('build_snapshot', {}).get('object_build_id'))
        _require(dependency_parent == dependency['build_id'],
                 'Private ZeroMQ reuse must come from the verified incremental parent')
        prefix = base/'agent_work/builds'/dependency['build_id']/'zeromq-prefix'
        library = prefix/'lib/libzmq.so.5.2.5'
        _owned(prefix); _owned(library)
        _require(hashlib.sha256(library.read_bytes()).hexdigest() == dependency['library_sha256'],
                 'Private ZeroMQ library identity differs')
    if operation=='inspect':
        _require(not root.exists(),'Existing build must be reconciled, never prepared twice')
        return {'read_only':True,'remote_writes':0,'can_prepare':True,'GPU_requested':False,
                'incremental_parent':str(previous) if previous else None,
                **headroom}
    if operation in ('prepare','resume'):
        if operation=='prepare':
            _require(not root.exists(),'Build already exists; reconcile only')
            _owned(base/'agent_work')
            parent=root.parent
            if not parent.exists():parent.mkdir(mode=0o700)
            _owned(parent);root.mkdir(mode=0o700)
            _new(root/'manifest.json',(json.dumps(manifest,sort_keys=True)+'\n').encode())
            for name_,entry in payload['assets'].items():
                _require(name_ in ('source.bundle','build_driver.py','zeromq-4.3.5.tar.gz'),'Unexpected build asset')
                data=base64.b64decode(entry['base64'],validate=True)
                _require(len(data)<=32*2**20 and hashlib.sha256(data).hexdigest()==entry['sha256']
                         ==manifest['assets_sha256'].get(name_),'Build asset identity/size differs')
                _new(root/name_,data)
        else:
            _owned(root);_owned(root/'source_repo')
            _require(previous is not None and json.loads((root/'manifest.json').read_text())==manifest,
                     'Resume requires the exact unsuccessful incremental prepare manifest')
            _require(not any((root/name_).exists() for name_ in
                             ('PREPARED.json','START_INTENT.json','RESUME_SOURCE_INTENT.json')),
                     'Do not repeat a prepared/started/uncertain resume')
            absent=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=15)
            _require(absent.returncode!=0 and 'No such object' in absent.stderr,
                     'A CPU container may already exist; reconcile only')
            for name_,expected_sha in manifest['assets_sha256'].items():
                _owned(root/name_)
                _require(hashlib.sha256((root/name_).read_bytes()).hexdigest()==expected_sha,
                         'Partial prepare asset differs')
        env=dict(os.environ,GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL='/dev/null',GIT_TERMINAL_PROMPT='0',
                 GIT_NO_LAZY_FETCH='1',GIT_ALLOW_PROTOCOL='file')
        common=['git','-c','core.hooksPath=/dev/null','-c','init.templateDir=']
        target=root/'source_repo'
        if operation=='resume':
            current=subprocess.check_output(common+['-C',str(target),'rev-parse','HEAD'],env=env,text=True,timeout=15).strip()
            clean=subprocess.check_output(common+['-C',str(target),'status','--porcelain'],env=env,text=True,timeout=30)
            _require(current==manifest['incremental_reuse']['source_commit'] and not clean.strip(),
                     'Partial prepare source is not the verified clean parent snapshot')
            subprocess.run(common+['-C',str(target),'cat-file','-e',manifest['source_commit']+'^{commit}'],
                           env=env,capture_output=True,check=True,timeout=15)
            _new(root/'RESUME_SOURCE_INTENT.json',(json.dumps({'operation':'finish_verified_partial_prepare',
                 'source_commit':manifest['source_commit'],'refetch':False,'GPU_requested':False})+'\n').encode())
        elif previous or snapshot:
            preserved_source = previous/'source_repo' if previous else snapshot['source']
            subprocess.run(['cp','-a','--reflink=auto',str(preserved_source),str(target)],
                           env=env,capture_output=True,check=True,timeout=120)
            preserved=preserved_source/'CMakeLists.txt'
            _require(preserved.stat().st_mtime_ns==(target/'CMakeLists.txt').stat().st_mtime_ns,
                     'Incremental snapshot did not preserve source mtimes')
            if snapshot:
                subprocess.run(['cp', '-a', '--reflink=auto', str(snapshot['objects']), str(root/'build')],
                               env=env, capture_output=True, check=True, timeout=180)
                _new(root/'ISOLATED_OBJECT_SNAPSHOT.json', (json.dumps({
                    'parent_source': str(preserved_source), 'parent_objects': str(snapshot['objects']),
                    'private_objects': str(root/'build'), 'baseline_write_mounts': False})+'\n').encode())
        else:
            subprocess.run(common+['clone','--shared','--no-checkout',str(source),str(target)],env=env,capture_output=True,check=True,timeout=90)
        if operation!='resume':
            subprocess.run(common+['-C',str(target),'fetch','--no-tags','--recurse-submodules=no',
                                   str(root/'source.bundle'),manifest['source_commit']],
                           env=env,capture_output=True,check=True,timeout=60)
        subprocess.run(common+['-C',str(target),'checkout','--detach',manifest['source_commit']],env=env,capture_output=True,check=True,timeout=90)
        modules=subprocess.check_output(['git','-C',str(target),'config','--file','.gitmodules','--get-regexp','path'],env=env,text=True).splitlines()
        _require(len(modules)==14,'Submodule set changed')
        for row in modules:
            relative=row.split(None,1)[1];_require(relative.startswith('third_party/') and '..' not in Path(relative).parts,'Unsafe submodule path')
            original=source/relative;_owned(original)
            commit=subprocess.check_output(['git','-C',str(target),'ls-tree','HEAD',relative],env=env,text=True).split()[2]
            actual=subprocess.check_output(['git','-C',str(original),'rev-parse','HEAD'],env=env,text=True).strip()
            if relative in manifest.get('uninitialized_platform_submodules', []):
                _require(relative in ('third_party/torch_npu_ops', 'third_party/xllm_ops'),
                         'Only unused NPU-only modules may remain uninitialized in this CUDA build')
                continue
            subprocess.run(common+['-C',str(original),'cat-file','-e',commit+'^{commit}'],
                           env=env,capture_output=True,check=True,timeout=15)
            if previous or snapshot:
                copied=subprocess.check_output(['git','-C',str(target/relative),'rev-parse','HEAD'],
                                                env=env,text=True,timeout=15).strip()
                _require(copied==commit,'Incremental submodule differs')
            else:
                subprocess.run(common+['clone','--shared','--no-checkout',str(original),str(target/relative)],env=env,capture_output=True,check=True,timeout=90)
                subprocess.run(common+['-C',str(target/relative),'checkout','--detach',commit],env=env,capture_output=True,check=True,timeout=90)
        inner='/workspace/agent_work/builds/'+build_id
        extra_mounts=[]
        if stable_cache is not None:
            (root/'cache').mkdir()
            _owned(root/'cache')
            extra_mounts+=['--mount','type=bind,src='+str(root/'cache')+',dst=/run_cache']
        if previous or snapshot:
            virtual='/workspace/agent_work/builds/'+(previous.name if previous else snapshot['virtual_build_id'])
            writable_build = previous/'build' if previous else root/'build'
            extra_mounts+=['--mount','type=bind,src='+str(target)+',dst='+virtual+'/source_repo,readonly',
                          '--mount','type=bind,src='+str(writable_build)+',dst='+virtual+'/build',
                          '--mount','type=bind,src='+str(target)+',dst='+inner+'/source_repo,readonly']
            for relative in generated:
                writable=target/relative;_owned(writable)
                extra_mounts+=['--mount','type=bind,src='+str(writable)+',dst='+virtual+'/source_repo/'+relative]
        command=['docker','create','--name',name,'--label','kvra.cpu_build_id='+build_id,
                 '--label','kvra.uid='+str(os.getuid()),'--runtime','nvidia','--user',str(os.getuid())+':'+str(os.getgid()),
                 '--network','none','--ipc','private','--shm-size','8g','--cpus','4','--memory','64g','--memory-swap','64g',
                 '--pids-limit','1024','--cap-drop','ALL','--security-opt','no-new-privileges',
                 '--security-opt','seccomp=unconfined','--read-only','--tmpfs','/tmp:rw,nosuid,nodev,size=8g,mode=1777',
                 '--mount','type=bind,src='+str(base)+',dst=/workspace,readonly',
                 '--mount','type=bind,src='+str(base)+',dst='+str(base)+',readonly',
                 '--mount','type=bind,src='+str(root)+',dst='+inner,
                 *extra_mounts,
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
    except subprocess.CalledProcessError as error:
        stderr=error.stderr.decode(errors='replace') if isinstance(error.stderr,bytes) else (error.stderr or '')
        stdout=error.stdout.decode(errors='replace') if isinstance(error.stdout,bytes) else (error.stdout or '')
        result={'ok':False,'error':'CalledProcessError: '+str(error),
                'stderr_tail':stderr[-4096:],'stdout_tail':stdout[-1024:]}
    except Exception as error:result={'ok':False,'error':type(error).__name__+': '+str(error)}
    sys.stdout.write(json.dumps(result,sort_keys=True)+'\n')
