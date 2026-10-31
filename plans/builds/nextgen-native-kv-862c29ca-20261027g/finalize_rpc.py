"""Seal an actually compiled/tested binary after an artifact-path driver failure.

The original unsuccessful build result is preserved. One new isolated release
directory and GPU-invisible metadata-verification container are created once.
"""

from __future__ import annotations

import hashlib
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import sys


VERIFY_PROGRAM = '''import json,os,subprocess,sys
from pathlib import Path
nodes=list(Path('/dev').glob('nvidia[0-9]*'))
if nodes:raise ValueError('CPU artifact verification must have no GPU devices')
binary=sys.argv[1]
result={'GPU_device_nodes':[],'uid':os.getuid()}
for name,command in [('ldd',['ldd',binary]),('help',[binary,'--help'])]:
 p=subprocess.run(command,capture_output=True,text=True,timeout=45)
 result[name]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps(result))
'''


def _require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def _owned(path: Path) -> None:
    _require(path.resolve(strict=True) == path and path.stat().st_uid == os.getuid(), 'Noncanonical/foreign path')


def _sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8*2**20), b''):
            hasher.update(data)
    return hasher.hexdigest()


def _new(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def _json(path: Path, value: dict) -> None:
    _new(path, (json.dumps(value, sort_keys=True, indent=2)+'\n').encode())


def run(payload: dict) -> dict:
    approval, manifest, operation = payload['approval'], payload['manifest'], payload['operation']
    expected = approval['approved_remote_identity']
    _require((os.getuid(), os.getgid(), socket.gethostname()) ==
             (expected['uid'], expected['gid'], expected['hostname']), 'Remote identity changed')
    base = Path(approval['approved_workspace']); _owned(base)
    _require(operation in ('inspect', 'finalize', 'status'), 'Unknown finalization operation')
    _require(manifest['source_commit'] == '862c29cade05e47691147b8f50c979a812495470'
             and manifest['build_id'] == 'nextgen-native-kv-862c29ca-20261027g', 'Unpinned release identity')
    root = base/'agent_work/builds'/manifest['build_id']
    if operation == 'status':
        _owned(root)
        result = {'root': str(root), 'remote_writes': False, 'files': {}}
        for name in ('manifest.json', 'FINALIZE_INTENT.json', 'PREPARED.json', 'verification.json', 'release_result.json'):
            path = root/name
            if path.exists():
                _owned(path); result['files'][name] = {'sha256': _sha(path), 'value': json.loads(path.read_text())}
        return result
    if operation == 'finalize':
        expiry = datetime.datetime.fromisoformat(approval['valid_until_utc'].replace('Z','+00:00'))
        _require(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=300)<expiry,
                 'Artifact finalization would exceed workspace authority expiry')
    parent = base/'agent_work/builds/nextgen-native-kv-862c29ca-20261027f'
    source = parent/'source_repo'
    binary = base/'agent_work/builds/nextgen-native-kv-59fc32ac-20261027d/build/xllm/xllm'
    for path in (parent, source, binary, parent/'build_result.json'):
        _owned(path)
    _require(stat.S_ISREG(binary.stat().st_mode), 'Compiled binary is not a regular file')
    _require(_sha(parent/'build_result.json') == manifest['failed_driver_result_sha256'], 'Original failed result changed')
    failure = json.loads((parent/'build_result.json').read_text())
    _require(failure['success'] is False and failure['GPU_job'] is False
             and failure['source_commit'] == manifest['source_commit']
             and "FileNotFoundError" in failure['error'] and '/build/bin/xllm' in failure['error'],
             'This finalizer only fixes the documented post-test artifact path failure')
    container = json.loads(subprocess.check_output(['docker','inspect',manifest['parent_container_id']], text=True, timeout=15))[0]
    _require(container['Config']['Labels'].get('kvra.cpu_build_id') == parent.name
             and container['State']['Status'] == 'exited' and container['State']['Pid'] == 0
             and not container['State']['OOMKilled'], 'Parent build is not safely exited')
    _require(_sha(binary) == manifest['binary_sha256'] and binary.stat().st_size == manifest['binary_bytes'],
             'Actual compiled binary differs from read-only evidence')
    actual = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True, timeout=15).strip()
    clean = subprocess.check_output(['git','-C',str(source),'status','--porcelain'], text=True, timeout=30)
    _require(actual == manifest['source_commit'] and not clean.strip(), 'Compiled source snapshot changed')
    tests = {}
    for target, expected_count in [('kv_cache_estimation_test',26), ('cuda_kv_cache_admission_test',8), ('kv_cache_test',2)]:
        path = parent/(target+'.log'); _owned(path)
        _require(_sha(path) == manifest['test_log_sha256'][target], 'Actual test log changed')
        text = path.read_text()
        running = re.findall(r'\[==========\] Running (\d+) tests?', text)
        passed = re.findall(r'\[\s*PASSED\s*\]\s*(\d+) tests?', text)
        _require(running == passed == [str(expected_count)] and '[  FAILED  ]' not in text
                 and '[  SKIPPED ]' not in text, 'C++ tests did not all execute/pass')
        tests[target] = expected_count
    _require(not root.exists(), 'Release already exists; reconcile read-only, never repeat finalization')
    _require(os.statvfs(base).f_bavail*os.statvfs(base).f_frsize >= 10*2**30, 'Insufficient project disk headroom')
    image = json.loads(subprocess.check_output(['docker','image','inspect',approval['approved_image_id']], text=True, timeout=15))[0]
    if operation == 'inspect':
        return {'read_only': True, 'remote_writes': False, 'can_finalize': True,
                'binary_sha256': manifest['binary_sha256'], 'CXX_tests_passed': tests, 'new_GPU': False}
    _require(approval['enabled'] and approval['allow_remote_user_workspace_writes'], 'No workspace write authority')
    _owned(root.parent); root.mkdir(mode=0o700)
    _json(root/'manifest.json', manifest)
    _json(root/'FINALIZE_INTENT.json', {'operation': 'seal_existing_compiled_tested_binary_once', 'GPU_requested': False})
    artifacts = root/'artifacts'; artifacts.mkdir(mode=0o700)
    frozen = artifacts/'xllm'; shutil.copy2(binary, frozen)
    _require(_sha(frozen) == manifest['binary_sha256'], 'Frozen binary differs from compiled source artifact')
    adjacent = {}
    for library in binary.parent.glob('*.so*'):
        resolved = library.resolve(strict=True)
        _require(resolved.is_relative_to(base/'agent_work/builds') and resolved.stat().st_uid == os.getuid()
                 and resolved.stat().st_size < 256*2**20 and len(adjacent)<32, 'Unexpected adjacent shared library')
        target = artifacts/library.name
        shutil.copy2(resolved, target)
        adjacent[library.name] = {'bytes': target.stat().st_size, 'sha256': _sha(target)}
    _new(root/'verify.py', VERIFY_PROGRAM.encode())
    inner = '/workspace/agent_work/builds/'+root.name
    image_environment = dict(entry.split('=',1) for entry in image['Config'].get('Env',[]) if '=' in entry)
    libraries = '/workspace/.venv/lib/python3.12/site-packages/tvm_ffi/lib:/workspace/deps/prefix/lib'
    if image_environment.get('LD_LIBRARY_PATH'):
        libraries += ':'+image_environment['LD_LIBRARY_PATH']
    command = ['docker','create','--name','kvra-cpu-finalize-'+root.name,
               '--label','kvra.cpu_finalize_id='+root.name,'--runtime','nvidia','--user',str(os.getuid())+':'+str(os.getgid()),
               '--network','none','--ipc','private','--shm-size','128m','--cpus','1','--memory','8g','--memory-swap','8g',
               '--pids-limit','256','--cap-drop','ALL','--security-opt','no-new-privileges','--read-only',
               '--tmpfs','/tmp:rw,nosuid,nodev,size=256m,mode=1777',
               '--mount','type=bind,src='+str(base)+',dst=/workspace,readonly',
               '--mount','type=bind,src='+str(base)+',dst='+str(base)+',readonly',
               '-e','HOME=/tmp','-e','NVIDIA_VISIBLE_DEVICES=none','-e','CUDA_VISIBLE_DEVICES=',
               '-e','PYTHONDONTWRITEBYTECODE=1','-e','LD_LIBRARY_PATH='+libraries,
               '--entrypoint','/workspace/.venv/bin/python',approval['approved_image_id'],
               '-B',inner+'/verify.py',inner+'/artifacts/xllm']
    _json(root/'docker_command.json', {'argv': command})
    identifier = subprocess.check_output(command, text=True, timeout=30).strip()
    _json(root/'PREPARED.json', {'container_id': identifier, 'GPU_requested': False})
    process = subprocess.run(['docker','start','-a',identifier], capture_output=True, text=True, timeout=110)
    _new(root/'verification_container.log', (process.stdout+process.stderr).encode())
    final = json.loads(subprocess.check_output(['docker','inspect',identifier], text=True, timeout=15))[0]
    _require(final['State']['Status']=='exited' and final['State']['ExitCode']==0
             and not final['State']['OOMKilled'] and not final['HostConfig'].get('Devices')
             and not final['HostConfig'].get('DeviceRequests'), 'CPU metadata verification did not exit cleanly')
    verification = json.loads(process.stdout)
    _json(root/'verification.json', verification)
    _require(verification['GPU_device_nodes']==[] and verification['uid']==os.getuid()
             and verification['ldd']['returncode']==0
             and 'not found' not in verification['ldd']['stdout']+verification['ldd']['stderr'],
             'New binary has unresolved runtime libraries')
    help_text = verification['help']['stdout']+verification['help']['stderr']
    _require(verification['help']['returncode'] in (0,1)
             and all(name in help_text for name in ('kv_cache_backend','kv_cache_k_scale','kv_cache_v_scale',
                                                    'kv_cache_prefill_backend','kv_cache_decode_backend')),
             'Compiled binary does not expose the typed native KV service options')
    result = {'state':'BINARY_CPU_VERIFIED_FROZEN_NOT_MODEL_OR_GPU_VALIDATED','source_commit':actual,
              'binary':str(frozen),'binary_sha256':_sha(frozen),'binary_bytes':frozen.stat().st_size,
              'source_repo':str(source),'CXX_tests_passed':tests,'ldd_no_missing_libraries':True,
              'typed_KV_flags_in_actual_binary_help':True,'adjacent_libraries':adjacent,
              'verification_container_id':identifier,'GPU_requested':False,'original_F_failure_preserved':True}
    _json(root/'release_result.json', result)
    return result


if __name__ == '__main__':
    try:
        response = {'ok':True, **run(json.load(sys.stdin))}
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
        response = {'ok':False,'error':type(error).__name__+': '+str(error)[-4000:]}
    sys.stdout.write(json.dumps(response)+'\n')
