"""Read-only artifact identity after one successful upstream CPU build."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def collect(payload: dict) -> dict:
    approval = payload['approval']
    identity = approval['approved_remote_identity']
    if (os.getuid(), os.getgid(), socket.gethostname()) != (identity['uid'], identity['gid'], identity['hostname']):
        raise ValueError('Remote artifact identity changed')
    build_id = payload['build_id']
    if re.fullmatch(r'upstream-ac-[0-9a-f]{8}-202611(?:02|05)[a-z]', build_id) is None:
        raise ValueError('Unknown build identity')
    base = Path(approval['approved_workspace'])
    root = base / 'agent_work/builds' / build_id
    if root.resolve(strict=True) != root or root.stat().st_uid != os.getuid():
        raise ValueError('Unowned/noncanonical build root')
    manifest = json.loads((root / 'manifest.json').read_bytes())
    result = json.loads((root / 'build_result.json').read_bytes())
    if result.get('success') is not True or result.get('CPU_tests_executed') is not True:
        raise ValueError('Do not publish artifact metadata for an unsuccessful build')
    value = json.loads(subprocess.check_output(['docker', 'inspect', 'kvra-cpu-' + build_id], text=True, timeout=20))[0]
    if (value['State']['Status'] != 'exited' or value['State']['ExitCode'] != 0 or value['State']['Pid'] != 0
            or value['State']['OOMKilled'] or value['Config']['Labels'].get('kvra.cpu_build_id') != build_id):
        raise ValueError('CPU build container is not the successful original terminal process')
    source = root / 'source_repo'
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True, timeout=20).strip()
    status = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True, timeout=30)
    if commit != manifest['source_commit'] or commit != result['source_commit'] or status.strip():
        raise ValueError('Successful source is no longer clean and pinned')
    binary_inner = result['binary']
    prefix = '/workspace/'
    if not binary_inner.startswith(prefix):
        raise ValueError('Build artifact escaped the project mount')
    binary = base / binary_inner[len(prefix):]
    if binary.resolve(strict=True) != binary or binary.stat().st_uid != os.getuid() or _hash(binary) != result['binary_sha256']:
        raise ValueError('Frozen successful executable identity differs')
    hashes = {}
    for name in payload['source_files']:
        path = source / name
        if not name.startswith('xllm/') or '..' in Path(name).parts or path.resolve(strict=True) != path:
            raise ValueError('Unexpected compiled source identity request')
        hashes[name] = _hash(path)
    dependency_id = manifest['zeromq_reuse']['build_id']
    libdir = base / 'agent_work/builds' / dependency_id / 'zeromq-prefix/lib'
    lib = libdir / 'libzmq.so.5.2.5'
    if lib.resolve(strict=True) != lib or _hash(lib) != manifest['zeromq_reuse']['library_sha256']:
        raise ValueError('Private ZeroMQ library identity differs')
    immutable = {name: _hash(root / name) for name in ('manifest.json', 'build_result.json', 'CPU_ONLY_ADMISSION.json')}
    gpu_test = root/'build/tests/core/kv_cache/storage/kv_cache_test'
    if not gpu_test.is_file() or gpu_test.resolve(strict=True) != gpu_test:
        raise ValueError('Missing frozen cache test executable for deferred GPU-context coverage')
    return {'ok': True, 'read_only': True, 'remote_writes': 0, 'GPU_requested': False,
            'build_id': build_id, 'source_commit': commit, 'source_status': status,
            'container_exited_successfully': True, 'build_result': result,
            'server_binary': str(binary), 'source_repo': str(source), 'backend_source_sha256': hashes,
            'model_config_sha256': _hash(Path(approval['read_only_model_path']) / 'config.json'),
            'runtime_library_directory': str(libdir), 'runtime_dependency_build_id': dependency_id,
            'zeromq_library_sha256': _hash(lib), 'immutable_files_sha256': immutable,
            'runtime_packages': json.loads((root / 'RUNTIME_PACKAGES.json').read_bytes()),
            'GPU_context_test_binary': str(gpu_test), 'GPU_context_test_sha256': _hash(gpu_test)}


if __name__ == '__main__':
    print(json.dumps(collect(json.load(sys.stdin))))
