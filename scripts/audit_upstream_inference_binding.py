"""Check actual commands and native cache bindings in completed quality/tool runs."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_upstream_run import audit as lifecycle
from scripts.audit_upstream_model import native_runtime
from scripts.controller.upstream_qualification import server_command


def audit(path: Path) -> dict:
    complete = lifecycle(path)
    manifest = json.loads(path.read_bytes())
    root = Path(manifest['mirror_directory'])/'files'
    request = json.loads((root/'ticket.json').read_bytes())['request']
    kind = request['workload']['kind']
    if kind not in ('upstream_AC_frozen_core_quality1720', 'upstream_AC_real_native_tools32'):
        raise ValueError('Require a completed actual upstream core-quality or tool run')
    output = root/'output'/request['arms'][0]
    expected = server_command(request, output)
    if kind == 'upstream_AC_real_native_tools32':
        expected += ['--tool_call_parser=qwen3_coder']
    command_path = output/'server_command.json'
    if json.loads(command_path.read_bytes()) != expected:
        raise ValueError('Recorded actual inference command differs from the frozen normal serving path')
    log_path = output/'server.log'
    bound = native_runtime(log_path.read_text(), request.get('native_configuration'))
    return {'run_id': request['run_id'], 'arm': request['arms'][0], 'kind': kind,
            'source_commit': request['server_source_commit'], 'binary_sha256': request['binary_sha256'],
            'complete_original_SHA_and_lifecycle_verified': complete['all_mirror_SHA_verified'],
            'actual_command_matches_frozen_request': True,
            'actual_native_bindings': bound,
            'source_manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'actual_server_command_sha256': hashlib.sha256(command_path.read_bytes()).hexdigest(),
            'actual_server_log_sha256': hashlib.sha256(log_path.read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sha-manifest', action='append', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [audit(p) for p in args.sha_manifest]
    if len({r['run_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate original inference source')
    result = {'schema': 'upstream-actual-quality-tool-execution-binding-v1', 'runs': rows,
              'remote_access_or_writes': False, 'new_model_requests_or_GPU_work': False,
              'quality_or_performance_acceptance': False}
    args.out.open('x').write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'actual_run_bindings_verified': len(rows), 'out': str(args.out)}))
