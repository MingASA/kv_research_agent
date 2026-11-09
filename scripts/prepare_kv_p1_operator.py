"""Freeze one reviewed P0/P1 operator package; this never dispatches work."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from scripts.controller.cli import validate_frozen_policy
from scripts.controller.durable import digest, write_new_json
from scripts.controller.idle_admission import FAST_ADMISSION
from scripts.controller.kv_p1_operator import EXTENSION, KIND, PROTOCOL, SCOPE, TOOLS, source_files
from scripts.controller.policy import validate_request
from scripts.controller.python_overlay_compatibility import make_overlay_proof, validate_overlay_request
from scripts.upstream_package_runtime import freeze_runtime


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args: argparse.Namespace) -> dict:
    if re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,127}', args.run_id) is None:
        raise ValueError('Invalid fresh run identity')
    repo = args.repo.resolve(strict=True)
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Review and commit candidate source before freezing')
    metadata = json.loads(args.build_metadata.read_bytes())
    build = metadata['build_result']
    if (build.get('success') is not True or build.get('GPU_job') is not False
            or metadata.get('container_exited_successfully') is not True
            or metadata.get('source_status') or build['source_commit'] != metadata['source_commit']):
        raise ValueError('Original clean compiled artifact receipt required')
    build_manifest = ROOT/'plans/builds'/metadata['build_id']/'manifest.json'
    manifest = json.loads(build_manifest.read_bytes())
    fixture = json.loads((ROOT/'plans/experiments/upstream-C-operators-qualification-r0-20261102a/package.json').read_bytes())
    request = dict(fixture['request'])
    remote = request['approved_workspace']+'/agent_work/runs/'+args.run_id
    hashes = {name: _sha(repo/name) for name in source_files(args.format)}
    for name, value in hashes.items():
        committed = subprocess.check_output(['git', '-C', str(repo), 'show', head+':'+name])
        if hashlib.sha256(committed).hexdigest() != value:
            raise ValueError('Worktree differs from immutable candidate: '+name)
    request.update(
        run_id=args.run_id, remote_output=remote, python_source_repo=remote+'/source_repo',
        python_source_commit=head, server_source_commit=metadata['source_commit'],
        source_repo=metadata['source_repo'], server_binary=metadata['server_binary'],
        binary_sha256=build['binary_sha256'], backend_source_sha256=hashes,
        compiled_backend_source_sha256={p: s for p, s in hashes.items() if not p.endswith('.py')},
        runtime_library_directory=metadata['runtime_library_directory'],
        runtime_dependency_build_id=metadata['runtime_dependency_build_id'],
        zeromq_library_sha256=metadata['zeromq_library_sha256'],
        GPU_context_test_binary=metadata['GPU_context_test_binary'],
        GPU_context_test_sha256=metadata['GPU_context_test_sha256'],
        dataset_path=remote+'/input/validation.json', arms=[args.format],
        gpu_uuids=[args.gpu_uuid], max_wall_seconds=3600, max_output_gib=2,
        workload={'kind': KIND, 'scope': SCOPE}, p1_extension=dict(EXTENSION),
        p1_protocol=dict(PROTOCOL), p1_phase=args.phase,
        probe_driver_sha256=hashes[TOOLS[args.format]],
        operators={args.format: {'planner': 'frozen_P0_P1', 'decode': 'frozen_P0_P1'}},
        native_configuration=None, server_binary_execution='none',
        resource_admission=dict(FAST_ADMISSION),
        research_reason='Finite source-matched Graph-mask, host planning and fixed-scale native writer qualification',
        experiment_key=digest({'source': head, 'format': args.format, 'phase': args.phase,
                               'protocol': PROTOCOL, 'tool': hashes[TOOLS[args.format]]}),
    )
    binary_identity = {
        'source_commit': metadata['source_commit'], 'binary_sha256': build['binary_sha256'],
        'binary_bytes': build['binary_bytes'], 'image_id': manifest['image_id'],
        'build_metadata_sha256': _sha(args.build_metadata), 'build_manifest_sha256': _sha(build_manifest),
        'private_generated_header_sha256': build['private_generated_header_sha256'],
        'zeromq_library_sha256': metadata['zeromq_library_sha256'],
    }
    request['python_overlay_binary_compatibility'] = make_overlay_proof(
        repo, metadata['source_commit'], head, json.loads(args.review.read_bytes()), binary_identity,
    )
    if args.format == 'mxfp4':
        if args.jit_source_manifest is None or (args.cpu_jit_receipt is None)==(args.jit_receipt is None):
            raise ValueError('MX P0/P1 requires its source manifest and exactly one CPU or qualified JIT cache')
        request['native_jit_source_manifest'] = json.loads(args.jit_source_manifest.read_bytes())
        if args.jit_receipt is not None:
            request['native_jit_artifact'] = json.loads(args.jit_receipt.read_bytes())
            request['next_stage_profile'] = 'kv_p0_p1_20261109'
        else:
            request['cpu_native_jit_artifact'] = json.loads(args.cpu_jit_receipt.read_bytes())
    elif args.jit_source_manifest is not None or args.cpu_jit_receipt is not None or args.jit_receipt is not None:
        raise ValueError('Unused MX artifacts on another format')
    if args.pilot_run_id or args.pilot_result_sha256:
        request['p1_pilot_reference'] = {'run_id': args.pilot_run_id, 'result_sha256': args.pilot_result_sha256}
    approval = json.loads((Path.home()/'.config/kv-research/approval.json').read_bytes())
    validate_overlay_request(request, repo)
    validate_request(request, approval)
    plan = ROOT/'plans/experiments'/args.run_id
    plan.mkdir()
    (plan/'input').mkdir()
    files = {}
    for name in ('validation.json', 'gdn_dependency_CPU_import.json'):
        target = plan/'input'/name
        shutil.copyfile(fixture['assets']['input/'+name]['local_path'], target)
        files['input/'+name] = target
    if _sha(files['input/validation.json']) != request['dataset_sha256']:
        raise ValueError('Frozen prerequisite fixture changed')
    bundle = plan/'input/source.bundle'
    subprocess.run(['git', '-C', str(repo), 'bundle', 'create', str(bundle),
                    'HEAD', '^'+metadata['source_commit']], check=True)
    files['input/source.bundle'] = bundle
    files.update(freeze_runtime(plan))
    package = {'schema': 'kv-frozen-local-package-v1', 'classification': 'proposal',
               'request': request, 'request_sha256': digest(request),
               'assets': {name: {'local_path': str(path), 'sha256': _sha(path)} for name, path in files.items()}}
    path = plan/'package.json'
    write_new_json(path, package)
    write_new_json(plan/'request.json', request)
    validate_frozen_policy(path, request, approval)
    return {'package': str(path), 'source_commit': head, 'phase': args.phase,
            'format': args.format, 'CPU_contracts_passed': True, 'GPU_dispatched': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', choices=('mxfp4', 'nvfp4'), required=True)
    parser.add_argument('--phase', choices=('pilot', 'main'), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--build-metadata', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--jit-source-manifest', type=Path)
    parser.add_argument('--cpu-jit-receipt', type=Path)
    parser.add_argument('--jit-receipt', type=Path)
    parser.add_argument('--pilot-run-id')
    parser.add_argument('--pilot-result-sha256')
    print(json.dumps(prepare(parser.parse_args())))
