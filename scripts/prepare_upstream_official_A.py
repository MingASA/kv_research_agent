"""Freeze one official A qualification package after the actual CPU build passes.

The caller first obtains a read-only artifact receipt. This script neither
connects to H800 nor dispatches GPU work; launch_guarded_once remains required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from scripts.controller.durable import digest
from scripts.controller.upstream_qualification import KIND, MODEL, OFFICIAL_COMMIT, BASELINE_COMMIT, SERVICE, SOURCE_FILES, CAMPAIGN_GPU

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path('/home/mingasa/projects/xllm-official-main-cuda-fix-20261102a')
RUN = 'upstream-A-qwen38-qualification-r0-20261102a'


def prepare(metadata_path: Path, run_id: str = RUN, replicate: int = 0, repair_result: Path | None = None) -> dict:
    metadata = json.loads(metadata_path.read_bytes())
    result = metadata['build_result']
    if (result.get('success') is not True or result.get('CPU_tests_executed') is not True
            or result.get('GPU_job') is not False or result.get('source_commit') != BASELINE_COMMIT
            or metadata.get('source_commit') != BASELINE_COMMIT
            or metadata.get('source_status') != '' or metadata.get('container_exited_successfully') is not True):
        raise ValueError('Official A requires the actual successful, exited, clean-source CPU build')
    if subprocess.check_output(['git', '-C', str(OFFICIAL), 'rev-parse', 'HEAD'], text=True).strip() != BASELINE_COMMIT:
        raise ValueError('Local frozen official worktree changed')
    hashes = {name: hashlib.sha256((OFFICIAL / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    if metadata['backend_source_sha256'] != hashes:
        raise ValueError('Live compiled official source differs from the frozen local source')
    approval = json.loads((Path.home() / '.config/kv-research/approval.json').read_bytes())
    old_config = json.loads((ROOT / 'state/NEXTGEN_CONFIG.local.json').read_bytes())
    old_package = json.loads((ROOT / 'plans/experiments/nextgen-bf16-qwen38-chat128-c16-r2-20261030e/package.json').read_bytes())
    overlay = old_package['request']['runtime_dependency_overlay']
    plan = ROOT / 'plans/experiments' / run_id
    plan.mkdir()
    inputs = plan / 'input'
    inputs.mkdir()
    shutil.copyfile(ROOT / 'plans/upstream-ac-20261102a/input/qualification/validation.json', inputs / 'validation.json')
    proof = Path(old_package['assets']['input/gdn_dependency_CPU_import.json']['local_path'])
    shutil.copyfile(proof, inputs / 'gdn_dependency_CPU_import.json')
    if hashlib.sha256(proof.read_bytes()).hexdigest() != overlay['CPU_import_evidence_sha256']:
        raise ValueError('Existing private GDN dependency proof changed')
    subprocess.run(['git', '-C', str(OFFICIAL), 'bundle', 'create', str(inputs / 'source.bundle'),
                    'HEAD', '^' + OFFICIAL_COMMIT], check=True)
    run_root = approval['approved_workspace'] + '/agent_work/runs/' + run_id
    request = {
        'schema': 'kv-nextgen-run-v1', 'run_id': run_id, 'replicate': replicate,
        'experiment_key': digest({'kind': KIND, 'baseline': BASELINE_COMMIT, 'service': SERVICE, 'model': MODEL, 'GPU': CAMPAIGN_GPU}),
        'research_reason': 'Qualify exact official normal CUDA Python Executor before any A/C service benchmark',
        'ssh_target': approval['ssh_target'], 'approved_host_fingerprint': approval['approved_host_fingerprint'],
        'approved_workspace': approval['approved_workspace'], 'remote_output': run_root,
        'image_id': approval['approved_image_id'], 'gpu_uuids': [CAMPAIGN_GPU],
        'max_wall_seconds': 2700, 'startup_seconds': 1500, 'max_output_gib': 4,
        'arms': ['bf16'], 'model_id': MODEL, 'model_path': approval['read_only_model_path'],
        'model_config_sha256': metadata['model_config_sha256'],
        'upstream_commit': OFFICIAL_COMMIT, 'server_source_commit': BASELINE_COMMIT,
        'python_source_commit': BASELINE_COMMIT, 'source_repo': metadata['source_repo'],
        'python_source_repo': run_root + '/source_repo', 'server_binary': metadata['server_binary'],
        'binary_sha256': result['binary_sha256'], 'backend_source_sha256': hashes,
        'runtime_library_directory': metadata['runtime_library_directory'],
        'runtime_dependency_build_id': metadata['runtime_dependency_build_id'],
        'zeromq_library_sha256': metadata['zeromq_library_sha256'],
        'dataset_path': run_root + '/input/validation.json',
        'dataset_sha256': hashlib.sha256((inputs / 'validation.json').read_bytes()).hexdigest(),
        'workload': {'kind': KIND, 'scope': 'official_model_prefix_graph_qualification'},
        'service': dict(SERVICE), 'graph_mode': 'cudagraphs', 'prefix_cache': True,
        'operators': {'bf16': {'planner': 'official_auto', 'decode': 'official_auto'}},
        'runtime_dependency_overlay': overlay, 'server_binary_execution': 'required',
        'GPU_context_test_binary': metadata['GPU_context_test_binary'],
        'GPU_context_test_sha256': metadata['GPU_context_test_sha256'],
    }
    from scripts.controller.policy import validate_request
    validate_request(request, approval)
    files = {'input/' + p.name: p for p in inputs.iterdir()}
    from scripts.upstream_package_runtime import freeze_runtime
    files.update(freeze_runtime(plan))
    if repair_result is not None:
        previous = json.loads(repair_result.read_bytes())['files']['result.json']
        old = previous['value']
        if (replicate != 1 or old['status'] != 'FAILED' or not old['gpu_release_verified']
                or old['live_owned_containers'] or old['arm_results']):
            raise ValueError('Packaging repair needs the actual released failed original run')
        request['packaging_recovery'] = {'prior_run_id': old['run_id'],
            'prior_result_sha256': previous['sha256'], 'reason': 'standalone_evidence_module_freeze',
            'CPU_import_evidence_sha256': hashlib.sha256((plan/'FROZEN_ENTRY_IMPORT_CPU.json').read_bytes()).hexdigest()}
        shutil.copyfile(plan/'FROZEN_ENTRY_IMPORT_CPU.json', inputs/'packaging_import_CPU.json')
        files['input/packaging_import_CPU.json'] = inputs/'packaging_import_CPU.json'
        validate_request(request, approval)
    elif replicate != 0:
        raise ValueError('A qualification retry must identify its actual predecessor')
    assets = {name: {'local_path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
              for name, path in files.items()}
    package = {'schema': 'kv-frozen-local-package-v1', 'classification': 'proposal', 'request': request,
               'request_sha256': digest(request), 'assets': assets}
    (plan / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
    (plan / 'package.json').write_text(json.dumps(package, indent=2) + '\n')
    # Keep the previous experimental deployment config intact. New CLI selects
    # this dedicated provenance record only for the new official A kind.
    config = {k: old_config[k] for k in ('ssh_target', 'known_host_ed25519_fingerprint', 'remote_base',
                                       'remote_identity', 'remote_model_path')}
    config.update(source_commit=BASELINE_COMMIT, binary_sha256=result['binary_sha256'],
                  remote_new_binary_path=metadata['server_binary'], remote_source_repo=metadata['source_repo'],
                  new_binary_CPU_tests_verified=True, new_binary_model_or_GPU_verified=False,
                  source_metadata_receipt=str(metadata_path.resolve()))
    config_path = ROOT / 'state/UPSTREAM_AC_A.local.json'
    if config_path.exists():
        if json.loads(config_path.read_bytes()) != config:
            raise ValueError('Do not overwrite an existing official deployment identity')
    else:
        with config_path.open('x') as stream:
            json.dump(config, stream, indent=2)
        config_path.chmod(0o600)
    return {'package': str(plan / 'package.json'), 'run_id': run_id, 'classification': 'frozen_proposal_not_dispatched',
            'source_commit': BASELINE_COMMIT, 'binary_sha256': result['binary_sha256']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--run-id', default=RUN)
    parser.add_argument('--replicate', type=int, default=0)
    parser.add_argument('--repair-result', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.metadata, args.run_id, args.replicate, args.repair_result)))


if __name__ == '__main__':
    main()
