"""Freeze existing32 native-tool tasks after actual same-version1720 scoring."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.controller.durable import digest
from scripts.controller.policy import validate_request
from scripts.controller.upstream_tools import KIND, TOOL_GPU, QUALITY_GPU, QUALITY_GPU_AFFINITY
from scripts.upstream_package_runtime import freeze_runtime, attach_remote_asset_reuse

ROOT = Path(__file__).resolve().parents[1]


def prepare(quality_package: Path, quality_scores: Path, run: str, reuse_assets_from_run: str | None,
            *, use_qualified_quality_gpu: bool = False):
    parent = json.loads(quality_package.read_bytes())
    request = dict(parent['request'])
    quality = json.loads(quality_scores.read_bytes())
    if (quality['run_id'] != request['run_id'] or quality.get('same_version_upstream_A_C') is not True
            or quality.get('prediction_count') != 1720
            or quality.get('all_raw_API_and_usage_contracts_independently_verified') is not True
            or any(quality.get(k) != request.get(k) for k in
                ('model_id', 'binary_sha256', 'server_source_commit', 'python_source_commit',
                 'service', 'operators', 'native_configuration'))):
        raise ValueError('Require actual independently scored same-version full1720, not historical quality')
    target_gpu = TOOL_GPU
    if use_qualified_quality_gpu:
        if request.get('gpu_uuids') != [QUALITY_GPU]:
            raise ValueError('Relocated tools must use the actual completed core-quality GPU')
        target_gpu = QUALITY_GPU
        request['tool_gpu_affinity'] = QUALITY_GPU_AFFINITY
    previous = json.loads((ROOT/'plans/experiments/nextgen-bf16-qwen38-mixed8192-tools32-r0-20261030zc/package.json').read_bytes())
    plan = ROOT/'plans/experiments'/run
    plan.mkdir()
    files = {}

    def add(name, path):
        target = plan/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        files[name] = target

    for name in ('input/validation.json', 'input/gdn_dependency_CPU_import.json', 'input/source.bundle'):
        add(name, parent['assets'][name]['local_path'])
    for name, asset in previous['assets'].items():
        if name.startswith('input/tool') and name != 'input/tool_quality_scores.json':
            add(name, asset['local_path'])
    add('input/tool_quality_scores.json', quality_scores)
    config = dict(previous['request']['tool_evaluation'],
                  quality_scores_sha256=hashlib.sha256(quality_scores.read_bytes()).hexdigest())
    run_root = request['approved_workspace']+'/agent_work/runs/'+run
    for field in ('quality_evaluation', 'range_observation', 'model_proof_reuse', 'graph_observer', 'packaging_recovery'):
        request.pop(field, None)
    request.update(run_id=run, remote_output=run_root, python_source_repo=run_root+'/source_repo', gpu_uuids=[target_gpu],
        dataset_path=run_root+'/input/validation.json', replicate=0, max_wall_seconds=5400, max_output_gib=4,
        research_reason='Original32 native-tool conversations/112user turns, same-version quality and normal upstream serving',
        workload={'kind': KIND, 'scope': 'same_version_original32_112turns_native_tools'}, tool_evaluation=config,
        experiment_key=digest({'kind': KIND, 'source': request['server_source_commit'],
            'configuration': request.get('native_configuration'), 'GPU': [target_gpu], 'tool_evaluation': config}))
    files.update(freeze_runtime(plan))
    approval = json.loads((Path.home()/'.config/kv-research/approval.json').read_bytes())
    validate_request(request, approval)
    assets = {name: {'local_path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
              for name, path in files.items()}
    package = {'schema': 'kv-frozen-local-package-v1', 'classification': 'proposal',
               'request': request, 'request_sha256': digest(request), 'assets': assets}
    attach_remote_asset_reuse(package, reuse_assets_from_run)
    (plan/'request.json').open('x').write(json.dumps(request, indent=2)+'\n')
    (plan/'package.json').open('x').write(json.dumps(package, indent=2)+'\n')
    print(json.dumps({'package': str(plan/'package.json'), 'run_id': run, 'dispatched': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality-package', type=Path, required=True)
    parser.add_argument('--quality-scores', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--reuse-assets-from-run')
    parser.add_argument('--use-qualified-quality-gpu', action='store_true',
                        help='Explicit repaired MXFP4 semantic-quality relocation; no tool performance comparison')
    args = parser.parse_args()
    prepare(args.quality_package, args.quality_scores, args.run_id, args.reuse_assets_from_run,
            use_qualified_quality_gpu=args.use_qualified_quality_gpu)
