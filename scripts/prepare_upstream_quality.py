"""Reuse frozen1720 quality inputs after same-version model/Graph qualification."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.controller.durable import digest
from scripts.controller.policy import validate_request
from scripts.controller.upstream_quality import KIND, LIMITS
from scripts.upstream_package_runtime import freeze_runtime

ROOT = Path(__file__).resolve().parents[1]


def prepare(package_path: Path, status_path: Path, run: str, reuse_assets_from_run: str | None = None):
    baseline = json.loads(package_path.read_bytes())
    request = dict(baseline['request'])
    terminal = json.loads(status_path.read_bytes())['files']['result.json']
    result = terminal['value']
    if (result['status'] != 'COMPLETE' or result['gpu_release_verified'] is not True
            or result['live_owned_containers'] or result['run_id'] != request['run_id']
            or result['request_sha256'] != digest(request)):
        raise ValueError('Require actual completed released same-version model/Graph qualification')
    reference = {'run_id': request['run_id'], 'result_sha256': terminal['sha256']}
    plan = ROOT/'plans/experiments'/run
    plan.mkdir()
    files = {}
    for name in ('input/validation.json', 'input/gdn_dependency_CPU_import.json', 'input/source.bundle'):
        target = plan/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(baseline['assets'][name]['local_path'], target)
        files[name] = target
    cohort = ROOT/'results/prepared/nextgen-quality-real-text-native-json-v2-20261028a'
    for target_name, source_name in [('quality_requests.jsonl', 'requests.jsonl'),
                                      ('quality_manifest.json', 'manifest.json'),
                                      ('scoring_protocol.json', 'scoring_protocol.json')]:
        target = plan/'input'/target_name
        shutil.copyfile(cohort/source_name, target)
        files['input/'+target_name] = target
    for field in ('packaging_recovery', 'model_proof_reuse', 'graph_observer'):
        request.pop(field, None)
    run_root = request['approved_workspace']+'/agent_work/runs/'+run
    evaluation = dict(LIMITS, requests_sha256=hashlib.sha256((cohort/'requests.jsonl').read_bytes()).hexdigest(),
                       source_manifest_sha256=hashlib.sha256((cohort/'manifest.json').read_bytes()).hexdigest())
    request.update(run_id=run, remote_output=run_root, python_source_repo=run_root+'/source_repo',
        dataset_path=run_root+'/input/validation.json', replicate=0, max_wall_seconds=7200, max_output_gib=5,
        qualification_reference=reference, quality_evaluation=evaluation, range_observation=False,
        research_reason='Same-version untruncated GSM8K1319 plus LongBench401, reuse fixed raw inputs and independent official scorers',
        workload={'kind': KIND, 'scope': 'same_version_GSM8K1319_LongBench401_full_text'},
        experiment_key=digest({'kind': KIND, 'source': request['server_source_commit'],
            'configuration': request.get('native_configuration'), 'GPU': request['gpu_uuids'], 'evaluation': evaluation}))
    files.update(freeze_runtime(plan))
    approval = json.loads((Path.home()/'.config/kv-research/approval.json').read_bytes())
    validate_request(request, approval)
    assets = {name: {'local_path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
              for name, path in files.items()}
    package = {'schema': 'kv-frozen-local-package-v1', 'classification': 'proposal',
               'request': request, 'request_sha256': digest(request), 'assets': assets}
    from scripts.upstream_package_runtime import attach_remote_asset_reuse
    attach_remote_asset_reuse(package, reuse_assets_from_run)
    (plan/'request.json').write_text(json.dumps(request, indent=2)+'\n')
    (plan/'package.json').write_text(json.dumps(package, indent=2)+'\n')
    print(json.dumps({'run_id': run, 'package': str(plan/'package.json'), 'requests': 1720, 'dispatched': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification-package', type=Path, required=True)
    parser.add_argument('--qualification-status', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--reuse-assets-from-run')
    args = parser.parse_args()
    prepare(args.qualification_package, args.qualification_status, args.run_id, args.reuse_assets_from_run)
