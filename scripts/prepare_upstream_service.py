"""Freeze the three predeclared production cases from a qualified deployment."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.controller.durable import digest
from scripts.controller.policy import validate_request
from scripts.controller.upstream_service import KIND, CASES

ROOT = Path(__file__).resolve().parents[1]


def prepare(package_path: Path, result_path: Path, run_id: str, reuse_assets_from_run: str | None = None):
    baseline = json.loads(package_path.read_bytes())
    request = dict(baseline['request'])
    raw = result_path.read_bytes()
    result = json.loads(raw)
    if (result.get('status') != 'COMPLETE' or result.get('run_id') != request['run_id']
            or result.get('request_sha256') != digest(request)
            or result.get('gpu_release_verified') is not True or result.get('live_owned_containers') != []):
        raise ValueError('Need actual successful released qualification result, not a plan or cached summary')
    reference = ROOT/'plans/experiments/nextgen-bf16-qwen38-mixed8192-prefix32-r1-20261030wc/package.json'
    prior = json.loads(reference.read_bytes())
    plan = ROOT/'plans/experiments'/run_id
    plan.mkdir()
    files = {}

    def add(name, source):
        target = plan/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[name] = target

    for name in ('input/validation.json', 'input/gdn_dependency_CPU_import.json', 'input/source.bundle'):
        add(name, Path(baseline['assets'][name]['local_path']))
    inputs = ROOT/'plans/upstream-ac-20261102a/input'
    hashes = {}
    for dataset, names in [('sharegpt', ['manifest', 'measured512', 'warmup16', 'retokenized_Qwen38_CPU-a1']),
                           ('govreport', ['manifest', 'measured64', 'warmup4'])]:
        for name in names:
            relative = dataset+'/'+name+'.json'
            add('input/production/'+relative, inputs/relative)
            hashes[relative] = hashlib.sha256((inputs/relative).read_bytes()).hexdigest()
    for name in ('input/full_CLI_protocol_CPU_audit.json', 'runtime_support/tokenizer_compat/sitecustomize.py'):
        add(name, Path(prior['assets'][name]['local_path']))
    from scripts.upstream_package_runtime import freeze_runtime
    files.update(freeze_runtime(plan))
    run_root = request['approved_workspace']+'/agent_work/runs/'+run_id
    config = {'input_files_sha256': hashes,
        'qualification_run_id': request['run_id'], 'qualification_result_sha256': hashlib.sha256(raw).hexdigest(),
        'aiperf_cli_config_sha256': prior['request']['performance_evaluation']['aiperf_cli_config_sha256'],
        'full_CLI_protocol_CPU_audit_sha256': prior['assets']['input/full_CLI_protocol_CPU_audit.json']['sha256']}
    request.update(run_id=run_id, remote_output=run_root, python_source_repo=run_root+'/source_repo',
        replicate=0,
        dataset_path=run_root+'/input/validation.json', max_wall_seconds=7200, max_output_gib=5,
        experiment_key=digest({'kind': KIND, 'source': request['server_source_commit'],
            'configuration': request.get('native_configuration'), 'service': request['service'],
            'GPU': request['gpu_uuids'], 'input_files_sha256': hashes}),
        research_reason='Fixed original ShareGPT512+16 C32/C64 and full GovReport64+4 C32, natural EOS',
        workload={'kind': KIND, 'scope': 'fixed512_sharegpt_and64_govreport_AC_v1'},
        benchmark_cases=[c[0] for c in CASES], service_benchmark=config)
    request.pop('packaging_recovery', None)
    request.pop('model_proof_reuse', None)
    request.pop('graph_observer', None)
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
    return {'package': str(plan/'package.json'), 'run_id': run_id, 'dispatched': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification-package', type=Path, required=True)
    parser.add_argument('--qualification-result', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--reuse-assets-from-run')
    args = parser.parse_args()
    print(json.dumps(prepare(args.qualification_package, args.qualification_result, args.run_id, args.reuse_assets_from_run)))
