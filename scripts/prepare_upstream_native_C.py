"""Freeze native C operator/model packages from actual successful C build metadata."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from scripts.controller.durable import digest
from scripts.controller.policy import validate_request
from scripts.controller.upstream_native import OPERATOR_KIND, MODEL_KIND, B_KIND, SOURCE_FILES, CONFIGURATIONS
from scripts.controller.upstream_qualification import BASELINE_COMMIT, OFFICIAL_COMMIT, MODEL, SERVICE

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path('/home/mingasa/projects/xllm-upstream-native-kv-20261102a')


def prepare(metadata_path: Path, mode: str, operator_gpu: str | None = None,
            regression_gpu: str | None = None):
    metadata = json.loads(metadata_path.read_bytes())
    result = metadata['build_result']
    commit = subprocess.check_output(['git', '-C', str(NATIVE), 'rev-parse', 'HEAD'], text=True).strip()
    if (result.get('success') is not True or result.get('CPU_tests_executed') is not True
            or result.get('GPU_job') is not False or metadata['source_commit'] != commit
            or metadata['container_exited_successfully'] is not True or metadata['source_status']):
        raise ValueError('C requires actual successful CPU build and exact source identity')
    if subprocess.check_output(['git', '-C', str(NATIVE), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Native local source must remain clean')
    subprocess.run(['git', '-C', str(NATIVE), 'merge-base', '--is-ancestor', BASELINE_COMMIT, commit], check=True)
    hashes = {name: hashlib.sha256((NATIVE/name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    if hashes != metadata['backend_source_sha256']:
        raise ValueError('C compiled and local source identities differ')
    baseline_plan = ROOT/'plans/experiments/upstream-A-qwen38-qualification-r1-20261102d'
    baseline = json.loads((baseline_plan/'package.json').read_bytes())
    request = dict(baseline['request'])
    run = 'upstream-C-' + mode + '-qualification-r0-20261102a'
    if regression_gpu is not None:
        if mode != 'bf16' or operator_gpu is not None:
            raise ValueError('A second regression GPU is only for untimed common-path BF16 checks')
        run = 'upstream-C-bf16-qualification-r0-20261102b'
        request['gpu_uuids'] = [regression_gpu]
    plan = ROOT/'plans/experiments'/run
    plan.mkdir()
    inputs = plan/'input'
    inputs.mkdir()
    for name in ('validation.json', 'gdn_dependency_CPU_import.json'):
        shutil.copyfile(baseline_plan/'input'/name, inputs/name)
    subprocess.run(['git', '-C', str(NATIVE), 'bundle', 'create', str(inputs/'source.bundle'),
                    'HEAD', '^' + BASELINE_COMMIT], check=True)
    run_root = request['approved_workspace']+'/agent_work/runs/'+run
    operator = mode == 'operators'
    if operator_gpu is not None:
        if not operator:
            raise ValueError('A/C model and service jobs retain the fixed campaign GPU')
        request['gpu_uuids'] = [operator_gpu]
    arm = 'bf16' if mode == 'bf16' else 'fp8_e4m3' if operator or mode == 'fp8' else 'nvfp4'
    request.update(run_id=run, remote_output=run_root, python_source_repo=run_root+'/source_repo',
        dataset_path=run_root+'/input/validation.json',
        experiment_key=digest({'kind': OPERATOR_KIND if operator else B_KIND if mode == 'bf16' else MODEL_KIND, 'mode': mode,
                               'source': commit, 'model': MODEL, 'service': SERVICE,
                               'GPU': request['gpu_uuids'],
                               'native_configuration': None if operator or mode == 'bf16' else CONFIGURATIONS[arm]}),
        research_reason='Bounded native C qualification before comparable production measurement',
        arms=[arm], native_base_commit=BASELINE_COMMIT, upstream_commit=OFFICIAL_COMMIT,
        server_source_commit=commit, python_source_commit=commit, source_repo=metadata['source_repo'],
        server_binary=metadata['server_binary'], binary_sha256=result['binary_sha256'],
        GPU_context_test_binary=metadata['GPU_context_test_binary'],
        GPU_context_test_sha256=metadata['GPU_context_test_sha256'],
        backend_source_sha256=hashes, model_id=MODEL, service=dict(SERVICE),
        server_binary_execution='none' if operator else 'required', max_wall_seconds=3600 if operator else 2700,
        workload={'kind': OPERATOR_KIND if operator else B_KIND if mode == 'bf16' else MODEL_KIND,
                  'scope': 'three_frozen_native_operator_candidates_no_model' if operator
                           else 'minimal_common_path_BF16_prefix_graph_regression' if mode == 'bf16'
                           else 'native_model_prefix_graph_qualification'})
    if not operator and mode != 'bf16':
        request['native_configuration'] = CONFIGURATIONS[arm]
    request['operators'] = {arm: {'planner': 'frozen_three_candidates', 'decode': 'frozen_three_candidates'}
                           if operator else {'planner': 'official_auto', 'decode': 'official_auto'} if mode == 'bf16'
                           else {'planner': CONFIGURATIONS[arm]['prefill'],
                                             'decode': CONFIGURATIONS[arm]['decode']}}
    approval = json.loads((Path.home()/'.config/kv-research/approval.json').read_bytes())
    validate_request(request, approval)
    files = {'input/'+p.name: p for p in inputs.iterdir()}
    from scripts.upstream_package_runtime import freeze_runtime
    files.update(freeze_runtime(plan))
    if not operator:
        from scripts.upstream_package_runtime import attach_graph_observer
        files.update(attach_graph_observer(plan, request))
    request.pop('packaging_recovery', None)
    request['replicate'] = 0
    assets = {name: {'local_path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
              for name, path in files.items()}
    package = {'schema': 'kv-frozen-local-package-v1', 'classification': 'proposal',
               'request': request, 'request_sha256': digest(request), 'assets': assets}
    (plan/'request.json').write_text(json.dumps(request, indent=2)+'\n')
    (plan/'package.json').write_text(json.dumps(package, indent=2)+'\n')
    config = json.loads((ROOT/'state/UPSTREAM_AC_A.local.json').read_bytes())
    config.update(source_commit=commit, binary_sha256=result['binary_sha256'],
                  remote_new_binary_path=metadata['server_binary'], remote_source_repo=metadata['source_repo'],
                  source_metadata_receipt=str(metadata_path.resolve()))
    config_path = ROOT/'state/UPSTREAM_AC_C.local.json'
    if config_path.exists():
        if json.loads(config_path.read_bytes()) != config:
            raise ValueError('Do not replace an existing C deployment config')
    else:
        with config_path.open('x') as f:
            json.dump(config, f, indent=2)
        config_path.chmod(0o600)
    return {'package': str(plan/'package.json'), 'run_id': run, 'dispatched': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--mode', choices=['operators', 'fp8', 'nvfp4', 'bf16'], required=True)
    parser.add_argument('--operator-gpu', help='Optional second admitted idle H800 for kernel-only qualification')
    parser.add_argument('--regression-gpu', help='Optional second admitted idle H800 for untimed BF16 regression')
    args = parser.parse_args()
    print(json.dumps(prepare(args.metadata, args.mode, args.operator_gpu, args.regression_gpu)))
