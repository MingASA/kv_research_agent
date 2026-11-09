"""Freeze a bounded read-only backend diagnostic; never dispatch a GPU."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from scripts.controller.cli import validate_frozen_policy
from scripts.controller.durable import digest, write_new_json
from scripts.controller.idle_admission import FAST_ADMISSION
from scripts.controller.native_next_stage import EXTRA_SOURCES
from scripts.controller.policy import validate_request
from scripts.controller.q4_component_profile import KIND, PROTOCOL, SCOPE, SOURCE_COMMIT, GRAPH_FAILURE_REFERENCE
from scripts.controller.upstream_native import SOURCE_FILES
from scripts.upstream_package_runtime import freeze_runtime


ROOT = Path(__file__).resolve().parents[1]
REPO = Path('/home/mingasa/projects/xllm-q4-readonly-profile-20261106a')


def prepare(phase: str, run_id: str, pilot_reference: dict | None = None) -> dict:
    if phase not in ('pilot', 'main') or re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,127}', run_id) is None:
        raise ValueError('Unknown diagnostic phase or fresh run ID')
    sha = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != SOURCE_COMMIT or subprocess.check_output(
            ['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Diagnostic must use the unchanged measured source in a clean worktree')
    driver = ROOT/'scripts/controller/q4_component_probe.py'
    if not driver.is_file():
        raise ValueError('Reviewed diagnostic driver is not frozen yet')
    previous = json.loads((ROOT/'plans/experiments/upstream-fast-20261106g-mxfp4-service-0001/package.json').read_bytes())
    fixture = json.loads((ROOT/'plans/experiments/upstream-C-operators-qualification-r0-20261102a/package.json').read_bytes())
    request = copy.deepcopy(previous['request'])
    for key in ('benchmark_cases', 'service_benchmark', 'operational_attempt', 'operational_retry_group',
                'compiled_backend_source_sha256', 'graph_observer', 'cpu_native_jit_artifact'):
        request.pop(key, None)
    plan = ROOT/'plans/experiments'/run_id
    plan.mkdir()
    (plan/'input').mkdir()
    files = {}
    for name in ('validation.json', 'gdn_dependency_CPU_import.json'):
        target = plan/'input'/name
        shutil.copyfile(fixture['assets']['input/'+name]['local_path'], target)
        files['input/'+name] = target
    bundle = plan/'input/source.bundle'
    subprocess.run(['git', '-C', str(REPO), 'bundle', 'create', str(bundle),
                    'HEAD', '^da7c6446ff4361c8060633b04d6bfb77e6f737ff'], check=True)
    files['input/source.bundle'] = bundle
    remote = request['approved_workspace']+'/agent_work/runs/'+run_id
    request.update(
        run_id=run_id, remote_output=remote, python_source_repo=remote+'/source_repo',
        python_source_commit=sha, dataset_path=remote+'/input/validation.json',
        dataset_sha256=hashlib.sha256(files['input/validation.json'].read_bytes()).hexdigest(),
        arms=['mxfp4'], max_wall_seconds=1800, max_output_gib=1,
        gpu_uuids=['GPU-0c628a64-5f31-3501-fa3c-723f5b69254a'],
        workload={'kind': KIND, 'scope': SCOPE}, native_configuration=None,
        operators={'mxfp4': {'planner': 'private_pinned_FA2_E8M0', 'decode': 'private_pinned_FA2_E8M0'}},
        server_binary_execution='none', profile_protocol=copy.deepcopy(PROTOCOL), profile_phase=phase,
        graph_qualification_failure_reference=copy.deepcopy(GRAPH_FAILURE_REFERENCE),
        probe_driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        resource_admission=dict(FAST_ADMISSION),
        backend_source_sha256={name: hashlib.sha256((REPO/name).read_bytes()).hexdigest()
                               for name in (*SOURCE_FILES, *EXTRA_SOURCES)},
        research_reason='Read-only fixed-shape Q4 backend cost attribution; no model/service/quality reruns',
        experiment_key=digest({'phase': phase, 'source': sha, 'driver': hashlib.sha256(driver.read_bytes()).hexdigest(),
                               'protocol': PROTOCOL}),
    )
    if pilot_reference is not None:
        request['profile_pilot_reference'] = dict(pilot_reference)
    approval = json.loads((Path.home()/'.config/kv-research/approval.json').read_bytes())
    validate_request(request, approval)
    files.update(freeze_runtime(plan))
    program = '''import importlib,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
driver=importlib.import_module('kv_controller.q4_component_probe')
guard=importlib.import_module('kv_controller.q4_component_profile')
assert Path(driver.__file__).resolve().is_relative_to(Path(sys.argv[1]))
assert driver.torch is None
assert not any(n.split('.')[0] in ('torch','flashinfer','triton','cuda') for n in sys.modules)
assert driver.describe('main')['geometry']=={'Q':24,'KV':4,'D':256,'page':128}
assert [[('prefill' if c['query_length']>1 else 'decode'),c['batch'],c['context']] for c in driver.describe('main')['cases']]==guard.PROTOCOL['main_cases']
print(json.dumps({'Q4_frozen_import_contract':'PASS','GPU_initialized':False,'installed_package_imported':False}))
'''
    with tempfile.TemporaryDirectory(prefix='q4-cold-import-') as directory:
        cold = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', program,
                               str((plan/'runtime').resolve())], cwd=directory,
                              capture_output=True, text=True, check=True, timeout=30)
    write_new_json(plan/'Q4_FROZEN_ENTRY_IMPORT_CPU.json', json.loads(cold.stdout))
    package = {'schema': 'kv-frozen-local-package-v1', 'classification': 'proposal', 'request': request,
               'request_sha256': digest(request), 'assets': {name: {'local_path': str(path),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in files.items()}}
    path = plan/'package.json'
    write_new_json(path, package)
    write_new_json(plan/'request.json', request)
    validate_frozen_policy(path, request, approval)
    return {'package': str(path), 'phase': phase, 'source': sha,
            'driver_sha256': request['probe_driver_sha256'], 'actual_frozen_policy_passed': True, 'GPU_dispatched': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('pilot', 'main'), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--pilot-run-id')
    parser.add_argument('--pilot-result-sha256')
    args = parser.parse_args()
    if bool(args.pilot_run_id) != bool(args.pilot_result_sha256):
        parser.error('Both immutable pilot identity fields are required together')
    reference = ({'run_id': args.pilot_run_id, 'result_sha256': args.pilot_result_sha256}
                 if args.pilot_run_id else None)
    print(json.dumps(prepare(args.phase, args.run_id, reference)))
