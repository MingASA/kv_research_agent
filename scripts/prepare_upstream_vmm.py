"""Freeze three untimed model checks of the existing elastic VMM implementation."""

import hashlib
import json
from pathlib import Path
import shutil

from scripts.controller.durable import digest
from scripts.controller.policy import validate_request
from scripts.controller.upstream_native import VMM_KIND
from scripts.controller.upstream_vmm import PROFILES
from scripts.upstream_package_runtime import freeze_runtime, attach_graph_observer


def main():
    root = Path(__file__).resolve().parents[1]
    parent = json.loads((root/'plans/experiments/upstream-C-bf16-qualification-r0-20261102b/package.json').read_bytes())
    request = dict(parent['request'])
    run = 'upstream-C-VMM-compatibility-r0-20261102a'
    plan = root/'plans/experiments'/run
    plan.mkdir()
    files = {}
    for name in ('input/validation.json', 'input/gdn_dependency_CPU_import.json', 'input/source.bundle'):
        target = plan/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent['assets'][name]['local_path'], target)
        files[name] = target
    run_root = request['approved_workspace']+'/agent_work/runs/'+run
    request.update(run_id=run, remote_output=run_root, python_source_repo=run_root+'/source_repo',
        dataset_path=run_root+'/input/validation.json', arms=['fp8_e4m3'], max_wall_seconds=5400,
        workload={'kind': VMM_KIND, 'scope': 'BF16_FP8_NVFP4_elastic_VMM_model_prefix_Graph_no_benchmark'},
        operators={'fp8_e4m3': {'planner': 'frozen_three_VMM_profiles', 'decode': 'flashinfer_or_official_auto'}},
        VMM_profiles=list(PROFILES), research_reason='Finite actual VMM model/Prefix/GDN/Graph compatibility; independent of formaldefaultA/C',
        experiment_key=digest({'kind': VMM_KIND, 'profiles': PROFILES, 'source': request['server_source_commit'],
                               'service': request['service'], 'GPU': request['gpu_uuids']}))
    files.update(freeze_runtime(plan))
    files.update(attach_graph_observer(plan, request))
    approval = json.loads((Path.home()/'.config/kv-research/approval.json').read_bytes())
    validate_request(request, approval)
    assets = {name: {'local_path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
              for name, path in files.items()}
    package = {'schema': 'kv-frozen-local-package-v1', 'classification': 'proposal',
               'request': request, 'request_sha256': digest(request), 'assets': assets}
    (plan/'request.json').open('x').write(json.dumps(request, indent=2)+'\n')
    (plan/'package.json').open('x').write(json.dumps(package, indent=2)+'\n')
    print(json.dumps({'run_id': run, 'package': str(plan/'package.json'), 'dispatched': False}))


if __name__ == '__main__':
    main()
