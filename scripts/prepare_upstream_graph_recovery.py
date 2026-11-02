"""Reuse completed A model/prefix proof and qualify only its missing serving Graph evidence."""

import hashlib
import json
from pathlib import Path
import shutil

from scripts.controller.durable import digest
from scripts.controller.policy import validate_request
from scripts.upstream_package_runtime import freeze_runtime, attach_graph_observer

ROOT = Path(__file__).resolve().parents[1]
RUN = 'upstream-A-qwen38-graph-r2-20261102e'


def prepare():
    original = ROOT/'plans/experiments/upstream-A-qwen38-qualification-r1-20261102d/package.json'
    package = json.loads(original.read_bytes())
    request = dict(package['request'])
    proof_path = ROOT/'reports/operations/upstream-ac-20261102a/A_MODEL_PREFIX_PARTIAL_AUDIT-d1.json'
    certificate = json.loads(proof_path.read_bytes())
    if (certificate.get('actual_model_and_prefix_verified') is not True
            or certificate.get('known_answers_checked') != 16 or certificate.get('original_run_status') != 'FAILED'
            or certificate['source_commit'] != request['server_source_commit']
            or certificate['binary_sha256'] != request['binary_sha256']):
        raise ValueError('Graph recovery requires the actual independently verified model/prefix components')
    plan = ROOT/'plans/experiments'/RUN
    plan.mkdir()
    inputs = plan/'input'
    inputs.mkdir()
    files = {}
    for name in ('input/validation.json', 'input/gdn_dependency_CPU_import.json', 'input/source.bundle'):
        target = plan/name
        shutil.copyfile(package['assets'][name]['local_path'], target)
        files[name] = target
    shutil.copyfile(proof_path, inputs/'partial_model_proof.json')
    files['input/partial_model_proof.json'] = inputs/'partial_model_proof.json'
    request.pop('packaging_recovery', None)
    run_root = request['approved_workspace']+'/agent_work/runs/'+RUN
    request.update(run_id=RUN, replicate=2, remote_output=run_root, python_source_repo=run_root+'/source_repo',
        dataset_path=run_root+'/input/validation.json',
        research_reason='Preserve original profiler failure; reuse verified16known+real8192prefix and complete only missing serving CUDA Graph evidence',
        model_proof_reuse={'prior_run_id': certificate['original_run_id'],
            'prior_result_sha256': certificate['prior_result_sha256'],
            'certificate_sha256': hashlib.sha256(proof_path.read_bytes()).hexdigest()})
    files.update(attach_graph_observer(plan, request))
    files.update(freeze_runtime(plan))
    approval = json.loads((Path.home()/'.config/kv-research/approval.json').read_bytes())
    validate_request(request, approval)
    assets = {name: {'local_path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
              for name, path in files.items()}
    frozen = {'schema': 'kv-frozen-local-package-v1', 'classification': 'proposal', 'request': request,
              'request_sha256': digest(request), 'assets': assets}
    (plan/'request.json').write_text(json.dumps(request, indent=2)+'\n')
    (plan/'package.json').write_text(json.dumps(frozen, indent=2)+'\n')
    print(json.dumps({'run_id': RUN, 'package': str(plan/'package.json'), 'reused_known_cases': 16,
                      'reused_prefix_hit_tokens': 8192, 'new_diagnostic_requests': 3, 'dispatched': False}))


if __name__ == '__main__':
    prepare()
