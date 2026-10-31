"""Freeze a CPU-verified private GDN dependency repair; never launches GPU work."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from scripts.controller.approval import TRUSTED_APPROVAL, load_approval_snapshot
from scripts.controller.durable import write_new_bytes, write_new_json
from scripts.controller.nextgen import GDN_OVERLAY
from scripts.controller.policy import validate_request


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    prior_id = 'nextgen-bf16-qwen38-validation-r1-20261027d'
    run_id = 'nextgen-bf16-qwen38-validation-r2-20261027e'
    original = json.loads((root/'plans/experiments'/prior_id/'package.json').read_text())
    request = copy.deepcopy(original['request'])
    ops = root/'reports/operations/nextgen-native-kv-20261027a'
    mirror = root/'results/mirrors'/(prior_id+'-failed-d1')/'files/support/agent_work/runs'/prior_id
    raw = (mirror/'result.json').read_bytes()
    result = json.loads(raw)
    audit = json.loads((ops/'BF16_d_main_independent_59SHA_a1.json').read_text())
    if (result['status'] != 'FAILED' or not result['gpu_release_verified'] or result['live_owned_containers']
            or audit['files_verified'] != 59 or audit['result_sha256'] != hashlib.sha256(raw).hexdigest()):
        raise ValueError('Original dependency failure lacks complete independent released evidence')
    proof_path = ops/'GDN_private_overlay_CPU_import_a1.json'
    proof = json.loads(proof_path.read_text())
    if (proof['CPU_only'] is not True or proof['core_files_SHA_verified'] != 208
            or proof['borrowed_files_SHA_verified'] != 195 or proof['probe']['GPU_device_nodes'] != []
            or not all(proof['probe'][name]['import_ok'] for name in ('cutlass', 'cutlass.cute', 'flashinfer.gdn_prefill'))):
        raise ValueError('Private dependency overlay has not passed actual CPU imports')
    remote = request['approved_workspace']+'/agent_work/runs/'+run_id
    request.update(run_id=run_id, remote_output=remote, dataset_path=remote+'/input/validation.json',
                   python_source_repo=remote+'/source_repo', replicate=2,
                   environment_recovery={
                       'prior_run_id': prior_id, 'prior_result_sha256': hashlib.sha256(raw).hexdigest(),
                       'reason': 'missing_DSL_core_and_embedded_site_path', 'overlay_name': GDN_OVERLAY,
                       'overlay_manifest_sha256': proof['manifest_sha256'],
                       'CPU_import_evidence_sha256': hashlib.sha256(proof_path.read_bytes()).hexdigest(),
                   })
    approval, approval_sha = load_approval_snapshot(TRUSTED_APPROVAL)
    validate_request(request, approval)
    output = root/'plans/experiments'/run_id
    output.mkdir(exist_ok=False)
    assets = {}

    def add(name: str, data: bytes) -> None:
        path = output/name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_new_bytes(path, data)
        assets[name] = {'local_path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}

    for name, entry in original['assets'].items():
        if name.startswith('runtime/'):
            continue
        data = Path(entry['local_path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Original validation asset drift: '+name)
        add(name, data)
    for path in sorted((root/'scripts/controller').glob('*.py')):
        add('runtime/kv_controller/'+path.name, path.read_bytes())
    add('input/gdn_dependency_CPU_import.json', proof_path.read_bytes())
    add('input/model_assets_full_SHA_a6.json', (ops/'model_assets_full_SHA_a6.json').read_bytes())
    write_new_json(output/'request.json', request)
    write_new_json(output/'package.json', {
        'schema': 'kv-frozen-local-package-v1', 'classification': 'proposal',
        'request': request, 'approval_sha256_at_preparation': approval_sha, 'assets': assets,
        'model_service_quality_performance_passed': False,
    })


if __name__ == '__main__':
    main()
