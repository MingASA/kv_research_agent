"""Use the established standalone evidence module and verify frozen entry imports."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def freeze_runtime(plan: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    runtime = plan/'runtime/kv_controller'
    runtime.mkdir(parents=True)
    for path in (root/'scripts/controller').glob('*.py'):
        source = root/'scripts/audit_native_results.py' if path.name == 'evidence.py' else path
        shutil.copyfile(source, runtime/path.name)
    program = '''import importlib,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
for name in ('worker','nextgen_multiturn','native_tools','upstream_qualification','upstream_native','upstream_operator','upstream_service','upstream_quality','upstream_tools','upstream_vmm','upstream_fp8_diagnosis','upstream_graph_observer','production','native_next_stage','fp8_optimization_operator','fp8_numeric_model','next_stage_deployment'):
 module=importlib.import_module('kv_controller.'+name)
 assert Path(module.__file__).resolve().is_relative_to(Path(sys.argv[1]))
assert not any(n=='scripts' or n.startswith('scripts.') for n in sys.modules)
assert not any(n.split('.')[0] in ('torch','flashinfer','triton','cuda') for n in sys.modules)
print(json.dumps({'frozen_entry_imports_verified':True,'control_checkout_required':False,'GPU_used':False}))
'''
    with tempfile.TemporaryDirectory(prefix='upstream-frozen-import-') as directory:
        result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', program, str(runtime.parent.resolve())],
            cwd=directory, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError('Frozen entry import failed before dispatch: '+result.stderr[-3000:])
    proof = json.loads(result.stdout)
    (plan/'FROZEN_ENTRY_IMPORT_CPU.json').open('x').write(json.dumps(proof, indent=2)+'\n')
    return {'runtime/kv_controller/'+p.name: p for p in runtime.glob('*.py')}


def attach_graph_observer(plan: Path, request: dict) -> dict:
    import hashlib
    root = Path(__file__).resolve().parents[1]
    dependency = root/'plans/dependencies/cuda-graph-counter-20261102c'
    proof = json.loads((dependency/'CPU_BUILD_AND_FORWARDING.json').read_bytes())
    if (proof['CPU_forwarding_verified'] is not True
            or hashlib.sha256((dependency/'libkvra_graph_counter.so').read_bytes()).hexdigest() != proof['library_sha256']
            or hashlib.sha256((root/'scripts/cuda_graph_counter.c').read_bytes()).hexdigest() != proof['source_sha256']):
        raise ValueError('Observer lacks its exact CPU forwarding and source proof')
    files = {}
    for name, source in [('libkvra_graph_counter.so', dependency/'libkvra_graph_counter.so'),
                         ('cuda_graph_counter.c', root/'scripts/cuda_graph_counter.c'),
                         ('graph_observer_CPU_proof.json', dependency/'CPU_BUILD_AND_FORWARDING.json')]:
        target = plan/'input'/name
        shutil.copyfile(source, target)
        files['input/'+name] = target
    request['graph_observer'] = {'kind': 'transparent_cuda_runtime_graph_counter_v1',
        'source_sha256': proof['source_sha256'], 'library_sha256': proof['library_sha256'],
        'scope': 'untimed_qualification_only_never_formal_service_benchmarks'}
    return files


def attach_remote_asset_reuse(package: dict, parent_run: str | None) -> None:
    """Use the existing SHA-bound remote reuse protocol for completed own inputs."""
    if parent_run is None:
        return
    import re
    from scripts.controller.durable import digest
    from scripts.controller.quota import QuotaLedger
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', parent_run) is None:
        raise ValueError('Invalid completed asset parent')
    root = Path(__file__).resolve().parents[1]
    ticket = json.loads((root/'state/controller/tickets'/(parent_run+'.json')).read_bytes())
    with QuotaLedger(root/'state/controller') as ledger:
        row = ledger.get(parent_run)
    result = json.loads(row['evidence_json'])
    if (row['state'] != 'SETTLED' or result['status'] != 'COMPLETE'
            or result['gpu_release_verified'] is not True or result['live_owned_containers'] != []
            or ticket['request']['model_id'] != package['request']['model_id']):
        raise ValueError('Input parent must be actually complete, released and bound to the same model')
    package['remote_asset_reuse'] = {name: {'sha256': asset['sha256'], 'reuse_from_run_id': parent_run,
        'reuse_parent_result_sha256': digest(result)} for name, asset in package['assets'].items()
        if name.startswith(('input/', 'runtime/')) and ticket['assets_sha256'].get(name) == asset['sha256']}
