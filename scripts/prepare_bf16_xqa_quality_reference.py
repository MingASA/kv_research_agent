#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Freeze the full, original 1719-record BF16-XQA quality reference on CPU."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.controller.durable import canonical_bytes, digest, secure_directory, write_new_bytes, write_new_json
from scripts.controller.policy import BF16_XQA_OPERATORS, QUALITY_SOURCES, XQA_SOURCES
from scripts.controller.protocols import TASKS, restore_quality_records

SOURCE = Path('/home/mingasa/projects/xllm_quant_kvcache-agent-bf16-xqa-20261027')
PERFORMANCE_SOURCE = '1c2be15976b7fbb1d8b79e469659b2c8e52cff00'
MIRROR = ROOT / 'results/mirrors/quality-protocols-20261026T075846Z/files'


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(base: Path, quality_assets: Path, bundle: Path, out: Path, run_id: str) -> dict:
    original = json.loads(base.read_text())
    historical = json.loads(quality_assets.read_text())
    request = copy.deepcopy(original['request'])
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=SOURCE):
        raise ValueError('Quality source must be committed before freezing')
    bundle_heads = subprocess.check_output(['git', 'bundle', 'list-heads', str(bundle)], text=True)
    if commit not in [line.split()[0] for line in bundle_heads.splitlines()]:
        raise ValueError('Bundle does not contain the committed quality reference')
    engine_hashes = {}
    for name in XQA_SOURCES:
        measured = subprocess.check_output(['git', 'show', PERFORMANCE_SOURCE + ':' + name], cwd=SOURCE)
        if measured != (SOURCE / name).read_bytes():
            raise ValueError('Quality reference changed a measured inference engine file: ' + name)
        engine_hashes[name] = _sha(SOURCE / name)
    batch = MIRROR / 'support/experiments/joint-kv-14b/joint14b-full-quality-v1-8gpu'
    records, parent = restore_quality_records(batch)
    if out.exists():
        raise FileExistsError('A quality package must use a new immutable directory')
    secure_directory(out)
    cohort = out / 'cohort'
    secure_directory(cohort)
    for task in TASKS:
        write_new_bytes(cohort / (task + '.jsonl'), b''.join(canonical_bytes(row) for row in records[task]))
    manifest = {'schema': 'kv-full-quality-reference-cohort-v1', 'classification': 'proposal',
                'scope': 'one_unique_dataset_round', 'samples_per_arm': 1719,
                'parent_shard_manifest_sha256': _sha(batch / 'shard_manifest.json'),
                'source_task_sha256': {task: parent['sources'][task]['sha256'] for task in TASKS},
                'source_indices': {task: list(range(len(records[task]))) for task in TASKS},
                'data_sha256': {task: _sha(cohort / (task + '.jsonl')) for task in TASKS},
                'selection_rule': 'All original frozen records in global source order; no subsampling or answer-based selection',
                'preparation_uses_gpu': False, 'model_predictions_generated': False,
                'preparation_script_sha256': _sha(Path(__file__))}
    write_new_json(cohort / 'full_reference_manifest.json', manifest)
    sys.path.insert(0, str(SOURCE))
    from tools import run_qwen25_joint_kv_quality as runner
    scorer = MIRROR / 'support/inputs/longbench-scorer'
    shots = Path(historical['assets']['input/five_train_examples.json']['local_path'])
    args = runner._parser().parse_args([
        '--model', str(MIRROR / 'model_metadata'), '--gsm8k', str(cohort / 'gsm8k.jsonl'),
        '--gsm8k-shots', str(shots), '--longbench-data', str(cohort), '--longbench-repo', str(scorer),
        '--attention-backend', 'native-flashinfer', '--native-planner', 'fa3', '--native-decode', 'xqa',
        '--fp8-k-scale', '0.125', '--fp8-v-scale', '0.0625', '--graph-mode', 'cudagraphs',
        '--arms', 'auto', '--max-seqs-per-batch', '128', '--concurrency', '8', '--cohort-scope', 'full',
        '--binary-source-commit', request['server_source_commit'], '--output', str(out / 'unused')])
    runner._validate_args(args)
    audit = runner._data_audit(args, {task: cohort / (task + '.jsonl') for task in TASKS})
    remote = request['approved_workspace'] + '/agent_work/runs/' + run_id
    for task, item in audit['datasets'].items():
        item['path'] = remote + '/input/cohort/' + task + '.jsonl'
    write_new_json(out / 'approved_data_manifest.json', {'data_audit': audit})
    proof = {'classification': 'confirmed_source_bytes_only', 'measured_performance_source_commit': PERFORMANCE_SOURCE,
             'quality_reference_source_commit': commit, 'engine_files_sha256': engine_hashes,
             'whole_runtime_identical_claim': False, 'quality_tools_changed': True,
             'new_FP8_predictions': 0, 'purpose': 'Validate the changed BF16 deployment reference, retain previous FP8 predictions'}
    write_new_json(out / 'reference_provenance.json', proof)
    files = {name: Path(entry['local_path']) for name, entry in historical['assets'].items()
             if name.startswith('input/scorer/') or name == 'input/five_train_examples.json'}
    files.update({name: Path(entry['local_path']) for name, entry in original['assets'].items()
                  if name.startswith('runtime_support/')})
    for module in ('__init__.py', 'approval.py', 'durable.py', 'policy.py', 'processes.py', 'supervisor.py',
                   'watchdog.py', 'worker.py', 'production.py', 'trace_serving.py', 'service_metrics.py'):
        files['runtime/kv_controller/' + module] = ROOT / 'scripts/controller' / module
    files['runtime/kv_controller/evidence.py'] = ROOT / 'scripts/audit_native_results.py'
    for name, path in list(files.items()):
        if name.startswith(('runtime/', 'runtime_support/')):
            frozen = out / 'frozen_assets' / name
            secure_directory(frozen.parent)
            write_new_bytes(frozen, path.read_bytes())
            files[name] = frozen
    files.update({'input/source.bundle': bundle.absolute(),
                  'input/approved_data_manifest.json': out / 'approved_data_manifest.json',
                  'input/reference_provenance.json': out / 'reference_provenance.json'})
    for path in cohort.iterdir():
        files['input/cohort/' + path.name] = path
    request.update(run_id=run_id, remote_output=remote, arms=['bf16'], concurrency=8, replicate=0,
                   python_source_repo=remote + '/source_repo', python_source_commit=commit,
                   backend_source_sha256={name: _sha(SOURCE / name) for name in XQA_SOURCES + QUALITY_SOURCES},
                   dataset_path=remote + '/input/cohort/full_reference_manifest.json',
                   dataset_sha256=_sha(cohort / 'full_reference_manifest.json'), max_wall_seconds=7200, max_output_gib=5,
                   gpu_uuids=['GPU-0cb28538-7e9f-1b66-750f-1b53ff309195'],
                   operators=copy.deepcopy(BF16_XQA_OPERATORS),
                   workload={'kind': 'native_quality_xqa_bf16_reference', 'cohort': 'full_reference',
                             'samples_per_arm': 1719, 'ignore_eos': False,
                             'source_content_sha256': _sha(cohort / 'full_reference_manifest.json'),
                             'scorer_dependencies': request['approved_workspace'] + '/experiments/joint-kv-14b/longbench-deps-py312',
                             'longbench_repo': remote + '/input/scorer',
                             'gsm8k_shots': remote + '/input/five_train_examples.json',
                             'approved_data_manifest': remote + '/input/approved_data_manifest.json'},
                   research_reason='Validate full1719 quality of the newly measured BF16 FA3-prefill/XQA-BF16-decode deployment reference; inference engine bytes unchanged from its serving pair. Existing FP8 predictions remain frozen; this is a new BF16 operator configuration, not a repeat of the old reference.')
    request['experiment_key'] = digest({key: request[key] for key in (
        'python_source_commit', 'binary_sha256', 'dataset_sha256', 'backend_source_sha256', 'image_id',
        'graph_mode', 'prefix_cache', 'fp8_scales', 'operators', 'service', 'concurrency', 'workload', 'arms')})
    package = {'schema': 'kv-frozen-local-package-v1', 'classification': 'proposal', 'request': request,
               'gpu_job_launched': False,
               'assets': {name: {'local_path': str(path), 'sha256': _sha(path)} for name, path in files.items()}}
    write_new_json(out / 'package.json', package)
    return package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'quality-assets', 'bundle', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    package = prepare(args.base, args.quality_assets, args.bundle, args.out.absolute(), args.run_id)
    print(json.dumps({'package': str(args.out / 'package.json'), 'arms': package['request']['arms'],
                      'samples_per_arm': 1719, 'gpu_job_launched': False}))


if __name__ == '__main__':
    main()
