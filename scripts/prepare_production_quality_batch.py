#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Freeze paired full1719 quality at the production C128 client envelope."""

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
from scripts.controller.durable import digest, secure_directory, write_new_bytes, write_new_json
from scripts.controller.policy import XQA_SOURCES
from scripts.prepare_bf16_xqa_quality_reference import PERFORMANCE_SOURCE, SOURCE


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relocate(value: object, old: str, new: str) -> object:
    if isinstance(value, dict):
        return {key: _relocate(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_relocate(item, old, new) for item in value]
    return value.replace(old, new) if isinstance(value, str) else value


def prepare(base: Path, out: Path, run_id: str) -> dict:
    parent = json.loads(base.read_text())
    if parent['request']['workload']['kind'] != 'native_quality_xqa_bf16_reference':
        raise ValueError('Use the frozen, fully verified BF16-XQA original1719 source package')
    for name, entry in parent['assets'].items():
        if _sha(Path(entry['local_path'])) != entry['sha256']:
            raise ValueError('Parent input/runtime/bundle asset changed: ' + name)
    for name in XQA_SOURCES:
        if (SOURCE/name).read_bytes() != subprocess.check_output(['git', 'show', PERFORMANCE_SOURCE+':'+name], cwd=SOURCE):
            raise ValueError('Production quality must keep the measured inference engine unchanged')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    if commit != parent['request']['python_source_commit'] or subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=SOURCE):
        raise ValueError('Quality tools must remain the committed, already verified source')
    if out.exists():
        raise FileExistsError('Production-batch quality needs a new immutable package')
    secure_directory(out)
    old = parent['request']['remote_output']
    remote = parent['request']['approved_workspace']+'/agent_work/runs/'+run_id
    request = _relocate(copy.deepcopy(parent['request']), old, remote)
    request.update(run_id=run_id, concurrency=128, arms=['bf16', 'fp8_e4m3'], replicate=0,
                   max_wall_seconds=7200, max_output_gib=5,
                   research_reason='Verify actual production C128 quality on the original full1719 records with both deployable BF16/FP8 XQA paths; previous C8 quality cannot certify C128. Same inference engine as serving source1c2, no new scale or answer-based tuning, not extra independent dataset examples.')
    request['workload'].update(kind='native_quality_xqa_production_batch', cohort='full_production_batch')
    approved = json.loads(Path(parent['assets']['input/approved_data_manifest.json']['local_path']).read_text())
    write_new_json(out/'approved_data_manifest.json', _relocate(approved, old, remote))
    proof = {'classification': 'proposal_CPU_only', 'source_commit': commit,
             'measured_engine_source_commit': PERFORMANCE_SOURCE, 'inference_engine_bytes_identical': True,
             'original_records_per_arm': 1719, 'client_concurrency': 128,
             'actual_GPU_batch_fixed': False, 'old_C8_predictions_reused': 0,
             'purpose': 'Changed production batching and same full-cohort paired comparison, not an enlarged independent sample',
             'parent_package_sha256': _sha(base), 'preparation_script_sha256': _sha(Path(__file__))}
    write_new_json(out/'production_batch_provenance.json', proof)
    files = {name: Path(entry['local_path']) for name, entry in parent['assets'].items()
             if not name.startswith('runtime/') and name not in ('input/approved_data_manifest.json', 'input/reference_provenance.json')}
    for name, entry in parent['assets'].items():
        if not name.startswith('runtime/'):
            continue
        module = Path(name).name
        source = ROOT/'scripts/audit_native_results.py' if module == 'evidence.py' else ROOT/'scripts/controller'/module
        frozen = out/'frozen_assets'/name; secure_directory(frozen.parent)
        write_new_bytes(frozen, source.read_bytes()); files[name] = frozen
    files['input/approved_data_manifest.json'] = out/'approved_data_manifest.json'
    files['input/production_batch_provenance.json'] = out/'production_batch_provenance.json'
    scientific = {key: request[key] for key in ('python_source_commit', 'binary_sha256', 'backend_source_sha256',
        'dataset_sha256', 'image_id', 'graph_mode', 'prefix_cache', 'fp8_scales', 'operators', 'service', 'concurrency')}
    scientific['workload'] = {key: request['workload'][key] for key in (
        'kind', 'cohort', 'samples_per_arm', 'ignore_eos', 'source_content_sha256')}
    request['experiment_key'] = digest(scientific)
    package = {'schema': 'kv-frozen-local-package-v1', 'classification': 'proposal', 'request': request,
               'gpu_job_launched': False,
               'assets': {name: {'local_path': str(path), 'sha256': _sha(path)} for name, path in files.items()}}
    write_new_json(out/'package.json', package)
    return package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    package = prepare(args.base, args.out.absolute(), args.run_id)
    print(json.dumps({'package': str(args.out/'package.json'), 'arms': package['request']['arms'],
                      'client_concurrency': 128, 'original_records_per_arm': 1719, 'GPU_started': False}))


if __name__ == '__main__':
    main()
