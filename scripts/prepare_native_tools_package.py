#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Freeze the complete natural native-tool cohort and guarded runtime, CPU only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.controller.durable import digest, secure_directory, write_new_json, write_new_bytes
from scripts.controller.native_tools import validate_native_inputs
from scripts.controller.policy import BF16_XQA_OPERATORS, NATIVE_TOOL_SOURCES


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/mingasa/projects/xllm_quant_kvcache-agent-bf16-xqa-20261027')


def prepare(base_path: Path, run_id: str, out: Path, gpu_uuid: str, prepared: Path | None = None) -> dict:
    if out.exists():
        raise FileExistsError('A new immutable native-tool package is required')
    base = json.loads(base_path.read_text())
    request = copy.deepcopy(base['request'])
    commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if (commit != '11e7cddbfe33eb21d209c7f48f5e461ac56545eb'
            or subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain', '--untracked-files=no'], text=True)):
        raise ValueError('Require the unchanged tested inference checkout; no new business-code patch')
    if request['python_source_commit'] != commit:
        raise ValueError('Source bundle/base package must bind the actual tested checkout')
    for name in NATIVE_TOOL_SOURCES:
        original = subprocess.check_output(['git', '-C', str(SOURCE), 'show',
            '1c2be15976b7fbb1d8b79e469659b2c8e52cff00:' + name])
        if (SOURCE / name).read_bytes() != original:
            raise ValueError('Native API experiment must retain all previously tested inference source bytes')
    data = prepared or ROOT / 'results/prepared/bfcl-native-tools-258-20261027a'
    validate_native_inputs(data / 'tool_inputs.json', data / 'manifest.json')
    manifest = json.loads((data / 'manifest.json').read_text())
    remote = request['approved_workspace'] + '/agent_work/runs/' + run_id
    request.update(run_id=run_id, remote_output=remote, replicate=0, max_wall_seconds=3600, max_output_gib=3,
        gpu_uuids=[gpu_uuid], concurrency=128, python_source_repo=remote + '/source_repo', python_source_commit=commit,
        dataset_path=remote + '/input/canonical_inputs.json', dataset_sha256=manifest['dataset_sha256'],
        backend_source_sha256={name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in NATIVE_TOOL_SOURCES},
        operators=copy.deepcopy(BF16_XQA_OPERATORS),
        workload={'kind': 'native_tools_xqa_bf16_reference', 'requests': 258, 'warmup': 16, 'ignore_eos': False,
            'source_commit': manifest['source_commit'], 'source_content_sha256': manifest['source_sha256'][
                'berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_live_simple.json'],
            'tool_call_parser': 'qwen25', 'content_scope': 'original_BFCL_live_simple_messages_native_tools',
            'response_mode': manifest.get('response_mode', 'streaming'),
            'arrival_mode': 'closed_loop_concurrency', 'input_manifest': remote + '/input/production_manifest.json'},
        research_reason='Production task-success gap: all258 original native tool-name/argument requests plus16 separatewarm, naturalEOS/auto toolchoice, advancedBF16-XQA versusFP8-XQA. Frozen existing inference code, explicit nativeAPI parser/usage/Graph gate; not anonymousToolAgent or fullmultistepagent/externaltool execution. Official Python checker applied independently; two empty-reference-candidate records retained in full denominator, not tuned or removed.')
    request.pop('dataset_archive', None)
    request.pop('diagnostic', None)
    request['experiment_key'] = digest({key: request[key] for key in (
        'python_source_commit', 'binary_sha256', 'dataset_sha256', 'backend_source_sha256', 'image_id',
        'graph_mode', 'prefix_cache', 'fp8_scales', 'operators', 'service', 'concurrency', 'workload')})
    files = {'runtime/kv_controller/' + path.name: path for path in (ROOT / 'scripts/controller').glob('*.py')}
    files['runtime/kv_controller/evidence.py'] = ROOT / 'scripts/audit_native_results.py'
    files['runtime_support/tokenizer_compat/sitecustomize.py'] = Path(base['assets'][
        'runtime_support/tokenizer_compat/sitecustomize.py']['local_path'])
    files.update({'input/canonical_inputs.json': data / 'tool_inputs.json',
                  'input/production_manifest.json': data / 'manifest.json',
                  'input/source.bundle': Path(base['assets']['input/source.bundle']['local_path'])})
    secure_directory(out)
    for name, path in list(files.items()):
        if name.startswith(('runtime/', 'runtime_support/')):
            frozen = out / 'frozen_assets' / name; secure_directory(frozen.parent)
            write_new_bytes(frozen, path.read_bytes()); files[name] = frozen
    package = {'schema': 'kv-frozen-local-package-v1', 'classification': 'proposal', 'request': request,
        'gpu_job_launched': False, 'production_decision_standard': 'ops/PRODUCTION_DECISION_STANDARD.md',
        'assets': {name: {'local_path': str(path.absolute()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in files.items()}}
    write_new_json(out / 'package.json', package)
    return package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-package', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--prepared', type=Path)
    args = parser.parse_args()
    package = prepare(args.base_package, args.run_id, args.out.absolute(), args.gpu_uuid, args.prepared)
    print(json.dumps({'run_id': args.run_id, 'assets': len(package['assets']), 'GPU_started': False}))


if __name__ == '__main__':
    main()
