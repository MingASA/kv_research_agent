#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Freeze genuine long-document QA service inputs, without GPU or remote writes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.controller.durable import digest, secure_directory, write_new_bytes, write_new_json
from scripts.controller.policy import BF16_XQA_OPERATORS, XQA_SOURCES
from scripts.controller.range_probe import LONG_QA_SCOPE, validate_longcontext_inputs

SOURCE = Path('/home/mingasa/projects/xllm_quant_kvcache-agent-bf16-xqa-20261027')


def prepare(base: Path, data: Path, bundle: Path, out: Path, run_id: str, gpu_uuid: str) -> dict:
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', run_id) is None:
        raise ValueError('Exact unique run ID required')
    if re.fullmatch(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', gpu_uuid) is None:
        raise ValueError('Exact discovered GPU UUID required')
    original = json.loads(base.read_text())
    request = original['request']
    manifest = json.loads((data/'manifest.json').read_text())
    canonical = json.loads((data/'canonical_inputs.json').read_text())
    validate_longcontext_inputs(canonical, manifest)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(data/'canonical_inputs.json') != manifest['canonical_sha256']:
        raise ValueError('Original natural QA input bytes differ')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    if (commit != '11e7cddbfe33eb21d209c7f48f5e461ac56545eb'
            or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=SOURCE, text=True)):
        raise ValueError('Use the clean, previously verified modern production source')
    if sha(bundle) != original['assets']['input/source.bundle']['sha256']:
        raise ValueError('Use the unchanged verified source bundle')
    if out.exists():
        raise FileExistsError('New immutable service package directory required')
    remote = request['approved_workspace']+'/agent_work/runs/'+run_id
    for key in ('dataset_archive', 'retry_evidence', 'diagnostic', 'scale_calibration'):
        request.pop(key, None)
    request.update(run_id=run_id, remote_output=remote, replicate=0, concurrency=128,
        gpu_uuids=[gpu_uuid], arms=['bf16', 'fp8_e4m3'], max_wall_seconds=1800, max_output_gib=5,
        python_source_repo=remote+'/source_repo', python_source_commit=commit,
        backend_source_sha256={name:sha(SOURCE/name) for name in XQA_SOURCES},
        operators=BF16_XQA_OPERATORS, fp8_scales={'key':.125,'value':.0625},
        dataset_path=remote+'/input/canonical_inputs.json', dataset_sha256=manifest['canonical_sha256'],
        workload={'kind':'long_document_QA_natural_xqa_bf16_reference', 'requests':128, 'warmup':0,
            'ignore_eos':False, 'content_scope':LONG_QA_SCOPE,
            'source_content_sha256':manifest['source_content_sha256'],
            'arrival_mode':'closed_loop_concurrency_finite_cohort',
            'official_generation_caps_preserved':True, 'activation_diagnostic_enabled':False,
            'aiperf_zero_warmup_policy':'omit_explicit_option',
            'aiperf_cli_config_sha256':'2be937b185ddfe3a19e8090aa4411f1015a2e00899d29db3055d3f7e19bdd42b',
            'input_manifest':remote+'/input/production_manifest.json'},
        research_reason='Production long-document QA input-heavy service:128 original real questions/documents, '
            'unmodified benchmark prompts/caps32/64/128 and natural EOS, no padding or forced generation. '
            'Compare modern FA3-prefill/XQA-BF16-decode to resident FP8 FA2-prefill/XQA-internal-FP8-Q decode '
            'with actual Graph/prefix/chunked scheduler. Finite C128 saturation cohort, not original arrival replay '
            'or established steady-state SLO capacity. Same inputs previously observed for BF16 KV calibration; '
            'no activation counter/profiler enabled here. Existing FP8 quality tradeoff remains unaccepted.')
    scientific = {k:request[k] for k in ('python_source_commit', 'binary_sha256', 'dataset_sha256',
        'backend_source_sha256', 'image_id', 'graph_mode', 'prefix_cache', 'fp8_scales',
        'operators', 'service', 'concurrency')}
    scientific['workload'] = {k:v for k,v in request['workload'].items() if k != 'input_manifest'}
    request['experiment_key'] = digest(scientific)
    secure_directory(out)
    files = {'runtime/kv_controller/'+p.name:p for p in (ROOT/'scripts/controller').glob('*.py')}
    files['runtime/kv_controller/evidence.py'] = ROOT/'scripts/audit_native_results.py'
    files['runtime_support/tokenizer_compat/sitecustomize.py'] = Path(
        original['assets']['runtime_support/tokenizer_compat/sitecustomize.py']['local_path'])
    files.update({'input/source.bundle':bundle, 'input/canonical_inputs.json':data/'canonical_inputs.json',
                  'input/production_manifest.json':data/'manifest.json'})
    for name, path in list(files.items()):
        target = out/'frozen_assets'/name
        secure_directory(target.parent)
        write_new_bytes(target, path.read_bytes())
        files[name] = target
    package = {'schema':'kv-frozen-local-package-v1', 'classification':'proposal', 'gpu_job_launched':False,
        'request':request, 'assets':{name:{'local_path':str(path.absolute()), 'sha256':sha(path)}
                                    for name,path in files.items()}}
    write_new_json(out/'package.json', package)
    return package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-package', 'data', 'bundle', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    package = prepare(args.base_package, args.data, args.bundle, args.out, args.run_id, args.gpu_uuid)
    print(json.dumps({'package':str(args.out/'package.json'), 'assets':len(package['assets']),
                      'requests':128, 'GPU_started':False}))


if __name__ == '__main__':
    main()
