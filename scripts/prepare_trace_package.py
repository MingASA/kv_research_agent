#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Freeze continuous arrival/length/prefix serving inputs; no GPU or remote writes."""

from __future__ import annotations

import argparse
import hashlib
import gzip
import lzma
import json
import math
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.controller.durable import digest, secure_directory, write_new_json, write_new_bytes, canonical_bytes
from scripts.controller.policy import XQA_SOURCES, COMPONENT_SOURCES, XQA_OPERATORS, BF16_XQA_OPERATORS
from scripts.controller.trace_serving import validate_inputs


SOURCE = Path('/home/mingasa/projects/xllm_quant_kvcache-agent-xqa-20261026')
BF16_XQA_SOURCE = Path('/home/mingasa/projects/xllm_quant_kvcache-agent-bf16-xqa-20261027')
COMPONENT_SOURCE = Path('/home/mingasa/projects/xllm_quant_kvcache-agent-components-20261027')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_runtime_files(original: dict, reuse_frozen: bool) -> dict:
    files = {name:Path(entry['local_path']) for name,entry in original['assets'].items()
             if name.startswith(('runtime/', 'runtime_support/'))}
    current = {'runtime/kv_controller/'+path.name:path
               for path in (ROOT/'scripts/controller').glob('*.py')}
    current['runtime/kv_controller/evidence.py'] = ROOT/'scripts/audit_native_results.py'
    if reuse_frozen:
        if {name for name in files if name.startswith('runtime/')} != set(current):
            raise ValueError('Reused trace runtime must retain the complete current module closure')
        if any(_sha(path) != original['assets'][name]['sha256'] for name,path in files.items()):
            raise ValueError('Previously frozen trace runtime bytes changed')
        for name in ('approval.py', 'durable.py', 'processes.py', 'supervisor.py', 'watchdog.py', 'worker.py'):
            relative = 'runtime/kv_controller/'+name
            if _sha(files[relative]) != _sha(current[relative]):
                raise ValueError('Reused trace runtime has different safety/ownership code: '+name)
    else:
        files.update(current)
    return files


def validate_pre_model_retry(proof: dict, prior: dict, *, replicate: int,
                             family: str, arrival_scaling: float, profile: str,
                             source_commit: str, dataset_sha256: str,
                             max_wall_seconds: int) -> None:
    """An orchestration retry must not duplicate any completed service arm."""
    if (proof.get('classification') != 'confirmed_original_pre_model_CONFLICT_no_service_observations'
            or proof.get('run_id') != prior['run_id']
            or proof.get('request_sha256') != digest(prior)
            or proof.get('original_status') != 'CONFLICT'
            or proof.get('GPU_release_verified') is not True
            or any(proof.get(k) is not False for k in ('model_loaded', 'AIPerf_ran', 'arm_measurement_present'))
            or type(proof.get('successful_model_predictions')) is not int
            or proof['successful_model_predictions'] != 0
            or re.fullmatch(r'[0-9a-f]{64}', proof.get('terminal_result_sha256', '')) is None):
        raise ValueError('Retry requires an independently audited, released pre-model CONFLICT with no service observations')
    expected_kind = {'xqa': 'anonymous_trace_xqa',
                     'xqa-bf16-reference': 'anonymous_trace_xqa_bf16_reference'}.get(profile)
    if (expected_kind is None or prior['workload']['kind'] != expected_kind
            or prior['workload']['family'] != family
            or prior['workload']['arrival_scaling'] != arrival_scaling
            or source_commit != prior['python_source_commit']
            or proof.get('source_commit') != source_commit
            or dataset_sha256 != prior['dataset_sha256']
            or proof.get('dataset_sha256') != dataset_sha256
            or replicate != prior['replicate'] + 1
            or max_wall_seconds > prior['max_wall_seconds']):
        raise ValueError('Pre-model retry must preserve source, inputs, operator profile and bounds with the next attempt number')


def prepare(base_package: Path, data: Path, run_id: str, out: Path, bundle: Path,
            family: str, arrival_scaling: float, gpu_uuid: str | None = None,
            replicate: int = 0, research_note: str = '', archive_codec: str = 'gzip', profile: str = 'xqa',
            max_wall_seconds: int = 7200, retry_failure_audit: Path | None = None,
            reuse_frozen_runtime: bool = False) -> dict:
    if family not in ('conversation', 'toolagent', 'synthetic', 'burstgpt'):
        raise ValueError('Unknown original project trace family')
    if not math.isfinite(arrival_scaling) or not .1 <= arrival_scaling <= 4:
        raise ValueError('Unbounded or invalid offered-arrival scaling')
    if type(replicate) is not int or replicate < 0:
        raise ValueError('Explicit nonnegative replicate required')
    if archive_codec not in ('gzip','xz'):
        raise ValueError('Unknown bounded trace codec')
    if profile not in ('xqa', 'xqa-bf16-reference', 'component-diagnostic'):
        raise ValueError('Unknown explicit trace operator comparison')
    if type(max_wall_seconds) is not int or not 600 <= max_wall_seconds <= 7200:
        raise ValueError('Trace time bound may only narrow the existing two-hour maximum')
    source_repo = COMPONENT_SOURCE if profile == 'component-diagnostic' else BF16_XQA_SOURCE if profile == 'xqa-bf16-reference' else SOURCE
    original = json.loads(base_package.read_text())
    request = original['request']
    prior_request = json.loads(json.dumps(request))
    request.pop('retry_evidence', None)
    if gpu_uuid is not None:
        if re.fullmatch(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', gpu_uuid) is None:
            raise ValueError('Expected an exact discovered GPU UUID')
        request['gpu_uuids'] = [gpu_uuid]
    remote = request['approved_workspace']+'/agent_work/runs/'+run_id
    rows, manifest = validate_inputs(data/'payload_trace.jsonl', data/'manifest.json')
    if manifest['arrival_scaling'] != 1 or _sha(data/'payload_trace.jsonl') != manifest['payload_trace_sha256']:
        raise ValueError('Expected the unchanged independently prepared parent trace')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source_repo, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=source_repo, text=True):
        raise ValueError('Actual candidate source must be committed and clean')
    if reuse_frozen_runtime and (profile != 'xqa-bf16-reference'
            or prior_request['workload']['kind'] != 'anonymous_trace_xqa_bf16_reference'
            or prior_request['python_source_commit'] != commit
            or _sha(bundle) != original['assets']['input/source.bundle']['sha256']):
        raise ValueError('Frozen runtime reuse is only the same modern trace inference profile/source')
    runtime_files = trace_runtime_files(original, reuse_frozen_runtime)
    for row in rows:
        row['timestamp'] /= arrival_scaling
    if retry_failure_audit is not None:
        payload_hash = hashlib.sha256()
        for row in rows:
            payload_hash.update(canonical_bytes(row))
        validate_pre_model_retry(json.loads(retry_failure_audit.read_text()), prior_request,
            replicate=replicate, family=family, arrival_scaling=arrival_scaling, profile=profile,
            source_commit=commit, dataset_sha256=payload_hash.hexdigest(),
            max_wall_seconds=max_wall_seconds)
        if _sha(bundle) != original['assets']['input/source.bundle']['sha256']:
            raise ValueError('Pre-model retry requires the original source bundle bytes')
    if out.exists():
        raise FileExistsError('New immutable trace package directory required')
    secure_directory(out)
    with (out/'payload_trace.jsonl').open('xb') as f:
        for row in rows:
            f.write(canonical_bytes(row))
    archive_name='payload_trace.jsonl.' + ('xz' if archive_codec=='xz' else 'gz')
    with (out/archive_name).open('xb') as f:
        stream=(lzma.LZMAFile(f,mode='wb',preset=6) if archive_codec=='xz'
                else gzip.GzipFile(fileobj=f,mode='wb',filename='',mtime=0))
        with stream as compressed:
            with (out/'payload_trace.jsonl').open('rb') as source:
                for block in iter(lambda: source.read(1024*1024), b''):
                    compressed.write(block)
    manifest.update(arrival_scaling=arrival_scaling, parent_payload_sha256=_sha(data/'payload_trace.jsonl'),
                    parent_manifest_sha256=_sha(data/'manifest.json'),
                    payload_trace_sha256=_sha(out/'payload_trace.jsonl'),
                    transformation='Only timestamp divided by arrival_scaling; payloads/session order/source prefix unchanged')
    write_new_json(out/'trace_manifest.json', manifest)
    validate_inputs(out/'payload_trace.jsonl', out/'trace_manifest.json')
    files = runtime_files
    if 'runtime_support/tokenizer_compat/sitecustomize.py' not in files:
        raise ValueError('Use the validated serving base package with its real offline tokenizer compatibility asset')
    # By default freeze the complete current controller. Explicit load points
    # may retain the exact validated runtime with unchanged safety code.
    if profile == 'component-diagnostic':
        files['runtime/kv_controller/component_profile.py'] = ROOT/'scripts/controller/component_profile.py'
    files.update({'input/source.bundle': bundle.absolute(), 'input/'+archive_name: out/archive_name,
                  'input/trace_manifest.json': out/'trace_manifest.json',
                  'input/parent_trace_manifest.json': data/'manifest.json',
                  'input/preparation_provenance.json': data.parent/'provenance.json'})
    if retry_failure_audit is not None:
        frozen_proof = out/'prior_pre_model_conflict_audit.json'
        write_new_bytes(frozen_proof, retry_failure_audit.read_bytes())
        files['input/prior_pre_model_conflict_audit.json'] = frozen_proof
    # A later controller change cannot silently change this proposal's runtime.
    for name, source_path in list(files.items()):
        if name.startswith(('runtime/', 'runtime_support/')):
            frozen = out/'frozen_assets'/name
            secure_directory(frozen.parent)
            write_new_bytes(frozen, source_path.read_bytes())
            files[name] = frozen
    request.update(run_id=run_id, remote_output=remote, concurrency=1024, replicate=replicate,
        python_source_repo=remote+'/source_repo', python_source_commit=commit,
        backend_source_sha256={name: _sha(source_repo/name) for name in (COMPONENT_SOURCES if profile == 'component-diagnostic' else XQA_SOURCES)},
        dataset_path=remote+'/input/payload_trace.jsonl', dataset_sha256=manifest['payload_trace_sha256'],
        dataset_archive={'relative_path': 'input/'+archive_name,'codec':archive_codec,
                         'uncompressed_size_bytes': (out/'payload_trace.jsonl').stat().st_size},
        max_output_gib=10, max_wall_seconds=max_wall_seconds, arms=['bf16', 'fp8_e4m3'], operators=XQA_OPERATORS,
        workload={'kind': 'anonymous_trace_xqa', 'family': family, 'requests': len(rows), 'warmup': 0, 'ignore_eos': True,
                  'arrival_mode': 'fixed_schedule_continuous_warm_cache', 'arrival_scaling': arrival_scaling,
                  'content_scope': 'anonymous_arrival_length_prefix_emulation',
                  'source_content_sha256': manifest['source_sha256'], 'trace_manifest': remote+'/input/trace_manifest.json'},
        research_reason='Production support decision on the original '+family+' workload: continuous anonymous arrival/length/prefix '
            'replay, original per-request OSL, explicit offered/actual scheduling, all errors and drain retained, modern deployable '
            'FA3 BF16 versus finite-initialized FA2-prefill/XQA-decode FP8. Separate semantic quality evidence is mandatory.')
    if profile in ('xqa-bf16-reference', 'component-diagnostic'):
        request['operators'] = BF16_XQA_OPERATORS
        request['workload']['kind'] = 'anonymous_trace_xqa_bf16_reference'
        request['research_reason'] = ('Production load curve with advanced BF16 reference: same continuous source '
            + family + ' arrivals/lengths/prefix relations, explicit clock scaling, FA3-prefill/XQA-BF16-decode '
            'versus FA2-prefill/XQA-internal-FP8-Q-decode, resident caches, Graph and real scheduler. '
            'All errors and drain retained; no original semantic text or quality acceptance claim.')
    if profile == 'component-diagnostic':
        if family != 'toolagent' or arrival_scaling != .125 or manifest['window_end_ms'] - manifest['window_begin_ms'] != 90000:
            raise ValueError('Component observer requires its frozen continuous90s source window and .125 clock')
        request.update(max_wall_seconds=5400, max_output_gib=3,
            diagnostic={'method':'owned_process_torch_profiler','max_windows':3,'max_forwards_per_window':16,
                        'max_seconds_per_window':6,'max_trace_mib':32,'window_source_seconds':90,
                        'semantic_quality_claim':False,'headline_performance_claim':False})
        request['workload']['kind'] = 'anonymous_trace_component_diagnostic'
        request['research_reason'] = ('Bounded owned-process production profiling of observed low-pressure FP8 latency regression. '
            'Retain original first60s source warm and next30s, sourcepayloads/prefix/OSL, production Graph/scheduler and '
            'modern BF16/FP8 kernels. Three prefill-started CUDA activity windows with explicit forward/time/output bounds. '
            'This is a changed observer diagnostic, not another headline performance replicate or semantic quality test.')
    if research_note:
        request['research_reason'] += ' ' + research_note
    scientific = {k: request[k] for k in ('python_source_commit', 'binary_sha256',
        'dataset_sha256', 'backend_source_sha256', 'image_id', 'graph_mode', 'prefix_cache', 'fp8_scales',
        'operators', 'service', 'concurrency')}
    scientific['workload'] = {k:v for k,v in request['workload'].items() if k != 'trace_manifest'}
    request['experiment_key'] = digest(scientific)
    if retry_failure_audit is not None:
        if request['experiment_key'] != prior_request['experiment_key']:
            raise ValueError('Orchestration retry changed the scientific experiment')
        request['retry_evidence'] = {'prior_run_id': prior_request['run_id'],
            'audit_sha256': _sha(frozen_proof), 'audit_path': remote+'/input/prior_pre_model_conflict_audit.json',
            'statistical_replication': False, 'reason': 'Released pre-model safety conflict; no previous service observations'}
    package = {'schema': 'kv-frozen-local-package-v1', 'classification': 'proposal', 'request': request,
               'gpu_job_launched': False, 'assets': {name: {'local_path': str(path.absolute()), 'sha256': _sha(path)}
                                                    for name,path in files.items()}}
    write_new_json(out/'package.json', package)
    return package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-package', 'data', 'out', 'bundle'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--family', choices=('conversation', 'toolagent', 'synthetic', 'burstgpt'), required=True)
    parser.add_argument('--arrival-scaling', type=float, default=1)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--replicate',type=int,default=0)
    parser.add_argument('--research-note',default='')
    parser.add_argument('--archive-codec',choices=('gzip','xz'),default='gzip')
    parser.add_argument('--profile', choices=('xqa', 'xqa-bf16-reference', 'component-diagnostic'), default='xqa')
    parser.add_argument('--max-wall-seconds', type=int, default=7200)
    parser.add_argument('--retry-failure-audit', type=Path)
    parser.add_argument('--reuse-frozen-runtime', action='store_true')
    args = parser.parse_args()
    result = prepare(args.base_package, args.data.absolute(), args.run_id, args.out.absolute(), args.bundle,
                     args.family, args.arrival_scaling, args.gpu_uuid,args.replicate,args.research_note,args.archive_codec,args.profile,
                     args.max_wall_seconds, args.retry_failure_audit, args.reuse_frozen_runtime)
    print(json.dumps({'package': str(args.out/'package.json'), 'requests': result['request']['workload']['requests'],
                      'gpu_started': False, 'experiment_key': result['request']['experiment_key']}))


if __name__ == '__main__':
    main()
