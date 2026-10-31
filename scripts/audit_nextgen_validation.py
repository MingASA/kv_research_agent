"""Independently audit mirrored native BF16 service correctness evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from scripts.controller.durable import canonical_bytes, write_new_json


def _check(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict]:
    """Our journals use LF; U+2028 inside valid JSON strings is ordinary data."""
    raw = path.read_bytes()
    _check(not raw or raw.endswith(b'\n'),'Incomplete response journal without final durable LF')
    return [json.loads(line) for line in raw.split(b'\n') if line]


def _mirror_paths(manifest: dict) -> tuple[Path, Path]:
    directory = Path(manifest['mirror_directory'])
    layout = manifest.get('layout','support')
    _check(layout in ('support','run_relative','direct_run'),'Unknown explicit mirror layout')
    base = directory if layout=='direct_run' else directory/('files/support' if layout=='support' else 'files')
    root = base/'agent_work/runs'/manifest['run_id'] if layout=='support' else base
    _check(base.is_dir() and root.is_dir(),'Missing explicit mirrored run directory')
    return base,root


def _stream(row: dict, case: dict, prompt_tokens: int, model: str) -> str:
    expected_request = {'model': model, 'messages': case['messages'], 'temperature': 0, 'max_tokens': 96,
                        'stream': True, 'stream_options': {'include_usage': True},
                        'chat_template_kwargs': {'enable_thinking': False}}
    _check(row['request'] == expected_request and row['error'] is None, 'Input changed or client failure')
    answer, usage, finish, done = [], None, None, 0
    for line in row['raw_SSE']:
        if not line.startswith('data: '):
            continue
        data = line[6:]
        if data == '[DONE]':
            done += 1
            continue
        chunk = json.loads(data)
        if chunk.get('usage') is not None:
            usage = chunk['usage']
        for choice in chunk.get('choices', []):
            answer.append(choice.get('delta', {}).get('content', ''))
            finish = choice.get('finish_reason') or finish
    text = ''.join(answer)
    _check(done == 1 and finish == 'stop' and text == row['answer'] and bool(text), 'Incomplete/mismatched raw SSE')
    _check(usage == row['usage'] and usage['prompt_tokens'] == prompt_tokens
           and 0 < usage['completion_tokens'] < 96
           and usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'], 'Raw token usage differs')
    _check(case['expected'] in text, 'Independent known-answer check failed')
    return text


def audit(manifest_path: Path, *, failed_quality_recovery: dict | None = None,
          interrupted_quality_fragment: bool = False) -> dict:
    manifest = _read(manifest_path)
    mirror,root = _mirror_paths(manifest)
    for row in manifest['files']:
        data = (mirror/row['base_rel']).read_bytes()
        _check(len(data) == row['size'] and hashlib.sha256(data).hexdigest() == row['remote_sha256'] == row['snapshot_sha256'],
               'Independent mirrored-file SHA mismatch')
    ticket, result = _read(root/'ticket.json'), _read(root/'result.json')
    request = ticket['request']
    _check(hashlib.sha256(canonical_bytes(request)).hexdigest() == ticket['request_sha256'] == result['request_sha256'],
           'Immutable request identity differs')
    _check(result['gpu_release_verified'] is True and not result['live_owned_containers'],
           'Original run did not release its GPU')
    _check(len(request['arms']) == 1 and request['arms'][0] in ('bf16', 'fp8_e4m3', 'nvfp4'),
           'Validation audit requires one actual model service arm')
    arm = request['arms'][0]
    output = root/'output'/arm
    if interrupted_quality_fragment:
        _check(failed_quality_recovery is None and result['status'] in ('FAILED','CONFLICT')
               and request['workload']['kind']=='native_hybrid_core_quality'
               and result.get('arm_results')==[], 'Not an interrupted unscored quality generation')
        measurement=_read(output/'validation_result.json')
        _check(not (root/'COMPLETE').exists(),'Do not relabel a completed full run as interrupted')
    elif failed_quality_recovery is None:
        _check(result['status']=='COMPLETE','Original run did not complete')
        measurement = _read(output/'measurement.json')
        _check(hashlib.sha256((output/'measurement.json').read_bytes()).hexdigest() == result['arm_results'][0]['result_sha256'],
               'Worker result hash is not bound to terminal run')
        for name, sha in measurement['evidence_sha256'].items():
            _check(hashlib.sha256((output/name).read_bytes()).hexdigest() == sha, 'Worker evidence changed: '+name)
    else:
        recovery=failed_quality_recovery
        _check(result['status']=='FAILED' and request['workload']['kind']=='native_hybrid_core_quality'
               and recovery.get('state')=='INDEPENDENT_RAW_QUALITY_RECOVERY_ORIGINAL_RUN_FAILED'
               and recovery.get('run_id')==manifest['run_id']
               and recovery.get('source_manifest_sha256')==hashlib.sha256(manifest_path.read_bytes()).hexdigest()
               and recovery.get('original_result_sha256')==hashlib.sha256((root/'result.json').read_bytes()).hexdigest()
               and recovery.get('prediction_journal_sha256')==hashlib.sha256((output/'quality_predictions.jsonl').read_bytes()).hexdigest()
               and recovery.get('all_raw_API_payloads_and_official_JSON_usage_verified') is True,
               'Failed quality lifecycle lacks an explicit bound independent raw recovery certificate')
        _check('Incomplete/invalid core quality responses; original bytes retained, no accuracy acceptance'
               in (output/'container.log').read_text(),'Original failure does not match recoverable client contract validation')
        for name,sha in ticket['assets_sha256'].items():
            _check(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,'Frozen failed-run input/runtime asset changed')
        measurement=_read(output/'validation_result.json')
    dataset = _read(root/'input/validation.json')
    _check(hashlib.sha256((root/'input/validation.json').read_bytes()).hexdigest() == request['dataset_sha256'],
           'Frozen validation input differs')
    cases = dataset['cases']
    rows = _jsonl(output/'validation_raw.jsonl')
    expected = cases[:4] + cases[4:] + cases[:4]
    _check(len(cases) == 16 and len(rows) == len(expected) == 20, 'Incomplete service validation')
    counts = {r['id']: r['prompt_tokens'] for r in _read(root/'input/validation_tokenization_CPU.json')['cases']}
    answers = []
    for row, case in zip(rows, expected):
        _check(row['id'] == case['id'], 'Validation order/identity changed')
        answers.append(_stream(row, case, counts[case['id']], request['model_id']))
    _check(answers[:4] == answers[-4:], 'A/B/A recurrent/cache lifecycle repeat differs')
    text = (output/'server.log').read_text()
    _check('NATIVE_KV_GRAPH_REPLAY' in text, 'Actual model Graph replay log missing')
    backend_rows = [json.loads(line.split('NATIVE_KV_BACKEND ', 1)[1]) for line in text.splitlines()
                    if 'NATIVE_KV_BACKEND ' in line]
    allocation_rows = [json.loads(line.split('KV_ALLOCATION ', 1)[1]) for line in text.splitlines()
                       if 'KV_ALLOCATION ' in line]
    _check(bool(backend_rows) and bool(allocation_rows), 'Missing raw native backend/allocation marker')
    backend, allocation = backend_rows[-1], allocation_rows[-1]
    _check(backend == measurement['native_backend'] and allocation == measurement['native_allocation'],
           'Optimistic cached backend/allocation differs from raw server evidence')
    profile = request['operators'][arm]
    classes = {'bf16': 'FlashInferBackend', 'fp8_e4m3': 'FlashInferFP8Backend', 'nvfp4': 'FlashInferNVFP4Backend'}
    dtypes = {'bf16': 'torch.bfloat16', 'fp8_e4m3': 'torch.float8_e4m3fn', 'nvfp4': 'torch.uint8'}
    _check(backend['class'] == classes[arm] and backend['planner_backend'] == profile['planner']
           and backend['decode_impl'] == profile['decode'] and backend['q_data_type'] == 'torch.bfloat16'
           and backend['kv_data_type'] == dtypes[arm] and backend['graph_enabled'] is True
           and backend['graph_captured_batches'] == [1<<index for index in range(request['service']['max_batch'].bit_length())],
           'Native format/kernels/Graph backend differs')
    if arm != 'bf16':
        _check(backend.get('full_BF16_KV_shadow') is False, 'Low-precision backend has no no-shadow attestation')
        alias = 'fp8_views_alias_resident_pages' if arm == 'fp8_e4m3' else 'scale_views_alias_resident_pages'
        _check(backend.get(alias) is True, 'Native attention does not alias resident low-precision storage')
    _check(allocation['full_attention_layer_ids'] == list(range(3, 64, 4)), 'Full/GDN layer binding differs')
    linear_ids = [r['layer_id'] for r in allocation['linear_state_layers']]
    _check(len(linear_ids) == 48 and sorted(linear_ids + allocation['full_attention_layer_ids']) == list(range(64)),
           'Missing/overlapping recurrent cache layers')
    sizes = {'torch.bfloat16': 2, 'torch.float32': 4, 'torch.uint8': 1, 'torch.int8': 1,
             'torch.float8_e4m3fn': 1}
    totals = []
    for layers in (allocation['layers'], allocation['linear_state_layers']):
        total = 0
        for layer in layers:
            actual = sum(math.prod(t['shape']) * sizes[t['dtype']] for t in layer['tensors'].values())
            _check(actual == layer['resident_bytes'] == sum(t['bytes'] for t in layer['tensors'].values()),
                   'Resident tensor storage arithmetic differs')
            total += actual
        totals.append(total)
    _check(totals == [allocation['full_attention_resident_bytes'], allocation['linear_state_resident_bytes']]
           and sum(totals) == allocation['total_resident_bytes'], 'Full/GDN byte accounting differs')
    events = _jsonl(root/'events.jsonl')
    _check(not any(r['action'] == 'GPU_CONFLICT_OBSERVATION' for r in events), 'Foreign GPU conflict observed')
    teardown = []
    for event in events:
        if event['action'] != 'OWNED_GPU_PID_EXIT_RECHECK':
            continue
        prior, current = event['retired_owned_pid'], event.get('process', event['retired_owned_pid'])
        _check(all(prior[k] == current[k] for k in ('pid', 'start_ticks', 'uid', 'boot_id'))
               and any(c in prior['cgroup'] for c in event['owned_containers']), 'Retired GPU PID lacks exact owned identity')
        if 'process_lifecycle' in event and event['process_lifecycle'] is not None:
            life = event['process_lifecycle']
            _check(life['state'] == 'Z' or bool(life['kernel_flags'] & 4), 'No positive kernel exit evidence')
        _check(all(prior['pid'] not in row['pids'] for row in event['fresh_selected'].values()),
               'Owned retiring PID did not leave driver on fresh read')
        teardown.append({'pid': prior['pid'], 'start_ticks': prior['start_ticks'], 'owned_identity_and_exit_verified': True})
    waits = [e for e in events if e['action'] == 'WAIT_GPU' and e.get('stable_since_ns') is not None]
    _check(len(waits) >= 2 and (datetime.fromisoformat(waits[-1]['utc']) - datetime.fromisoformat(waits[0]['utc'])).total_seconds() >= 120,
           'Full stable idle interval missing')
    _check(all(e['reservation_policy_clear'] and all(not r['pids'] and r['memory_mib'] == 0 for r in e['selected'].values())
               for e in waits), 'Stable idle/resource exclusion gate differs')
    return {'schema': 'kv-nextgen-independent-service-validation-audit-v1', 'run_id': manifest['run_id'],
            'classification': 'confirmed_' + ('BF16' if arm == 'bf16' else arm)
                              + '_text_correctness_and_cache_lifecycle_not_benchmark_quality',
            'arm': arm, 'native_backend': backend,
            'files_SHA_verified': len(manifest['files']), 'mirror_bytes_verified': sum(r['size'] for r in manifest['files']),
            'real_SSE_requests_verified': 20, 'known_answer_checks_passed': 20, 'A_B_A_exact_repeats': 4,
            'actual_prompt_tokens_match_new_tokenizer': True, 'max_actual_prompt_tokens': max(counts.values()),
            'actual_completion_tokens': sum(r['usage']['completion_tokens'] for r in rows),
            'actual_model_Graph_replay_logged': True, 'Graph_captured_batches': backend['graph_captured_batches'],
            'Full_attention_layers': 16, 'GDN_layers': 48, 'Full_KV_resident_bytes': totals[0], 'GDN_state_resident_bytes': totals[1],
            'total_cache_resident_bytes': sum(totals), 'Full_KV_physical_token_slots': allocation['token_capacity'],
            'global_scale_resident_bytes': backend.get('global_scale_resident_bytes', 0),
            'observer_counter_resident_bytes': backend.get('KV_observation_counter_storage_bytes', 0),
            'max_successful_token_capacity_measured': False, 'teardown_identity_events_verified': teardown,
            'gpu_release_verified': True, 'model_quality_benchmark_validated': False, 'AIPerf_or_service_performance_measured': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sha-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.sha_manifest)
    write_new_json(args.out, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
