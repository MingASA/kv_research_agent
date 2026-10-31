"""Independently verify real BF16 observations before emitting candidate scales."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path

from scripts.audit_nextgen_validation import _check, _jsonl, _mirror_paths, audit
from scripts.controller.durable import write_new_json
from scripts.controller.nextgen import MODEL, SERVICE_KIND
from scripts.controller.nextgen_service import OBSERVATION_METHOD


def _snapshot_scales(snapshot: dict) -> dict:
    _check(snapshot.get('method') == OBSERVATION_METHOD and snapshot.get('quality_or_performance_certified') is False,
           'Unknown observation or optimistic quality certificate')
    _check(snapshot['layer_ids'] == list(range(3, 64, 4)) and snapshot['heads'] == 4
           and snapshot['dimension'] == 256 and len(snapshot['rows']) == 256,
           'Range observations do not cover actual hybrid full-attention geometry')
    identities = {(r['layer_id'], r['phase'], r['kind'], r['head']) for r in snapshot['rows']}
    expected = {(layer, phase, kind, head) for layer in range(3, 64, 4)
                for phase in ('prefill', 'decode') for kind in ('key', 'value') for head in range(4)}
    _check(identities == expected, 'Duplicate/missing full-attention phase/head ranges')
    _check(all(r['finite_values'] > 0 and r['finite_values'] % 256 == 0 and r['nonfinite_values'] == 0
               and math.isfinite(r['absmax']) and r['absmax'] >= 0 for r in snapshot['rows']),
           'Empty/nonfinite/incomplete observed K/V write rows')
    _check(len(snapshot['phase_calls_by_layer']) == 16
           and all(len(row) == 2 and min(row) > 0 for row in snapshot['phase_calls_by_layer']),
           'Prefill/decode range kernels did not execute for every full-attention layer')
    maxima = {kind: max(r['absmax'] for r in snapshot['rows'] if r['kind'] == kind) for kind in ('key', 'value')}
    _check(maxima == snapshot['observed_absmax'], 'Global maxima differ from per-layer/head raw ranges')
    candidates = {}
    for fmt, denominator, key in [('fp8_e4m3', 448.0, 'FP8_E4M3_candidate_scales'),
                                  ('nvfp4', 6 * 448.0, 'NVFP4_global_candidate_scales')]:
        values = {kind: max(2**-126, 2.0**math.ceil(math.log2(maximum * 1.1 / denominator)))
                  if maximum else 1.0 for kind, maximum in maxima.items()}
        _check(values == snapshot[key], 'Snapshot scales do not match the frozen independent range rule')
        candidates[fmt] = values
    _check(snapshot['counter_storage_bytes'] == 16 * 2 * 2 * 4 * (4 + 8 + 8) + 16 * 2 * 8,
           'Observer persistent counters differ from the actual dtype/geometry')
    return candidates


def audit_calibration(manifest_path: Path, tokenizer_path: Path) -> dict:
    from transformers import AutoTokenizer

    lifecycle = audit(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    _,root = _mirror_paths(manifest)
    ticket = json.loads((root/'ticket.json').read_text())
    request = ticket['request']
    _check(request['arms'] == ['bf16'] and request.get('range_observation') is True
           and request['workload']['kind'] == SERVICE_KIND and request['model_id'] == MODEL,
           'Scales must come from the actual new-model BF16 diagnostic service')
    output = root/'output/bf16'
    gate = json.loads((output/'observer_Graph.json').read_text())
    _check(gate['actual_cuda_Graph_capture_replay'] and gate['changed_KV_and_padding_checked']
           and gate['quiescent_reset_verified'] and gate['SM'] == 90,
           'Range observer did not pass the real changed-input SM90 Graph gate')
    gate_rows = gate['snapshot']['rows']
    _check(len(gate_rows) == 32 and gate['snapshot']['layer_ids'] == [3, 7]
           and gate['snapshot']['phase_calls_by_layer'] == [[0, 2], [0, 2]],
           'Range Graph gate omitted original layer/head counters')
    _check(all(r['finite_values'] == (1280 if r['phase'] == 'decode' else 0)
               and r['nonfinite_values'] == 0
               and r['absmax'] == ((7.0 if r['kind'] == 'key' else 11.0) if r['phase'] == 'decode' else 0.0)
               for r in gate_rows), 'GPU range gate raw maxima/counters differ from changed valid input rows')
    raw = (root/'input/calibration.json').read_bytes()
    _check(hashlib.sha256(raw).hexdigest() == request['calibration_sha256'], 'Calibration input SHA changed')
    inputs = json.loads(raw)['requests']
    rows = _jsonl(output/'calibration_raw.jsonl')
    _check(len(inputs) == len(rows) == request['calibration_count'] == 136
           and len({r['id'] for r in rows}) == 136, 'Missing/duplicate actual calibration responses')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    capped = 0
    completion_tokens = 0
    for expected_row, row in zip(inputs, rows):
        _check(row['id'] == expected_row['id'] and row['request'] == expected_row['payload']
               and row.get('error') is None, 'Calibration order/payload/client contract changed')
        raw_response = base64.b64decode(row['raw_HTTP_body_base64'], validate=True)
        response = json.loads(raw_response)
        _check(response == row['response'] and not response.get('error'), 'Cached response differs from raw HTTP body')
        tokens = tokenizer.apply_chat_template(row['request']['messages'], tokenize=True, add_generation_prompt=True,
                                               enable_thinking=False, return_dict=True)['input_ids']
        usage = response['usage']
        _check(usage['prompt_tokens'] == expected_row['prompt_tokens_cpu'] == len(tokens)
               and 0 < usage['completion_tokens'] <= row['request']['max_tokens']
               and usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'],
               'Actual new-model tokenizer/request usage differs')
        _check(len(response['choices']) == 1 and response['choices'][0]['message']['content']
               and response['choices'][0]['finish_reason'] in ('stop', 'length'),
               'Incomplete calibration output')
        capped += response['choices'][0]['finish_reason'] == 'length'
        completion_tokens += usage['completion_tokens']
    snapshot_path = output/'kv_observer_snapshot.json'
    snapshot = json.loads(snapshot_path.read_text())
    for action in ('reset', 'export'):
        command = json.loads((output/('kv_observer_'+action+'.json')).read_text())
        ack = json.loads((output/('kv_observer_'+action+'_ack.json')).read_text())
        _check(command == {'action': action, 'request_sha256': ticket['request_sha256']}
               and ack['action'] == action and ack['request_sha256'] == ticket['request_sha256'],
               'Quiescent observation boundary differs')
        if action == 'export':
            _check(ack['snapshot_sha256'] == hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
                   'Export acknowledgement/snapshot SHA mismatch')
    _check(snapshot['request_sha256'] == ticket['request_sha256'] and snapshot['run_id'] == request['run_id'],
           'Observation snapshot belongs to another run')
    scales = _snapshot_scales(snapshot)
    _check(lifecycle['native_backend']['KV_observation_enabled'] is True
           and lifecycle['observer_counter_resident_bytes'] == snapshot['counter_storage_bytes'],
           'Actual model did not retain the observed GPU range counters')
    return {'state': 'BF16_CALIBRATION_INDEPENDENTLY_VERIFIED', 'model_id': MODEL, 'run_id': request['run_id'],
            'terminal_result_sha256': hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),
            'calibration_request_sha256': ticket['request_sha256'],
            'model_config_sha256': request['model_config_sha256'],
            'server_source_commit': request['server_source_commit'],
            'python_source_commit': request['python_source_commit'],
            'lifecycle_audit': lifecycle, 'actual_observer_GPU_Graph_passed': True,
            'calibration_requests_verified': len(rows), 'actual_completion_tokens': completion_tokens,
            'capped_calibration_responses': capped, 'snapshot_sha256': hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
            'observed_absmax': snapshot['observed_absmax'], 'candidate_scales': scales,
            'quality_certified': False, 'performance_certified': False,
            'scope': 'Observed BF16 post-RoPE K/projected V write ranges only; held-out real chat and long documents. '
                     'Scales are candidates, not guarantees for unseen requests or accepted model quality.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sha-manifest', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    write_new_json(args.out, audit_calibration(args.sha_manifest, args.tokenizer))


if __name__ == '__main__':
    main()
