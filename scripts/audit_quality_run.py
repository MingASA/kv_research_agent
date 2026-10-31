#!/usr/bin/env python3
"""Validate sealed model predictions and independently rescore native FA3 quality."""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.controller.durable import digest
from scripts.controller.production import verify_backend, _jsonl_rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(mirror: Path, run_id: str, source_repo: Path, tokenizer_path: Path,
          selected_arm: str | None = None, recover_conflict: bool = False) -> dict:
    sys.path.insert(0, str(source_repo))
    from tools import eval_kv_cache_quality as evaluator
    from transformers import AutoTokenizer
    if Path(evaluator.__file__).resolve() != (source_repo / 'tools/eval_kv_cache_quality.py').resolve():
        raise ValueError('Independent rescoring imported an unexpected evaluator')

    snapshot = json.loads((mirror / 'SNAPSHOT.json').read_text())
    for row in snapshot['files']:
        file = mirror / row['relative_path']
        if file.is_symlink() or file.stat().st_size != row['size'] or _sha(file) != row['sha256']:
            raise ValueError('Quality mirror hash/inventory drift')
    root = mirror / 'support/agent_work/runs' / run_id
    ticket = json.loads((root / 'ticket.json').read_text())
    request = ticket['request']
    for name,expected in ticket['assets_sha256'].items():
        if _sha(root/name)!=expected:raise ValueError('Frozen quality asset differs: '+name)
    for name,expected in request['backend_source_sha256'].items():
        if _sha(source_repo/name)!=expected:raise ValueError('Independent quality/runtime source differs: '+name)
    full_reference = request['workload']['kind'] == 'native_quality_xqa_bf16_reference'
    full_batch = request['workload']['kind'] == 'native_quality_xqa_production_batch'
    scale_coverage = request['workload'].get('scale_calibration') is not None
    if scale_coverage:
        from scripts.controller.scale_coverage import validate_prerequisites
        validate_prerequisites(request, input_root=root/'input')
    bf16q_candidate = request['workload']['kind'] == 'native_quality_bf16q_production_batch'
    if bf16q_candidate and (request['arms']!=['fp8_e4m3'] or request['concurrency']!=128
            or request['workload']['cohort']!='full_production_batch' or request['workload']['samples_per_arm']!=1719):
        raise ValueError('Full BF16-Q candidate requires the exact single-FP8 C128 original1719 scope')
    if full_reference and (request['arms'] != ['bf16'] or request['workload']['cohort'] != 'full_reference'
                           or request['workload']['samples_per_arm'] != 1719):
        raise ValueError('Full BF16 reference must preserve its explicit single-arm1719 scope')
    if full_batch and not scale_coverage and (len(request['arms']) != 2 or set(request['arms']) != {'bf16', 'fp8_e4m3'}
                       or request['concurrency'] != 128 or request['workload']['cohort'] != 'full_production_batch'
                       or request['workload']['samples_per_arm'] != 1719):
        raise ValueError('Full production-batch quality requires the explicit C128 paired1719 scope')
    result = json.loads((root / 'result.json').read_text())
    recovering = recover_conflict and selected_arm in request['arms'] and result['status'] == 'CONFLICT'
    if recover_conflict and not recovering:
        raise ValueError('Conflict recovery requires one explicit known arm and the original CONFLICT terminal')
    if (digest(request) != ticket['request_sha256'] or result['request_sha256'] != ticket['request_sha256']
            or not recovering and result['status'] != 'COMPLETE' or result['gpu_release_verified'] is not True
            or result['live_owned_containers']):
        raise ValueError('Quality run has no matching real completion/release')
    if not recovering and json.loads((root / 'COMPLETE').read_text())['result_sha256'] != _sha(root / 'result.json'):
        raise ValueError('Quality root completion seal differs')
    if recovering and (root / 'COMPLETE').exists():
        raise ValueError('Conflict terminal must not be rewritten as COMPLETE')
    arms = [selected_arm] if selected_arm is not None else request['arms']
    if any(arm not in request['arms'] for arm in arms):
        raise ValueError('Unselected quality arm')
    source = root / Path(request['dataset_path']).relative_to(request['remote_output'])
    if _sha(source) != request['dataset_sha256']:
        raise ValueError('Frozen quality cohort changed')
    cohort = json.loads(source.read_text())
    index_map = cohort.get('selected_source_indices', cohort.get('source_indices'))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    formats, caps, scorer, _ = evaluator._longbench_assets(root / 'input/scorer')
    shots = json.loads((root / 'input/five_train_examples.json').read_text())
    manifests, outputs, records, caps_count = {}, {}, {}, {}
    finished = []
    measurement_hashes = {}
    for arm in arms:
        directory = root / 'output' / arm
        measurement = json.loads((directory / 'measurement.json').read_text())
        expected = (_sha(directory / 'measurement.json') if recovering else
                    next(row['result_sha256'] for row in result['arm_results'] if row['arm'] == arm))
        if (_sha(directory / 'measurement.json') != expected or measurement['measurement_success'] is not True
                or measurement['request_sha256'] != ticket['request_sha256']):
            raise ValueError('Supervisor/worker measurement seal differs')
        measurement_hashes[arm] = expected
        for name, expected_sha in measurement['evidence_sha256'].items():
            file = directory / name
            if not file.resolve().is_relative_to(directory.resolve()) or _sha(file) != expected_sha:
                raise ValueError('Sealed quality evidence drift: ' + name)
        quality = directory / 'quality'
        manifest = json.loads((quality / 'experiment_manifest.json').read_text())
        manifests[arm] = manifest
        label = 'auto' if arm == 'bf16' else arm.replace('_', '-')
        native_arm = 'auto' if arm == 'bf16' else arm
        attempt = quality / 'arms' / native_arm / 'attempt-0001'
        verify_backend((attempt / 'server.log').read_text(), arm, request['operators'][arm], request['fp8_scales'])
        launch = json.loads((attempt / 'launch.json').read_text())
        if '--enable_prefix_cache=true' not in launch['command'] or '--python_graph_backend=cudagraphs' not in launch['command']:
            raise ValueError('Quality did not use production prefix/Graph configuration')
        outputs[arm], caps_count[arm] = {}, {}
        for task, data in manifest['data_audit']['datasets'].items():
            file = root / Path(data['path']).relative_to(request['remote_output'])
            if _sha(file) != data['sha256']:
                raise ValueError('Original quality input hash differs')
            rows = list(_jsonl_rows(file))
            chat = task not in ('lcc', 'repobench-p')
            prompts = [evaluator._gsm_prompt(row['question'], shots) if task == 'gsm8k'
                       else formats[task].format(context=row['context'], input=row['input']) for row in rows]
            counts = [evaluator._token_count(tokenizer, prompt, chat) for prompt in prompts]
            if counts != manifest['prepared'][task]['prompt_tokens']:
                raise ValueError('Independent tokenizer and real H800 preparation differ')
            journal = quality / 'runs' / task / (label + '.jsonl')
            indices = evaluator._validate_resume_prefix(journal, rows, prompts, counts, chat)
            if indices != set(range(len(rows))):
                raise ValueError('Missing or repeated actual model predictions')
            predicted = sorted(_jsonl_rows(journal), key=lambda row: row['_eval_index'])
            for row in predicted:
                choice = row['api_response']['choices'][0]
                if choice['finish_reason'] not in ('stop', 'length') or row['api_response']['usage']['completion_tokens'] < 1:
                    raise ValueError('Quality API prediction did not complete normally')
                finished.append(datetime.datetime.fromisoformat(row['request_finished_utc']))
            outputs[arm][task] = [row['prediction'] for row in predicted]
            caps_count[arm][task] = sum(row['api_response']['choices'][0]['finish_reason'] == 'length' for row in predicted)
            if task in records and records[task] != rows:
                raise ValueError('Two quality arms do not pair on the same original records')
            records[task] = rows
    baseline = next(iter(manifests.values()))
    if full_reference or full_batch or bf16q_candidate:
        expected_indices = {task: list(range(1319 if task == 'gsm8k' else 50)) for task in records}
        if (set(records) != {'gsm8k', 'narrativeqa', 'qasper', 'multifieldqa_en', '2wikimqa',
                             'hotpotqa', 'musique', 'lcc', 'repobench-p'} or index_map != expected_indices
                or any(len(records[task]) != len(indices) for task, indices in expected_indices.items())
                or baseline['plan']['scope'] != 'one_unique_dataset_round'):
            raise ValueError('Full reference omitted, duplicated or changed original global records')
    for other in manifests.values():
        for key in ('model_provenance', 'source_sha256', 'binary_sha256', 'versions'):
            if baseline[key] != other[key]:
                raise ValueError('Paired quality provenance differs: ' + key)
    conflict_after = None
    if recovering:
        events = list(_jsonl_rows(root/'events.jsonl'))
        conflicts = [row for row in events if row.get('action') == 'STOP_OR_FAILURE' and row.get('status') == 'CONFLICT']
        if not conflicts or not finished:
            raise ValueError('No precise conflict/prediction timeline for recovery')
        conflict_after = (datetime.datetime.fromisoformat(conflicts[0]['utc'])-max(finished)).total_seconds()
        if conflict_after <= 0:
            raise ValueError('Conflict overlapped actual quality prediction requests; cannot recover clean quality')
    scores, items, flips = {}, {}, {}
    for task, rows in records.items():
        scoring_order = [arm for arm in ('bf16', 'fp8_e4m3') if arm in arms]
        scores[task], items[task] = evaluator._score_pair(task, rows, {arm: outputs[arm][task] for arm in scoring_order}, scorer)
        for row in items[task]:
            row['original_source_index'] = index_map[task][row['index']]
        flips[task] = dict(collections.Counter(row['flexible_flip'] if task == 'gsm8k' else row['flip'] for row in items[task]))
    macro = {arm: sum(scores[task][arm] for task in records if task != 'gsm8k') / 8 for arm in arms}
    fragment = request['workload']['cohort'] == 'completion_fragment'
    return {'schema': 'kv-quality-local-audit-v1', 'run_id': run_id,
            'classification': ('confirmed_full_scale_coverage_candidate_prediction_integrity_and_rescoring' if scale_coverage else
                               'confirmed_full_BF16Q_candidate_prediction_integrity_and_rescoring' if bf16q_candidate else
                               'confirmed_full_production_batch_prediction_integrity_and_rescoring' if full_batch else
                               'confirmed_full_bf16_reference_prediction_integrity_and_rescoring' if full_reference else
                               'confirmed_recovered_fragment_arm_prediction_integrity_and_rescoring' if recovering else
                               'confirmed_fragment_arm_prediction_integrity_and_rescoring' if fragment and len(arms)==1 else
                               'confirmed_fragment_prediction_integrity_and_rescoring' if fragment else
                               'confirmed_pilot_arm_prediction_integrity_and_rescoring' if len(arms)==1 else
                               'confirmed_pilot_prediction_integrity_and_rescoring'),
            'terminal_result_sha256': _sha(root / 'result.json'), 'root_complete': not recovering,
            'terminal_status_preserved': result['status'], 'conflict_seconds_after_last_prediction': conflict_after,
            'audited_arms': arms, 'arm_measurement_sha256': measurement_hashes, 'gpu_release_verified': True,
            'samples_per_arm': sum(map(len, records.values())), 'task_scores': scores, 'longbench_8_task_macro': macro,
            'paired_flips': flips, 'paired_items': items, 'length_finish_counts': caps_count,
            'model_weights_sha256': baseline['model_provenance']['weights']['sha256'],
            'runtime_sources_digest': digest(baseline['source_sha256']), 'versions': baseline['versions'],
            'model_metadata_sha256': baseline['model_provenance']['metadata_sha256'],
            'evaluator_sha256': baseline['runner_source_sha256']['tools/eval_kv_cache_quality.py'],
            'production_configuration': {key: request[key] for key in ('binary_sha256', 'model_config_sha256', 'image_id',
                'operators', 'service', 'graph_mode', 'fp8_scales', 'concurrency')},
            'protocol_files_sha256': baseline['data_audit']['protocol_files'],
            'actual_gpu_hours': sum((i['end_ns'] - i['start_ns']) / 3.6e12 for i in result['allocation_intervals_ns']),
            'quality_acceptance': ('requires_comparison_to_retained_modern_BF16_and_quality_acceptance' if bf16q_candidate or scale_coverage else
                                   'full_production_batch_scores_require_user_quality_acceptance' if full_batch else
                                   'reference_only_requires_comparison_to_existing_FP8_predictions' if full_reference else
                                   'requires_verified_full_cohort_merge' if fragment else 'not_established_by_small_stratified_pilot'),
            'full_1719_evaluation': full_reference or full_batch or bf16q_candidate, 'reuse_mapping': index_map,
            'audited_source_commit': request['python_source_commit'],
            'audited_source_files_sha256': baseline['source_sha256'],
            **({'scale_calibration':request['workload']['scale_calibration'],
                'scale_quality_provenance_sha256':_sha(root/'input/scale_quality_provenance.json')}
               if scale_coverage else {}),
            'limitations': ['Full1719 preregistered V-scale coverage candidate only; retained modern reference is a different run; no single-cause or service benefit claim' if scale_coverage else
                           'Full1719 BF16-Q/FP8-KV C128 candidate only; retained BF16 reference is a different run, no Q-causality or service claim' if bf16q_candidate else
                           'Full1719 paired C128 deployment validation; client limit is not a fixed actual GPU batch size' if full_batch else
                           'Full1719 BF16-XQA reference only; not a new paired FP8 run or production acceptance' if full_reference else
                           'Completion excludes the verified pilot80 and is not a complete1719 cohort' if fragment
                           else 'GSM64 is stratified; LongBench has only shortest+longest per task',
                           'Code completion length caps are the frozen benchmark protocol, not a separate model-failure verdict',
                           'Quality scores apply to the exact declared production prefill/decode/Q/K/V/scales configuration']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mirror', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--source-repo', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices=('bf16', 'fp8_e4m3'))
    parser.add_argument('--recover-conflict', action='store_true')
    args = parser.parse_args()
    result = audit(args.mirror, args.run_id, args.source_repo, args.tokenizer, args.arm, args.recover_conflict)
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('run_id', 'samples_per_arm', 'task_scores', 'longbench_8_task_macro', 'quality_acceptance')}))


if __name__ == '__main__':
    main()
