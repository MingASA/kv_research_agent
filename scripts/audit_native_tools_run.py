#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Independently verify native-tool serving and original BFCL argument matches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bfcl_evaluator import load_official_checker, score_native_tool_calls
from scripts.controller.durable import digest, write_new_json
from scripts.controller.evidence import verify_allocation, _percentile
from scripts.controller.native_tools import audit_native_tools
from scripts.controller.production import verify_backend
from scripts.prepare_tool_calls import PREFIX, SOURCE_COMMIT


def _sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit(mirror: Path, run_id: str, source_root: Path, prepared: Path, source_repo: Path) -> dict:
    snapshot = json.loads((mirror / 'SNAPSHOT.json').read_text())
    for row in snapshot['files']:
        path = mirror / row['relative_path']
        if path.is_symlink() or path.stat().st_size != row['size'] or _sha(path) != row['sha256']:
            raise ValueError('Original mirrored evidence changed')
    root = mirror / 'support/agent_work/runs' / run_id
    ticket = json.loads((root / 'ticket.json').read_text()); request = ticket['request']
    terminal = json.loads((root / 'result.json').read_text())
    if (digest(request) != ticket['request_sha256'] or terminal['request_sha256'] != ticket['request_sha256']
            or terminal['run_id'] != run_id or terminal['status'] != 'COMPLETE'
            or terminal['gpu_release_verified'] is not True or terminal['live_owned_containers']
            or json.loads((root / 'COMPLETE').read_text())['result_sha256'] != _sha(root / 'result.json')):
        raise ValueError('Require actual clean terminal seal and driver release')
    for name, expected in request['backend_source_sha256'].items():
        if _sha(source_repo / name) != expected:
            raise ValueError('Independent local source checkout differs from actual pinned inference/API source')
    if (_sha(root / 'input/canonical_inputs.json') != request['dataset_sha256']
            or _sha(prepared / 'tool_inputs.json') != request['dataset_sha256']
            or _sha(root / 'input/production_manifest.json') != _sha(prepared / 'manifest.json')):
        raise ValueError('Native input differs from the pre-answer frozen cohort')
    manifest = json.loads((prepared / 'manifest.json').read_text())
    reference = json.loads((source_root / 'post_freeze_answer_manifest.json').read_text())
    if (reference['commit'] != SOURCE_COMMIT or reference['pre_gold_dataset_sha256'] != manifest.get('parent_source_dataset_sha256', request['dataset_sha256'])
            or reference['answers_fetched_only_after_prompt_freeze'] is not True):
        raise ValueError('Reference answers do not bind the immutable pre-answer cohort')
    answer_entry = reference['files'][0]; answer_path = source_root / answer_entry['path']
    if _sha(answer_path) != answer_entry['sha256']:
        raise ValueError('Reference answer bytes changed')
    answers = {row['id']: row['ground_truth'] for row in map(json.loads, answer_path.read_text().splitlines())}
    task_path = source_root / (PREFIX + 'data/BFCL_v4_live_simple.json')
    if _sha(task_path) != request['workload']['source_content_sha256']:
        raise ValueError('Original public tasks differ')
    tasks = {row['id']: row for row in map(json.loads, task_path.read_text().splitlines())}
    records = {row['session_id']: row for row in manifest['records'] if row['phase'] == 'measurement'}
    if len(records) != 258 or {row['source_id'] for row in records.values()} != set(tasks) or set(tasks) != set(answers):
        raise ValueError('Official full task denominator differs')
    checker, checker_provenance = load_official_checker(source_root)
    arms = {}
    for arm in request['arms']:
        output = root / 'output' / arm
        measurement = json.loads((output / 'measurement.json').read_text())
        if (measurement.get('measurement_success') is not True or measurement.get('official_aiperf_complete') is not True
                or measurement['arm'] != arm or measurement['request_sha256'] != ticket['request_sha256']
                or next(row['result_sha256'] for row in terminal['arm_results'] if row['arm'] == arm) != _sha(output / 'measurement.json')):
            raise ValueError('Native-tool worker measurement seal differs')
        for name, expected in measurement['evidence_sha256'].items():
            path = output / name
            if not path.resolve().is_relative_to(output.resolve()) or _sha(path) != expected:
                raise ValueError('Sealed original native-tool evidence differs')
        backend = verify_backend((output / 'server.log').read_text(), arm, request['operators'][arm])
        allocation = verify_allocation(json.loads((output / 'allocation.json').read_text()), arm)
        metrics = audit_native_tools(output / 'artifacts', root / 'input/canonical_inputs.json', root / 'input/production_manifest.json')
        # AIPerf's process RC is a worker report; raw completion, attempts and
        # all request outcomes are independently recomputed here.
        metrics['aiperf_exit_code'] = measurement['metrics']['aiperf_exit_code']
        if metrics != measurement['metrics'] or metrics['aiperf_exit_code'] not in (0, 1):
            raise ValueError('Independent full raw native-tool outcomes differ from worker seal')
        scored = {}
        for row in metrics['requests']:
            if row['phase'] != 'profiling':
                continue
            record = records[row['session_id']]; identity = record['source_id']
            score = (score_native_tool_calls(checker, tasks[identity]['function'], row['tool_calls'], answers[identity])
                     if row['service_success'] else {'task_correct': False, 'native_contract_valid': False,
                                                     'error_type': 'recorded_service_failure'})
            scored[identity] = {'session_id': row['session_id'], 'service_success': row['service_success'], **score}
        correct = sum(row['task_correct'] for row in scored.values())
        successful = [row for row in metrics['requests'] if row['phase'] == 'profiling' and row['service_success']]
        latency = {field: {name: _percentile([row[field] for row in successful if row[field] is not None], q)
                          if any(row[field] is not None for row in successful) else None
                    for name, q in (('p50', .5), ('p95', .95), ('p99', .99))}
                   for field in ('ttft_ms', 'content_request_latency_ms', 'complete_request_ms')}
        arms[arm] = {'backend': backend, 'allocation': allocation, 'metrics': metrics, 'scored': scored,
            'official_argument_matches': correct, 'full258_argument_match_accuracy': correct / 258,
            'official_argument_match_rps': correct / metrics['observation_seconds'], 'latency': latency,
            'measurement_sha256': _sha(output / 'measurement.json')}
    baseline, candidate = arms['bf16'], arms['fp8_e4m3']
    flips = {'both_correct': 0, 'both_incorrect': 0, 'BF16_only_correct': 0, 'FP8_only_correct': 0}
    for identity in tasks:
        left, right = baseline['scored'][identity]['task_correct'], candidate['scored'][identity]['task_correct']
        flips['both_correct' if left and right else 'BF16_only_correct' if left else
              'FP8_only_correct' if right else 'both_incorrect'] += 1
    return {'classification': 'confirmed_native_tool_API_raw_integrity_and_official_argument_matching',
        'run_id': run_id, 'terminal_result_sha256': _sha(root / 'result.json'), 'gpu_release_verified': True,
        'actual_gpu_hours': sum((r['end_ns'] - r['start_ns']) / 3.6e12 for r in terminal['allocation_intervals_ns']),
        'arms': arms, 'paired_task_outcomes': flips, 'official_checker': checker_provenance,
        'reference_answer_sha256': answer_entry['sha256'], 'script_sha256': _sha(Path(__file__)),
        'reference_required_array_parameters_with_empty_candidate_lists': ['live_simple_106-63-0', 'live_simple_112-68-0'],
        'quality_or_SLO_accepted_by_user': False,
        'limits': ['All258 original requests and official denominator retained, including two inconsistent reference records',
                   'Native function name/argument matching is not external tool execution or full multistep agent success',
                   'One same-GPU deployment pair; no run-level confidence interval or pure KV-dtype causal claim',
                   'Actual natural output lengths and errors retained; buffered mode reports full JSON latency and no streaming TTFT/ITL']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mirror', 'source', 'prepared', 'source-repo', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    result = audit(args.mirror, args.run_id, args.source, args.prepared, args.source_repo)
    write_new_json(args.out, result)
    print(json.dumps({'classification': result['classification'], 'run_id': args.run_id,
                      'matches': {a: r['official_argument_matches'] for a,r in result['arms'].items()}}))


if __name__ == '__main__':
    main()
