#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Join genuine QA service exports to source answers and complete SSE timing."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_production_run import audit as audit_service
from scripts.audit_trace_run import _application
from scripts.controller.durable import digest, write_new_json
from scripts.controller.production import _jsonl_rows
from scripts.controller.range_probe import LONG_QA_TASK_CAPS, validate_longcontext_inputs
from scripts.controller.trace_serving import _distribution


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def _answers(archive: Path, manifest: dict) -> dict:
    if _sha(archive) != manifest['source_content_sha256']:
        raise ValueError('Official QA archive changed')
    selected = {r['record_id']:r for r in manifest['selected_lineage']}
    found = {}
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.infolist()) != 35 or sum(r.file_size for r in bundle.infolist()) > 400*2**20:
            raise ValueError('Unexpected official source archive expansion')
        for task in LONG_QA_TASK_CAPS:
            member = 'data/'+task+'.jsonl'
            if bundle.getinfo(member).file_size > 64*2**20:
                raise ValueError('Unbounded source task')
            with bundle.open(member) as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    identity = row['_id']
                    if identity not in selected:
                        continue
                    origin = selected[identity]
                    if (identity in found or origin['task'] != task
                            or digest(row) != origin['source_record_sha256']
                            or hashlib.sha256(row['context'].encode()).hexdigest() != origin['context_sha256']):
                        raise ValueError('Selected source record/context/task differs')
                    found[identity] = row
    if set(found) != set(selected):
        raise ValueError('Missing original QA references')
    return found


def _raw_outputs(path: Path) -> dict:
    result = {}
    for raw in _jsonl_rows(path):
        meta = raw['metadata']
        identity = meta['conversation_id']
        if identity in result or meta['benchmark_phase'] != 'profiling' or meta['turn_index'] != 0:
            raise ValueError('Unexpected repeated/nonprofiling QA request')
        text = []
        for response in raw['responses']:
            for packet in response.get('packets', []):
                if packet.get('name') != 'data' or packet.get('value') == '[DONE]':
                    continue
                event = json.loads(packet['value'])
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    content = delta.get('content')
                    if content:
                        if choice.get('index', 0) != 0 or not isinstance(content, str):
                            raise ValueError('Unsupported QA output choice/content')
                        text.append(content)
        result[identity] = {'prediction':''.join(text),
            'complete_sse_ms':(meta['request_end_ns']-meta['request_start_ns'])/1e6}
    return result


def audit(mirror: Path, run_id: str, source: Path, archive: Path, scorer: Path) -> dict:
    base = audit_service(mirror, run_id)
    root = mirror/'support/agent_work/runs'/run_id
    ticket = json.loads((root/'ticket.json').read_text());request = ticket['request']
    terminal = json.loads((root/'result.json').read_text())
    if (request['workload']['kind'] != 'long_document_QA_natural_xqa_bf16_reference'
            or terminal['status'] != 'COMPLETE'
            or json.loads((root/'COMPLETE').read_text())['result_sha256'] != _sha(root/'result.json')):
        raise ValueError('Require the actual completed paired natural QA service')
    for name, expected in ticket['assets_sha256'].items():
        if _sha(root/name) != expected:
            raise ValueError('Frozen runtime/input asset differs: '+name)
    for name, expected in request['backend_source_sha256'].items():
        if _sha(source/name) != expected:
            raise ValueError('Pinned inference source differs: '+name)
    data = json.loads((root/'input/canonical_inputs.json').read_text())
    manifest = json.loads((root/'input/production_manifest.json').read_text())
    validate_longcontext_inputs(data, manifest)
    references = _answers(archive, manifest)
    sys.path.insert(0, str(source))
    from tools import eval_kv_cache_quality as evaluator
    if Path(evaluator.__file__).resolve() != (source/'tools/eval_kv_cache_quality.py').resolve():
        raise ValueError('Unexpected quality evaluator import')
    formats, caps, metrics, _ = evaluator._longbench_assets(scorer)
    if (_sha(scorer/'LongBench/config/dataset2prompt.json') != manifest['prompt_template_sha256']
            or _sha(scorer/'LongBench/config/dataset2maxlen.json') != manifest['task_generation_cap_config_sha256']
            or _sha(scorer/'LongBench/metrics.py') !=
                'e22e2a2662e0f7e683137fa3541f64edb6a801e9138d16d2f3459a6ab9941323'):
        raise ValueError('Pinned official prompt templates/caps/metric source differ')
    by_id = {row['session_id']:row for row in manifest['selected_lineage']}
    per_arm, task_scores = {}, {}
    for arm in request['arms']:
        directory = root/'output'/arm
        if (directory/'range_probe').exists():
            raise ValueError('Activation probe cannot be enabled in the service candidate')
        outputs = _raw_outputs(directory/'artifacts/profile_export_raw.jsonl')
        records = list(_jsonl_rows(directory/'artifacts/profile_export.jsonl'))
        if set(outputs) != set(by_id) or len(records) != 128:
            raise ValueError('Missing genuine service requests')
        scores, rows = collections.defaultdict(list), []
        for record in records:
            identity = record['metadata']['conversation_id'];origin = by_id[identity]
            task = origin['task'];ref = references[origin['record_id']];out = outputs[identity]
            values = record['metrics']
            if caps[task] != origin['max_tokens'] or values['input_sequence_length']['value'] != origin['prompt_tokens_cpu']:
                raise ValueError('Actual model prompt usage/official cap differs')
            expected_prompt = formats[task].format(context=ref['context'], input=ref['input'])
            if hashlib.sha256(expected_prompt.encode()).hexdigest() != origin['prompt_sha256']:
                raise ValueError('Official source prompt differs')
            f1 = max(metrics.qa_f1_score(out['prediction'], answer) for answer in ref['answers'])
            scores[task].append(f1)
            rows.append({'session_id':identity, 'task':task, 'reference_record_id':origin['record_id'],
                'answer_f1':f1, 'output_tokens':values['output_token_count']['value'],
                'official_content_latency_ms':values['request_latency']['value'],
                'complete_sse_ms':out['complete_sse_ms'],
                'prediction_sha256':hashlib.sha256(out['prediction'].encode()).hexdigest()})
        task_scores[arm] = {task:100*statistics.mean(scores[task]) for task in LONG_QA_TASK_CAPS}
        sample_bounds = {'warm_cache_end_epoch_ns':min(r['metadata']['request_start_ns'] for r in records),
            'per_request':[{'request_end_ns':r['metadata']['request_end_ns']} for r in records]}
        application = _application(directory/'service_metrics.jsonl', run_id, ticket['request_sha256'],
            base['sealed_arms'][arm]['allocation']['pages_per_layer'], sample_bounds)
        per_arm[arm] = {'original_service_audit':base['sealed_arms'][arm],
            'per_request':rows, 'task_answer_f1_pct':task_scores[arm],
            'sampled_application_observations':application,
            'complete_sse_ms':_distribution([row['complete_sse_ms'] for row in rows])}
    return {'classification':'confirmed_natural_long_document_QA_service_integrity_and_original_reference_rescoring',
        'run_id':run_id, 'request_sha256':ticket['request_sha256'], 'root_complete':True,
        'gpu_release_verified':True, 'source_archive_sha256':_sha(archive),
        'official_scorer_sha256':_sha(scorer/'LongBench/metrics.py'),
        'actual_gpu_hours':base['actual_gpu_hours'], 'arms':per_arm,
        'four_task_macro_f1_pct':{arm:statistics.mean(scores.values()) for arm,scores in task_scores.items()},
        'quality_accepted':False, 'new_blind_population':False, 'steady_state_SLO_capacity_established':False,
        'limitations':['Finite128 closed-loop real document QA cohort, original task caps and natural EOS',
            'Inputs previously used for BF16 range observation; not fresh blind data or original arrival replay',
            'Different actual output lengths and answers must accompany throughput/latency comparison',
            'Sampled bvars are not exact event counters or HBM traffic; no retrieval system or code/multiturn claim']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mirror', 'source', 'archive', 'scorer', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    result = audit(args.mirror, args.run_id, args.source, args.archive, args.scorer)
    write_new_json(args.out, result)
    print(json.dumps({k:result[k] for k in ('classification', 'run_id', 'actual_gpu_hours', 'four_task_macro_f1_pct')}))


if __name__ == '__main__':
    main()
