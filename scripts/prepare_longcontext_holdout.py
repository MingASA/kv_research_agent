#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Freeze answer-independent, exact-context-disjoint real long-document QA inputs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.controller.durable import canonical_bytes, digest, secure_directory, write_new_bytes, write_new_json
from scripts.controller.range_probe import LONG_QA_SCOPE, LONG_QA_TASK_CAPS, validate_longcontext_inputs

TASK_CAPS = LONG_QA_TASK_CAPS
CONTENT_SCOPE = LONG_QA_SCOPE
REVISION = '5e628be450b7e67fb7ae6e201bd6d8f7056f7672'
ARCHIVE_SHA256 = 'cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64'


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            value.update(chunk)
    return value.hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _token_count(tokenizer, prompt: str) -> int:
    tokens = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True)
    if hasattr(tokens, 'get'):
        tokens = tokens['input_ids']
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    return len(tokens)


def prepare(source: Path, excluded: Path, config: Path, tokenizer_path: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError('Use a new immutable long-document input directory')
    receipt = json.loads((source / 'download_receipt.json').read_text())
    archive = source / 'data.zip'
    if (receipt['revision'] != REVISION or receipt['archive_sha256'] != ARCHIVE_SHA256
            or archive.stat().st_size != 113932529 or _sha(archive) != ARCHIVE_SHA256):
        raise ValueError('Pinned official archive length, SHA or revision differs')
    templates = json.loads((config / 'dataset2prompt.json').read_text())
    caps = json.loads((config / 'dataset2maxlen.json').read_text())
    if any(caps.get(task) != cap for task, cap in TASK_CAPS.items()):
        raise ValueError('Original task generation caps changed')
    old_files = {p.stem: p for p in excluded.glob('*.jsonl') if p.stem != 'gsm8k'}
    if len(old_files) != 8:
        raise ValueError('Need all eight previously measured LongBench task files')
    old = {task: [json.loads(s) for s in path.read_text().split('\n') if s]
           for task, path in old_files.items()}
    if any(len(rows) != 50 for rows in old.values()):
        raise ValueError('Expected the original 8x50 evaluation cohort')
    old_rows = [r for rows in old.values() for r in rows]
    old_ids = {r['_id'] for r in old_rows}
    old_contexts = {_text_sha(r['context']) for r in old_rows}
    old_records = {digest(r) for r in old_rows}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    selected = []
    sources = {}
    seen_contexts = set(old_contexts)
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.infolist()) != 35 or sum(r.file_size for r in bundle.infolist()) > 400 * 2**20:
            raise ValueError('Unexpected official archive expansion')
        for task, cap in TASK_CAPS.items():
            member = 'data/' + task + '.jsonl'
            if bundle.getinfo(member).file_size > 64 * 2**20:
                raise ValueError('Source task exceeds the local input bound')
            raw = bundle.read(member)
            rows = [json.loads(s) for s in raw.split(b'\n') if s]
            by_id = {row['_id']: (index, row) for index, row in enumerate(rows)}
            if len(by_id) != len(rows) or any(
                    row['_id'] not in by_id or by_id[row['_id']][1] != row for row in old[task]):
                raise ValueError('Frozen evaluation records do not match the pinned source by ID and all fields')
            old_source_indices = [by_id[row['_id']][0] for row in old[task]]
            pool = []
            counts = Counter()
            for index, original in enumerate(rows):
                # Public source records contain labels. Projection deliberately drops them;
                # neither membership nor length/ranking uses answers or previous predictions.
                row = {key: original[key] for key in ('_id', 'context', 'input')}
                context_sha = _text_sha(row['context'])
                if (row['_id'] in old_ids or context_sha in seen_contexts
                        or digest(original) in old_records):
                    counts['excluded_old_or_duplicate_context'] += 1
                    continue
                seen_contexts.add(context_sha)
                prompt = templates[task].format(context=row['context'], input=row['input'])
                tokens = _token_count(tokenizer, prompt)
                if tokens < 4096 or tokens + cap > 32768:
                    counts['outside_4K_to_untruncated_model_limit'] += 1
                    continue
                pool.append({'task': task, 'source_index': index, 'record_id': row['_id'],
                             'source_record_sha256': digest(original), 'context_sha256': context_sha,
                             'prompt_sha256': _text_sha(prompt), 'prompt_tokens_cpu': tokens,
                             'max_tokens': cap, 'prompt': prompt})
            pool.sort(key=lambda r: (r['prompt_tokens_cpu'], r['source_index']))
            if len(pool) < 32:
                raise ValueError('Insufficient distinct, eligible real documents: ' + task)
            for quartile in range(4):
                bucket = pool[len(pool) * quartile // 4:len(pool) * (quartile + 1) // 4]
                bucket.sort(key=lambda r: digest({'seed': 20261027, 'task': task, 'id': r['record_id']}))
                if len(bucket) < 8:
                    raise ValueError('Insufficient independent documents in quartile')
                selected.extend(dict(row, length_quartile=quartile) for row in bucket[:8])
            sources[task] = {'member': member, 'sha256': hashlib.sha256(raw).hexdigest(),
                             'records': len(rows), 'eligible_distinct_documents': len(pool),
                             'selected': 32, 'exclusions': dict(counts),
                             'previous_evaluation_source_indices': old_source_indices,
                             'previous_evaluation_all_fields_equal_by_ID': True}
    selected.sort(key=lambda r: (list(TASK_CAPS).index(r['task']), r['source_index']))
    sessions, lineage = [], []
    for row in selected:
        payload = {'model': 'Qwen2.5-14B-Instruct', 'messages': [{'role': 'user', 'content': row['prompt']}],
                   'stream': True, 'stream_options': {'include_usage': True},
                   'temperature': 0, 'top_p': 1, 'seed': 17, 'max_tokens': row['max_tokens']}
        identity = 'longbench-heldout-' + row['task'] + '-' + str(row['source_index'])
        sessions.append({'session_id': identity, 'payloads': [payload]})
        lineage.append({key: value for key, value in row.items() if key != 'prompt'}
                       | {'session_id': identity, 'payload_sha256': digest(payload)})
    data = {'data': sessions}
    manifest = {'schema': 'kv-real-heldout-longcontext-QA-input-v1',
        'classification': 'CPU_input_proposal_no_activations', 'content_scope': CONTENT_SCOPE,
        'requests': 128, 'canonical_sha256': digest(data), 'source_content_sha256': ARCHIVE_SHA256,
        'official_repository': 'zai-org/LongBench', 'official_revision': REVISION,
        'source_archive_url': receipt['archive_url'], 'source_download_receipt_sha256': _sha(source / 'download_receipt.json'),
        'source_records_contain_public_labels': True, 'quality_labels_used_in_selection': False,
        'answers_exported_to_probe': False, 'previous_predictions_used_in_selection': False,
        'official_generation_caps_preserved': True, 'messages_and_natural_stop_controls_unchanged': True,
        'selection': 'Four document-QA tasks,32 each;8 per eligible distinct-document prompt-length quartile;SHA rank seed20261027;no truncation,padding,answer or model-output ranking',
        'evaluation_excluded_count': 400, 'excluded_source_sha256': {task: _sha(path) for task, path in old_files.items()},
        'excluded_record_ids_sha256': digest(sorted(old_ids)), 'excluded_context_sha256_set_digest': digest(sorted(old_contexts)),
        'overlap_with_previous_evaluation': {'record_ids': 0, 'exact_records': 0, 'exact_contexts': 0},
        'semantic_or_training_contamination_ruled_out': False, 'selected_lineage': lineage,
        'source_tasks': sources, 'preparation_source_sha256': _sha(Path(__file__)),
        'prompt_template_sha256': _sha(config / 'dataset2prompt.json'),
        'task_generation_cap_config_sha256': _sha(config / 'dataset2maxlen.json'),
        'tokenizer_metadata_sha256': {p.name: _sha(p) for p in sorted(tokenizer_path.iterdir()) if p.is_file()},
        'prompt_tokens_cpu': {'min': min(r['prompt_tokens_cpu'] for r in lineage),
            'max': max(r['prompt_tokens_cpu'] for r in lineage), 'total': sum(r['prompt_tokens_cpu'] for r in lineage),
            'at_least8K': sum(r['prompt_tokens_cpu'] >= 8192 for r in lineage),
            'at_least16K': sum(r['prompt_tokens_cpu'] >= 16384 for r in lineage)},
        'model_inference_performed': False, 'gpu_used': False,
        'limits': ['Exact ID/record/context independence, not proof of semantic or model-training independence',
                   'Long-document QA only; no code-completion or FP8 generation feedback coverage',
                   'Selection is diagnostic stratification, not production task traffic proportions',
                   'No model quality, throughput or clipping result exists at CPU preparation time']}
    validate_longcontext_inputs(data, manifest)
    secure_directory(out)
    write_new_bytes(out / 'canonical_inputs.json', canonical_bytes(data))
    write_new_json(out / 'manifest.json', manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'excluded', 'config', 'tokenizer', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.excluded, args.config, args.tokenizer, args.out)
    print(json.dumps({k: result[k] for k in ('classification', 'requests', 'canonical_sha256',
                                            'prompt_tokens_cpu', 'overlap_with_previous_evaluation')}))


if __name__ == '__main__':
    main()
