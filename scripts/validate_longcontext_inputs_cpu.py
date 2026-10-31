#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Independently bind every frozen diagnostic prompt to original public records."""

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
from scripts.controller.durable import digest, write_new_json
from scripts.controller.range_probe import validate_longcontext_inputs


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(prepared: Path, source: Path, excluded: Path, config: Path, tokenizer_path: Path) -> dict:
    manifest = json.loads((prepared / 'manifest.json').read_text())
    data = json.loads((prepared / 'canonical_inputs.json').read_text())
    validate_longcontext_inputs(data, manifest)
    if sha(source / 'data.zip') != manifest['source_content_sha256']:
        raise ValueError('Pinned archive SHA differs')
    if sha(ROOT / 'scripts/prepare_longcontext_holdout.py') != manifest['preparation_source_sha256']:
        raise ValueError('Final input preparation source differs')
    templates = json.loads((config / 'dataset2prompt.json').read_text())
    if (sha(config / 'dataset2prompt.json') != manifest['prompt_template_sha256']
            or sha(config / 'dataset2maxlen.json') != manifest['task_generation_cap_config_sha256']):
        raise ValueError('Original prompt/cap configuration differs')
    old = []
    for task, expected in manifest['excluded_source_sha256'].items():
        path = excluded / (task + '.jsonl')
        if sha(path) != expected:
            raise ValueError('Previously measured evaluation data differs')
        old.extend(json.loads(line) for line in path.read_text().split('\n') if line)
    old_ids = {r['_id'] for r in old}
    old_contexts = {hashlib.sha256(r['context'].encode()).hexdigest() for r in old}
    old_records = {digest(r) for r in old}
    if (len(old) != 400 or digest(sorted(old_ids)) != manifest['excluded_record_ids_sha256']
            or digest(sorted(old_contexts)) != manifest['excluded_context_sha256_set_digest']):
        raise ValueError('Original full400 exclusion set differs')
    actual_metadata = {p.name: sha(p) for p in sorted(tokenizer_path.iterdir()) if p.is_file()}
    if actual_metadata != manifest['tokenizer_metadata_sha256']:
        raise ValueError('Actual tokenizer/model metadata differs')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    sources, records = {}, []
    for row in data['data']:
        if len(row['payloads']) != 1:
            raise ValueError('Unexpected additional payload')
    with zipfile.ZipFile(source / 'data.zip') as archive:
        for task, origin in manifest['source_tasks'].items():
            raw = archive.read(origin['member'])
            if hashlib.sha256(raw).hexdigest() != origin['sha256']:
                raise ValueError('Original source task bytes differ')
            sources[task] = [json.loads(line) for line in raw.split(b'\n') if line]
        for session, lineage in zip(data['data'], manifest['selected_lineage']):
            task = lineage['task']
            original = sources[task][lineage['source_index']]
            prompt = templates[task].format(context=original['context'], input=original['input'])
            context_sha = hashlib.sha256(original['context'].encode()).hexdigest()
            if (original['_id'] != lineage['record_id'] or digest(original) != lineage['source_record_sha256']
                    or original['_id'] in old_ids or digest(original) in old_records or context_sha in old_contexts
                    or context_sha != lineage['context_sha256']
                    or session['payloads'][0]['messages'] != [{'role':'user','content':prompt}]):
                raise ValueError('Original prompt identity/content or measured-evaluation independence differs')
            tokens = tokenizer.apply_chat_template([{'role':'user','content':prompt}],
                tokenize=True, add_generation_prompt=True)
            if hasattr(tokens, 'get'):
                tokens = tokens['input_ids']
            if tokens and isinstance(tokens[0], list):
                tokens = tokens[0]
            if len(tokens) != lineage['prompt_tokens_cpu']:
                raise ValueError('Independent actual tokenizer count differs')
            records.append({'session_id':session['session_id'], 'task':task,
                'prompt_tokens':len(tokens), 'source_index':lineage['source_index'],
                'context_sha256':context_sha, 'payload_sha256':digest(session['payloads'][0])})
    if len({r['context_sha256'] for r in records}) != 128:
        raise ValueError('Repeated document contexts in calibration')
    quartiles = Counter((r['task'], r['length_quartile']) for r in manifest['selected_lineage'])
    if len(quartiles) != 16 or any(n != 8 for n in quartiles.values()):
        raise ValueError('Declared length-stratified task sampling differs')
    return {'classification':'confirmed_CPU_real_prompt_source_tokenizer_and_exact_context_independence_only',
        'GPU_used':False, 'model_inference_performed':False, 'requests_verified':128,
        'dataset_sha256':sha(prepared / 'canonical_inputs.json'), 'manifest_sha256':sha(prepared / 'manifest.json'),
        'source_archive_sha256':manifest['source_content_sha256'], 'official_source_revision':manifest['official_revision'],
        'old_evaluation_records_excluded':400, 'actual_exact_ID_record_context_overlap':0,
        'answer_based_selection':False, 'answers_in_requests':False,
        'public_source_contains_answers':True, 'semantic_independence_certified':False,
        'original_prompt_and_task_caps_preserved':True, 'tokenizer_metadata_sha256':actual_metadata,
        'prompt_tokens':{'min':min(r['prompt_tokens'] for r in records),
            'max':max(r['prompt_tokens'] for r in records), 'total':sum(r['prompt_tokens'] for r in records)},
        'records':records, 'validator_sha256':sha(Path(__file__))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','source','excluded','config','tokenizer','out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.prepared, args.source, args.excluded, args.config, args.tokenizer)
    write_new_json(args.out, result)
    print(json.dumps({k:result[k] for k in ('classification','requests_verified','prompt_tokens','GPU_used')}))


if __name__ == '__main__':
    main()
