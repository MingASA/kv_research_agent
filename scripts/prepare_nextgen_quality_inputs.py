"""Freeze real-text quality requests with the new model's exact tokenizer.

No generation, GPU work, truncation, answer-dependent selection or downloads.
The cached GSM8K cohort and original official LongBench archive are reused as
data; no old-model result is a new model quality baseline. LongBench selection
is made directly from the original archive, without inheriting a model-specific
historical context filter.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import zipfile

from scripts.controller.durable import canonical_bytes, write_new_bytes, write_new_json


TASKS = ('gsm8k', 'narrativeqa', 'qasper', 'multifieldqa_en', '2wikimqa', 'hotpotqa', 'musique', 'lcc', 'repobench-p')
MODEL = 'Qwen3.8-27B'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--scorer', type=Path, required=True)
    parser.add_argument('--shots', type=Path, required=True)
    parser.add_argument('--official-archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    parent = json.loads((args.cohort/'full_reference_manifest.json').read_text())
    archive_receipt = json.loads((args.official_archive.parent/'download_receipt.json').read_text())
    if _sha(args.official_archive) != archive_receipt['archive_sha256']:
        raise ValueError('Official cached archive differs from its pinned source receipt')
    prompt_path = args.scorer/'LongBench/config/dataset2prompt.json'
    cap_path = args.scorer/'LongBench/config/dataset2maxlen.json'
    prompts = json.loads(prompt_path.read_text())
    caps = json.loads(cap_path.read_text())
    shots = json.loads(args.shots.read_text())
    if len(shots) != 5:
        raise ValueError('Expected the verified five training examples')
    examples = '\n\n'.join(f"Question: {row['question']}\n Answer: {row['answer']}" for row in shots)
    requests, labels, statistics = [], [], {}

    def add(task: str, index: int, row: dict, group: str) -> None:
        prompt = (f"{examples}\n\nQuestion: {row['question']}\n Answer:" if task == 'gsm8k'
                  else prompts[task].format(context=row['context'], input=row['input']))
        chat = task not in ('lcc', 'repobench-p')
        request = {'model': MODEL, 'temperature': 0, 'max_tokens': 512 if task == 'gsm8k' else int(caps[task])}
        if chat:
            messages = [{'role': 'user', 'content': prompt}]
            request.update(messages=messages, chat_template_kwargs={'enable_thinking': False})
            tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   enable_thinking=False, return_dict=True)['input_ids']
        else:
            request['prompt'] = prompt
            tokens = tokenizer(prompt, add_special_tokens=True)['input_ids']
        if len(tokens) + request['max_tokens'] > 131072:
            raise ValueError(f'Untruncated real request exceeds declared 128K budget: {task}/{index}')
        identifier = f'{group}-{task}-{index}'
        context_sha = hashlib.sha256(row.get('context', '').encode()).hexdigest() if task != 'gsm8k' else None
        payload = {'id': identifier, 'task': task, 'group': group, 'source_index': index,
                   'endpoint': '/v1/chat/completions' if chat else '/v1/completions', 'request': request,
                   'prompt_tokens_cpu': len(tokens), 'source_record_sha256': hashlib.sha256(canonical_bytes(row)).hexdigest(),
                   'context_sha256': context_sha}
        requests.append(payload)
        labels.append({'id': identifier, 'task': task, 'group': group, 'source_index': index,
                       'context_sha256': context_sha, 'reference': row})
        statistics.setdefault(group, {}).setdefault(task, []).append(len(tokens))

    with zipfile.ZipFile(args.official_archive) as archive:
        for task in TASKS:
            if task == 'gsm8k':
                path = args.cohort/(task+'.jsonl')
                if _sha(path) != parent['data_sha256'][task]:
                    raise ValueError('Original GSM8K cohort SHA differs')
                rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            else:
                rows = [json.loads(line) for line in archive.read('data/'+task+'.jsonl').decode().splitlines() if line][:50]
            if len(rows) != (1319 if task == 'gsm8k' else 50):
                raise ValueError('New model core cohort is incomplete')
            for index, row in enumerate(rows):
                add(task, index, row, 'core')
    # One question per document, longest documents first; answers are never used.
    with zipfile.ZipFile(args.official_archive) as archive:
        narrative = [json.loads(line) for line in archive.read('data/narrativeqa.jsonl').decode().splitlines() if line]
    documents = {}
    for index, row in enumerate(narrative):
        context_sha = hashlib.sha256(row['context'].encode()).hexdigest()
        documents.setdefault(context_sha, (index, row))
    chosen = sorted(documents.values(), key=lambda pair: (-len(pair[1]['context']), pair[0]))[:8]
    core_ids = {entry['source_record_sha256'] for entry in requests}
    supplement = []
    for index, row in chosen:
        if hashlib.sha256(canonical_bytes(row)).hexdigest() in core_ids:
            continue
        add('narrativeqa', index, row, 'long_unique_document')
        supplement.append(index)
    args.out.mkdir(parents=True, exist_ok=False)
    write_new_bytes(args.out/'requests.jsonl', b''.join(canonical_bytes(row) for row in requests))
    write_new_bytes(args.out/'labels.jsonl', b''.join(canonical_bytes(row) for row in labels))
    manifest = {
        'schema': 'kv-nextgen-real-quality-input-v1', 'classification': 'CPU_frozen_input_not_model_results',
        'model_id': MODEL, 'model_revision': REVISION, 'model_inference_performed': False, 'GPU_used': False,
        'core_samples': 1719, 'long_supplement_source_indices': supplement,
        'core_selection': 'All1319 verified GSM8K test records; first50 unmodified official archive records per8 LongBench tasks',
        'old_model_context_filter_inherited': False,
        'long_document_selection': 'First question per exact context SHA; top8 documents by characters; skip identical core records',
        'all_contexts_untruncated': True, 'enable_thinking': False, 'natural_stopping': True,
        'input_budget_tokens': 131072, 'declared_model_max_context_tokens': 262144,
        'input_budget_is_not_evidence_of_128K_requests_or_model_support': True,
        'request_payload_sha256': _sha(args.out/'requests.jsonl'), 'labels_sha256': _sha(args.out/'labels.jsonl'),
        'parent_cohort_manifest_sha256': _sha(args.cohort/'full_reference_manifest.json'),
        'official_archive_sha256': _sha(args.official_archive), 'five_train_examples_sha256': _sha(args.shots),
        'official_archive_source': {'repository': archive_receipt['repository'], 'revision': archive_receipt['revision'],
                                    'url': archive_receipt['archive_url'], 'source_receipt_sha256': _sha(args.official_archive.parent/'download_receipt.json')},
        'official_prompt_format_sha256': _sha(prompt_path), 'official_generation_caps_sha256': _sha(cap_path),
        'official_metrics_source_sha256': _sha(args.scorer/'LongBench/metrics.py'),
        'tokenizer_file_sha256': {p.name: _sha(p) for p in args.tokenizer.iterdir() if p.is_file()},
        'versions': {name: importlib.metadata.version(name) for name in ('transformers', 'tokenizers')},
        'prompt_token_statistics': {group: {task: {'count': len(values), 'min': min(values), 'max': max(values),
                                                  'total': sum(values)} for task, values in tasks.items()}
                                    for group, tasks in statistics.items()},
        'preparation_source_sha256': _sha(Path(__file__)),
        'limitations': ['LongBench code similarity is not executable code correctness',
                        'No tool-agent benchmark or 128K real-text request is certified by this input freeze',
                        'Shared document prefixes and per-document clustering must be retained in analysis'],
    }
    write_new_json(args.out/'manifest.json', manifest)
    print(json.dumps({'output': str(args.out), 'core': 1719, 'long_supplement': len(supplement),
                      'maximum_actual_prompt_tokens': max(row['prompt_tokens_cpu'] for row in requests),
                      'model_predictions_generated': False}))


if __name__ == '__main__':
    main()
