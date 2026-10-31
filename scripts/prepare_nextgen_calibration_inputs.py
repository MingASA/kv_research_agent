"""Retokenize the existing held-out real conversations for the new model.

No weights or model predictions are loaded. Input conversations,
natural stopping and generation caps are preserved from the audited holdout.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from scripts.controller.durable import write_new_json


def main() -> None:
    from transformers import AutoTokenizer

    root = Path(__file__).resolve().parents[1]
    parent = root / 'results/prepared/calibration-sharegpt-heldout128-20261027a2'
    parent_manifest = json.loads((parent / 'manifest.json').read_text())
    raw = (parent / 'canonical_inputs.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != parent_manifest['canonical_sha256']:
        raise ValueError('Existing held-out source changed')
    sessions = json.loads(raw)['data']
    tokenizer_path = root / '.deps/nextgen-official-models-20261027-a1/Qwen3.8-27B'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    rows = []
    for session in sessions:
        if len(session['payloads']) != 1:
            raise ValueError('Do not flatten or change conversation/session semantics')
        original = session['payloads'][0]
        if original.get('ignore_eos', False) or original['max_tokens'] != 1024:
            raise ValueError('Audited natural stopping controls changed')
        payload = {name: value for name, value in original.items() if name not in ('stream', 'stream_options')}
        payload.update(model='Qwen3.8-27B', stream=False, chat_template_kwargs={'enable_thinking': False})
        tokens = tokenizer.apply_chat_template(payload['messages'], tokenize=True,
                                               add_generation_prompt=True, enable_thinking=False,
                                               return_dict=True)['input_ids']
        if len(tokens) + payload['max_tokens'] > 131072:
            raise ValueError('Calibration input exceeds the declared input budget')
        messages = json.dumps(payload['messages'], ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
        rows.append({'id': session['session_id'], 'payload': payload,
                     'prompt_tokens_cpu': len(tokens), 'messages_sha256': hashlib.sha256(messages).hexdigest()})
    if len(rows) != 128 or len({row['id'] for row in rows}) != 128:
        raise ValueError('Unexpected independent calibration inventory')
    quality = root / 'results/prepared/nextgen-quality-real-text-20261027a'
    quality_manifest = json.loads((quality / 'manifest.json').read_text())
    quality_raw = (quality / 'requests.jsonl').read_bytes()
    if hashlib.sha256(quality_raw).hexdigest() != quality_manifest['request_payload_sha256']:
        raise ValueError('Frozen quality context exclusion source changed')
    excluded_contexts = {json.loads(line).get('context_sha256') for line in quality_raw.splitlines()}
    scorer = root / 'results/mirrors/quality-protocols-20261026T075846Z/files/support/inputs/longbench-scorer/LongBench/config'
    prompt_path, cap_path = scorer / 'dataset2prompt.json', scorer / 'dataset2maxlen.json'
    if (hashlib.sha256(prompt_path.read_bytes()).hexdigest() != quality_manifest['official_prompt_format_sha256']
            or hashlib.sha256(cap_path.read_bytes()).hexdigest() != quality_manifest['official_generation_caps_sha256']):
        raise ValueError('Official prompt/cap definitions changed')
    prompts, caps = json.loads(prompt_path.read_text()), json.loads(cap_path.read_text())
    archive = root / '.deps/longbench-official-5e628be-20261027-a1/data.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != quality_manifest['official_archive_sha256']:
        raise ValueError('Cached official LongBench archive changed')
    with zipfile.ZipFile(archive) as data:
        for task in ('gov_report', 'qmsum'):
            candidates = [json.loads(line) for line in data.read('data/' + task + '.jsonl').splitlines()]
            ranked = sorted(enumerate(candidates), key=lambda item: (-len(item[1]['context']), item[0]))
            selected = 0
            for index, record in ranked:
                context_sha = hashlib.sha256(record['context'].encode()).hexdigest()
                if context_sha in excluded_contexts:
                    continue
                excluded_contexts.add(context_sha)
                prompt = prompts[task].format(context=record['context'], input=record['input'])
                messages = [{'role': 'user', 'content': prompt}]
                tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       enable_thinking=False, return_dict=True)['input_ids']
                payload = {'model': 'Qwen3.8-27B', 'messages': messages, 'temperature': 0,
                           'max_tokens': int(caps[task]), 'stream': False,
                           'chat_template_kwargs': {'enable_thinking': False}}
                if len(tokens) + payload['max_tokens'] > 131072:
                    raise ValueError('Untruncated held-out long context exceeds input budget')
                rows.append({'id': 'long-holdout-' + task + '-' + str(index), 'payload': payload,
                             'prompt_tokens_cpu': len(tokens), 'context_sha256': context_sha,
                             'source_index': index, 'task': task,
                             'messages_sha256': hashlib.sha256(json.dumps(messages, ensure_ascii=False,
                                sort_keys=True, separators=(',', ':')).encode()).hexdigest()})
                selected += 1
                if selected == 4:
                    break
            if selected != 4:
                raise ValueError('Insufficient independent held-out long documents')
    output = root / 'results/prepared/nextgen-calibration-chat128-long8-20261027b'
    output.mkdir(mode=0o700, exist_ok=False)
    write_new_json(output / 'calibration.json', {'schema': 'nextgen-real-heldout-calibration-input-v1',
                                                'requests': rows})
    write_new_json(output / 'manifest.json', {
        'classification': 'CPU_frozen_real_input_no_model_predictions_or_scales',
        'requests': 136, 'original_chat_requests': 128, 'independent_long_document_requests': 8,
        'canonical_sha256': hashlib.sha256((output / 'calibration.json').read_bytes()).hexdigest(),
        'parent_canonical_sha256': parent_manifest['canonical_sha256'],
        'parent_manifest_sha256': hashlib.sha256((parent / 'manifest.json').read_bytes()).hexdigest(),
        'parent': str(parent), 'source_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'new_model_tokenizer': str(tokenizer_path), 'model_id': 'Qwen3.8-27B',
        'original_128_chat_messages_unchanged': True, 'natural_stopping_preserved': True,
        'enable_thinking': False, 'original_chat_output_cap': 1024, 'long_document_output_caps': caps['gov_report'],
        'long_document_source_archive_sha256': quality_manifest['official_archive_sha256'],
        'long_document_selection': 'Four longest unique contexts each of gov_report/qmsum, excluded exact quality context SHAs; no answer-derived selection',
        'prompt_tokens': {'min': min(row['prompt_tokens_cpu'] for row in rows),
                          'max': max(row['prompt_tokens_cpu'] for row in rows),
                          'total': sum(row['prompt_tokens_cpu'] for row in rows)},
        'quality_labels_or_answers_used_for_selection': False,
        'limitations': ['Real held-out ShareGPT, with parent selection established before new-model outputs.',
                       'Eight independent long documents extend observed context coverage; no unseen-range or tool-agent certificate.',
                       'Corpus separation does not establish semantic nonoverlap with every evaluation question.'],
    })


if __name__ == '__main__':
    main()
