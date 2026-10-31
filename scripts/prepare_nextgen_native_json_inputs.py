"""Correct CPU token metadata while preserving every actual API payload/label."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from scripts.controller.durable import canonical_bytes, write_new_bytes, write_new_json
from scripts.nextgen_tokenization import OfficialJSONTokenizer


def main() -> None:
    root=Path(__file__).resolve().parents[1]
    original=root/'results/prepared/nextgen-quality-real-text-20261027a'
    destination=root/'results/prepared/nextgen-quality-real-text-native-json-v2-20261028a'
    manifest=json.loads((original/'manifest.json').read_text())
    source=(original/'requests.jsonl').read_bytes()
    if hashlib.sha256(source).hexdigest()!=manifest['request_payload_sha256']:
        raise ValueError('Preserve the immutable original input source')
    tokenizer_dir=root/'.deps/nextgen-official-models-20261027-a1/Qwen3.8-27B'
    tokenizer=OfficialJSONTokenizer(tokenizer_dir)
    rows=[json.loads(line) for line in source.split(b'\n') if line]
    corrected=[];changes=[];statistics=defaultdict(list)
    payload_stream=b''.join(canonical_bytes({'id':row['id'],'endpoint':row['endpoint'],'request':row['request']}) for row in rows)
    for row in rows:
        count=tokenizer.count(row['request'])
        if count!=row['prompt_tokens_cpu']:
            changes.append({'id':row['id'],'old_AutoTokenizer_count':row['prompt_tokens_cpu'],'official_JSON_count':count})
        updated={**row,'prompt_tokens_cpu':count}
        if updated['request']!=row['request'] or updated['endpoint']!=row['endpoint']:
            raise ValueError('Token metadata correction must not alter an API payload')
        corrected.append(updated);statistics[(row['group'],row['task'])].append(count)
    emitted=b''.join(canonical_bytes(row) for row in corrected)
    labels=(original/'labels.jsonl').read_bytes()
    if hashlib.sha256(labels).hexdigest()!=manifest['labels_sha256']:
        raise ValueError('Original independent labels changed')
    destination.mkdir(mode=0o700,exist_ok=False)
    write_new_bytes(destination/'requests.jsonl',emitted)
    write_new_bytes(destination/'labels.jsonl',labels)
    write_new_bytes(destination/'scoring_protocol.json',(original/'scoring_protocol.json').read_bytes())
    updated=json.loads(json.dumps(manifest))
    updated.update(schema='kv-nextgen-real-quality-input-native-json-v2',
                   request_payload_sha256=hashlib.sha256(emitted).hexdigest(),
                   original_input_metadata_sha256=manifest['request_payload_sha256'],
                   canonical_API_payloads_sha256=hashlib.sha256(payload_stream).hexdigest(),
                   actual_API_payloads_and_labels_unchanged=True,
                   counting_protocol='Pinned official tokenizer.json BPE; AutoTokenizer renders chat only, no rebuilt pre-tokenizer',
                   token_metadata_corrections=changes,
                   tokenizer_JSON_sha256=hashlib.sha256((tokenizer_dir/'tokenizer.json').read_bytes()).hexdigest(),
                   preparation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    stats={}
    for (group,task),counts in statistics.items():
        stats.setdefault(group,{})[task]={'count':len(counts),'min':min(counts),'max':max(counts),'total':sum(counts)}
    updated['prompt_token_statistics']=stats
    write_new_json(destination/'manifest.json',updated)


if __name__=='__main__':main()
