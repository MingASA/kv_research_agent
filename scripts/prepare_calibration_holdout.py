#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Freeze real held-out prompts for future untimed activation calibration only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from controller.durable import canonical_bytes, digest, secure_directory, write_new_bytes, write_new_json


def prepare(parent: Path, out: Path) -> dict:
    inputs = parent/'canonical_inputs.json'
    manifest_path = parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(inputs.read_bytes()).hexdigest() != manifest['canonical_sha256']:
        raise ValueError('Frozen parent changed')
    sessions = json.loads(inputs.read_text())['data']
    lineage = {row['session_id']:row for row in manifest['session_lineage']}
    if len(sessions) != 1007 or len(lineage) != len(sessions):
        raise ValueError('Unexpected independently audited natural-chat parent')
    excluded = {row['session_id'] for row in sessions[:528]}
    pool = sorted(range(528,len(sessions)),key=lambda i:(lineage[sessions[i]['session_id']]['prompt_tokens_cpu'],i))
    selected = []
    for quartile in range(4):
        bucket = pool[len(pool)*quartile//4:len(pool)*(quartile+1)//4]
        ranked = sorted(bucket,key=lambda i:digest({'seed':20261027,'session_id':sessions[i]['session_id']}))
        selected.extend(ranked[:32])
    selected.sort()
    if len(selected) != 128 or len(set(selected)) != 128:
        raise ValueError('Insufficient independent calibration requests')
    mapping = []
    for index in selected:
        row=sessions[index]; evidence=lineage[row['session_id']]
        if row['session_id'] in excluded or len(row['payloads']) != 1:
            raise ValueError('Service overlap or changed session semantics')
        payload=row['payloads'][0]
        if hashlib.sha256(canonical_bytes(payload)).hexdigest() != evidence['payload_sha256']:
            raise ValueError('Parent canonical payload SHA, including its final LF, changed')
        if payload.get('ignore_eos',False) or payload['max_tokens'] != 1024:
            raise ValueError('Natural stopping controls changed')
        mapping.append({'parent_index':index,**evidence})
    if out.exists():
        raise FileExistsError('Preserve all existing calibration proposals')
    secure_directory(out)
    data=canonical_bytes({'data':[sessions[i] for i in selected]})
    write_new_bytes(out/'canonical_inputs.json',data)
    result={'schema':'kv-real-heldout-calibration-input-v1','classification':'CPU_input_proposal_no_activations',
        'preparation_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_parent_sha256':manifest['canonical_sha256'],
        'parent_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'source_content_sha256':manifest['source_sha256'],'canonical_sha256':hashlib.sha256(data).hexdigest(),
        'selection':'Unused479 after original528 service requests;32 per prompt-length quartile, SHA ranking seed20261027',
        'requests':128,'service_excluded_count':528,'service_excluded_session_ids_sha256':digest(sorted(excluded)),
        'quality_labels_loaded':False,'messages_and_natural_stop_controls_unchanged':True,
        'model_inference_performed':False,'gpu_used':False,'selected_lineage':mapping,
        'prompt_tokens_cpu':{'min':min(v['prompt_tokens_cpu'] for v in mapping),
                            'max':max(v['prompt_tokens_cpu'] for v in mapping),
                            'total':sum(v['prompt_tokens_cpu'] for v in mapping)},
        'limitations':['No long-document32K or production code coverage claim',
                       'Separate data origin, not a proof of semantic nonoverlap with every evaluation question',
                       'No scales derived; existing first-step observer cannot calibrate this complete cohort']}
    write_new_json(out/'manifest.json',result)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();value=prepare(args.parent,args.out)
    print(json.dumps({key:value[key] for key in ('classification','requests','prompt_tokens_cpu','gpu_used')}))


if __name__ == '__main__':
    main()
