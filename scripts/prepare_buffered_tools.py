#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Derive complete-JSON native tool requests without changing task semantics."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.controller.durable import digest, secure_directory, write_new_json


def derive(parent: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError('New immutable buffered tool input directory required')
    data = json.loads((parent / 'tool_inputs.json').read_text())
    original = copy.deepcopy(data)
    manifest = json.loads((parent / 'manifest.json').read_text())
    if digest(data) != manifest['dataset_sha256'] or len(data['data']) != 274:
        raise ValueError('Original pre-answer frozen tool cohort differs')
    for session, record in zip(data['data'], manifest['records']):
        payload = session['payloads'][0]
        if digest(payload) != record['payload_sha256'] or payload.get('stream') is not True:
            raise ValueError('Only the original frozen streaming request can be derived')
        record['parent_payload_sha256'] = record['payload_sha256']
        payload['stream'] = False
        payload.pop('stream_options')
        record['payload_sha256'] = digest(payload)
    for old, new in zip(original['data'], data['data']):
        left = old['payloads'][0]; right = new['payloads'][0]
        if {k:v for k,v in left.items() if k not in ('stream', 'stream_options')} != {
                k:v for k,v in right.items() if k not in ('stream', 'stream_options')}:
            raise ValueError('Buffered transport must not change messages/tools/sampling/cap')
    manifest.update(response_mode='buffered', parent_source_dataset_sha256=manifest['dataset_sha256'],
        parent_manifest_sha256=hashlib.sha256((parent/'manifest.json').read_bytes()).hexdigest(),
        dataset_sha256=digest(data), changed_fields={'stream': False, 'stream_options': 'removed'},
        derived_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        derivation_reason='Production complete-JSON native function calling; streaming parser corruption retained separately',
        ground_truth_used_for_selection_or_prompt=False)
    secure_directory(out); write_new_json(out/'tool_inputs.json', data); write_new_json(out/'manifest.json', manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); result = derive(args.parent, args.out)
    print(json.dumps({'dataset_sha256': result['dataset_sha256'], 'requests': 258, 'warmup': 16, 'GPU': False}))


if __name__ == '__main__':
    main()
