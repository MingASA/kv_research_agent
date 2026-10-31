#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Derive a continuous diagnostic window without changing source payloads."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.controller.durable import canonical_bytes, digest, secure_directory, write_new_json
from scripts.controller.trace_serving import validate_inputs


def prepare(parent: Path, out: Path) -> dict:
    rows, manifest = validate_inputs(parent/'payload_trace.jsonl',parent/'manifest.json')
    if manifest['arrival_scaling'] != 1 or manifest['window_end_ms']-manifest['window_begin_ms'] != 300000:
        raise ValueError('Require the original verified300s clock, not an already sliced/scaled stream')
    if out.exists():
        raise FileExistsError('New diagnostic window required')
    end = manifest['window_begin_ms'] + 90000
    records = [r for r in manifest['records'] if r['source_timestamp_ms'] < end]
    identifiers = {r['replay_id'] for r in records}
    retained = [r for r in rows if r['session_id'] in identifiers]
    if (len(records) != len(retained) or not 128 <= len(retained) <= 10000
            or retained != rows[:len(retained)]
            or sum(r['logical_phase']=='warm_cache' for r in records) != manifest['logical_phase_counts']['warm_cache']):
        raise ValueError('Diagnostic must preserve complete preceding warm and the continuous prefix, without thinning')
    for row, record in zip(retained, records):
        if row['session_id'] != record['replay_id'] or digest(row['payload']) != record['payload_sha256']:
            raise ValueError('Diagnostic changed payload identity/order')
    secure_directory(out)
    data=out/'toolagent';secure_directory(data)
    with (data/'payload_trace.jsonl').open('xb') as stream:
        for row in retained:stream.write(canonical_bytes(row))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    source_indices={r['source_data_row_index_zero_based'] for r in records}
    manifest.update(window_end_ms=end,selected_requests=len(retained),records=records,
        logical_phase_counts=dict(collections.Counter(r['logical_phase'] for r in records)),
        payload_trace_sha256=sha(data/'payload_trace.jsonl'),
        source_input_tokens_total=sum(r['source_input_tokens'] for r in records),
        source_output_tokens_total=sum(r['source_output_tokens'] for r in records),
        parallel_decode_length_corrections=[r for r in manifest['parallel_decode_length_corrections'] if r['source_index'] in source_indices],
        shared_prefix_validation={'mode':'inherited_bit_identical_payloads_from_verified_full_window',
            'parent_manifest_sha256':sha(parent/'manifest.json'),
            'parent_shared_prefix_checks':manifest.get('shared_prefix_token_sha_checks'), 'new_tokenization_checks':0},
        shared_prefix_token_sha_checks=0,
        parent_full_payload_sha256=sha(parent/'payload_trace.jsonl'),
        parent_full_manifest_sha256=sha(parent/'manifest.json'),
        transformation='Original continuous first60s source warm plus next30s; unchanged payloads/session order/clock/prefix')
    write_new_json(data/'manifest.json',manifest)
    validate_inputs(data/'payload_trace.jsonl',data/'manifest.json')
    provenance={'classification':'confirmed_CPU_input_derivation_not_GPU_measurement',
        'parent':str(parent),'parent_payload_sha256':sha(parent/'payload_trace.jsonl'),
        'parent_manifest_sha256':sha(parent/'manifest.json'),'source_seconds':90,
        'requests':len(retained),'phase_counts':manifest['logical_phase_counts'],
        'all_payloads_byte_equivalent':True,'original_prefix_relation_preserved':True,
        'new_tokenization_checks':0,'GPU_job_launched':False,
        'script_sha256':sha(Path(__file__))}
    write_new_json(out/'provenance.json',provenance)
    return provenance


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.parent,args.out)))


if __name__=='__main__':main()
