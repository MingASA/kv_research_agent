"""Recompute per-request native prefix-cache usage from SHA-verified raw SSE.

This measures the service's reported cached-token counter. Under chunked prefill
it can include the same request's already-computed chunks. It cannot certify
cross-request reuse, physical HBM traffic or maximum admission capacity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.audit_nextgen_performance import audit as audit_performance
from scripts.audit_nextgen_validation import _check, _mirror_paths
from scripts.controller.durable import write_new_json


def cache_row(raw: dict) -> dict:
    usages=[]
    for response in raw['responses']:
        for packet in response.get('packets',[]):
            if packet.get('name')!='data' or packet.get('value')=='[DONE]':continue
            item=json.loads(packet['value'])
            if item.get('usage') is not None:usages.append(item['usage'])
    _check(len(usages)==1,'Expected exactly one native final usage per raw request')
    usage=usages[0];prompt=usage['prompt_tokens'];cached=usage['prompt_tokens_details']['cached_tokens']
    _check(type(prompt) is int and type(cached) is int and 0<=cached<=prompt,
           'Native cached tokens are missing, invalid or exceed actual input tokens')
    meta=raw['metadata']
    _check(meta['benchmark_phase']=='profiling' and type(meta['turn_index']) is int,
           'Unexpected raw request phase or turn identity')
    return {'request_id':meta['x_request_id'],'conversation_id':meta['conversation_id'],
            'turn_index':meta['turn_index'],'prompt_tokens':prompt,'cached_tokens':cached}


def totals(rows: list[dict]) -> dict:
    prompt=sum(row['prompt_tokens'] for row in rows);cached=sum(row['cached_tokens'] for row in rows)
    return {'requests':len(rows),'prompt_tokens':prompt,'cached_tokens':cached,
            'requests_with_nonzero_cached_tokens':sum(row['cached_tokens']>0 for row in rows),
            'cached_input_fraction':cached/prompt if prompt else None}


def audit(manifest_path: Path,performance_path: Path,source_repo: Path) -> dict:
    performance=audit_performance(manifest_path,source_repo=source_repo)
    _check(performance==json.loads(performance_path.read_text()),'Provided performance audit differs from full raw re-audit')
    manifest=json.loads(manifest_path.read_text());_,root=_mirror_paths(manifest)
    request=json.loads((root/'ticket.json').read_text())['request'];arm=request['arms'][0]
    raw_path=root/'output'/arm/'measured/artifacts/profile_export_raw.jsonl'
    rows=[cache_row(json.loads(line)) for line in raw_path.read_bytes().split(b'\n') if line]
    _check(len(rows)==performance['successful_requests'] and len({row['request_id'] for row in rows})==len(rows),
           'Missing or duplicate actual native cache observations')
    summary=totals(rows)
    _check(summary['prompt_tokens']==round(performance['profile']['isl_mean']*len(rows)),
           'Raw usage does not cover the complete audited input cohort')
    # Bind the manually reviewed native API mapping to the actual binary's
    # committed C++ source and verify that the deployed Python source agrees.
    snippets={
        'xllm/core/framework/request/request.cpp':[
            'usage.num_cached_tokens = static_cast<int32_t>(num_prefix_cache_tokens_);',
            'record_prefix_cache_hit_metrics(state_.prompt_tokens.size(),',
            'std::max(num_prefix_cache_tokens_, num_prefix_cache_tokens);'],
        'xllm/api_service/utils.h':['prompt_tokens_details->set_cached_tokens(usage.num_cached_tokens);'],
        'xllm/core/framework/request/sequence.cpp':['kv_state_.shared_tokens_num()'],
        'xllm/core/scheduler/scheduler_policy.cpp':['request->record_num_prefix_cache_tokens();',
            '// Re-match at every chunk boundary.']}
    sources={}
    for name,needles in snippets.items():
        versions=[subprocess.check_output(['git','-C',str(source_repo),'show',commit+':'+name])
                  for commit in (request['server_source_commit'],request['python_source_commit'])]
        _check(versions[0]==versions[1] and all(needle.encode() in versions[0] for needle in needles),
               'Native cached-token API mapping source changed: '+name)
        sources[name]=hashlib.sha256(versions[0]).hexdigest()
    return {'schema':'nextgen-independent-native-API-cache-usage-audit-v2','run_id':request['run_id'],
        'arm':arm,'case':performance['case'],'classification':'confirmed_native_API_cached_counter_only',
        'all_measured_requests_independently_verified':True,'summary':summary,
        'by_turn_index':{str(turn):totals([row for row in rows if row['turn_index']==turn])
                         for turn in sorted({row['turn_index'] for row in rows})},
        'per_request':sorted(rows,key=lambda row:(row['conversation_id'],row['turn_index'])),
        'raw_SSE_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        'source_SHA_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'performance_audit_path':str(performance_path),
        'performance_audit_sha256':hashlib.sha256(performance_path.read_bytes()).hexdigest(),
        'native_CXX_API_mapping_source_sha256':sources,'binary_source_commit':request['server_source_commit'],
        'source_equivalent_CXX_commit':request['python_source_commit'],
        'GPU_or_model_calls':False,'remote_file_writes':False,
        'cross_request_prefix_reuse_certified':False,
        'limitations':['Native cached_tokens can include the same request own computed chunks after prefix re-match; it is not a cross-request hit rate.',
            'The request maximum shared-prefix field and once-only initial-admission hit counter have different semantics.',
            'Hits may include external warmup or earlier requests; turn 0 is not assumed to be a cold-cache baseline.',
            'No instantaneous allocator pressure, maximum capacity or per-kernel speed is inferred.']}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sha-manifest',type=Path,required=True)
    parser.add_argument('--performance-audit',type=Path,required=True)
    parser.add_argument('--source-repo',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=audit(args.sha_manifest,args.performance_audit,args.source_repo)
    write_new_json(args.out,result);print(json.dumps(result['summary']))


if __name__=='__main__':main()
