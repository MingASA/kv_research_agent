"""Export full service tails and GPU-time costs after re-auditing raw mirrors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from scripts.audit_nextgen_performance import audit
from scripts.audit_nextgen_validation import _check,_mirror_paths
from scripts.controller.durable import write_new_json


def tabulate(registry_path: Path,source_repo: Path,output: Path) -> None:
    registry=json.loads(registry_path.read_text());rows=[];provenance=[]
    for source in registry:
        manifest_path=Path(source['sha_manifest']);audit_path=Path(source['performance_audit'])
        result=audit(manifest_path,source_repo=source_repo)
        _check(result==json.loads(audit_path.read_text()),'Stored result differs from raw re-audit')
        manifest=json.loads(manifest_path.read_text());_,root=_mirror_paths(manifest)
        terminal=json.loads((root/'result.json').read_text());request=json.loads((root/'ticket.json').read_text())['request']
        _check(len(request['gpu_uuids'])==1,'GPU-time costs here require an actual single-GPU run')
        intervals=terminal['allocation_intervals_ns']
        lease=sum((interval['end_ns']-interval['start_ns'])/1e9 for interval in intervals)
        profile=result['profile'];warm=result['external_warmup'];count=result['successful_requests']
        log=(root/'output'/result['arm']/'server.log').read_text(errors='replace')
        chunks={int(value) for value in re.findall(r'max_tokens_per_chunk_for_prefill: (\d+)',log)}
        _check(len(chunks)==1,'Actual C++ chunk budget is missing or ambiguous')
        chunk=chunks.pop()
        _check(chunk==request['service'].get('max_tokens_per_chunk_for_prefill',request['service']['max_tokens_per_batch']),
               'Actual C++ chunk budget differs from the declared profile/default')
        remainder=lease-profile['observation_seconds']-warm['observation_seconds']
        _check(remainder>=0,'Benchmark observation exceeds original resource lease')
        row={'run_id':result['run_id'],'case':result['case'],'arm':result['arm'],
             'requests':count,'concurrency':result['concurrency'],
             'batch_token_budget':request['service']['max_tokens_per_batch'],
             'actual_logged_prefill_chunk_tokens':chunk,
             'max_batch':request['service']['max_batch'],'linear_state_slots':request['service']['linear_state_slots'],
             'actual_prompt_tokens':round(profile['isl_mean']*count),
             'successful_request_fraction':count/result['offered_requests'],
             'stop_requests':profile['finish_reason_counts'].get('stop',0),
             'length_requests':profile['finish_reason_counts'].get('length',0),
             'warmup_seconds':warm['observation_seconds'],'GPU_lease_seconds':lease,
             'other_lease_seconds':remainder,
             'measured_GPU_seconds_per_successful_request':profile['observation_seconds']/count,
             'full_lease_GPU_seconds_per_successful_request':lease/count,
             'max_successful_admission_capacity_measured':False,'SLO_goodput_measured':False,
             'production_quality_accepted':False,'binary_sha256':request['binary_sha256'],
             'server_source_commit':request['server_source_commit'],'python_source_commit':request['python_source_commit'],
             'GPU_UUID':request['gpu_uuids'][0]}
        for key in ('observation_seconds','request_rps','output_tps','actual_output_tokens','isl_mean',
                    'osl_mean','osl_min','osl_max','ttft_p50_ms','ttft_p95_ms','ttft_p99_ms',
                    'itl_p50_ms','itl_p95_ms','itl_p99_ms','e2e_p50_ms','e2e_p95_ms','e2e_p99_ms'):
            row[key]=profile[key]
        rows.append(row);provenance.append({**source,'run_id':result['run_id'],
            'sha_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'performance_audit_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest()})
    _check(rows and len({row['run_id'] for row in rows})==len(rows),'Duplicate or empty service registry')
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    with (output/'service_observations.csv').open('x',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    write_new_json(output/'provenance.json',{'schema':'nextgen-service-tabular-export-v1',
        'input_registry_sha256':hashlib.sha256(registry_path.read_bytes()).hexdigest(),'sources':provenance,
        'every_result_reaudited_from_complete_SHA_mirror':True,
        'GPU_time_cost_scope':'One GPU; measured interval and entire resource lease reported separately.',
        'other_lease_seconds_scope':'Includes cold operator/JIT/model start, correctness gates, export and cleanup; not isolated writer cost.',
        'monetary_price_or_HBM_bandwidth_claim':False,'rows':rows})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--source-repo',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();tabulate(args.registry,args.source_repo,args.out)
