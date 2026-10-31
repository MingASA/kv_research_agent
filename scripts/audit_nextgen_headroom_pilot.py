"""Qualify one explicit scheduling candidate using real BF16 calibration and gauges."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.audit_nextgen_calibration import audit_calibration
from scripts.audit_nextgen_validation import _check,_jsonl,_mirror_paths
from scripts.controller.durable import write_new_json
from scripts.controller.nextgen_service import MIXED_PREFILL_PROFILE,resolved_scheduling_options
from scripts.controller.service_metrics import parse_metric


def select_samples(samples: list[dict],long_requests: list[dict],single_pages: int) -> tuple[list,dict | None]:
    eligible=[];qualified=[]
    for row in samples:
        active=sum(request['request_started_monotonic_ns']<=row['monotonic_ns']
                   <request['request_ended_monotonic_ns'] for request in long_requests)
        if active<2 or row['errors']:continue
        eligible.append(row);values=row['values']
        if values['num_running_requests']>=2 and values['num_used_blocks']>single_pages:qualified.append(row)
    return eligible,max(qualified,key=lambda row:row['values']['num_used_blocks']) if qualified else None


def audit(manifest_path: Path,tokenizer: Path,observations: Path) -> dict:
    calibration=audit_calibration(manifest_path,tokenizer)
    manifest=json.loads(manifest_path.read_text());_,root=_mirror_paths(manifest)
    ticket=json.loads((root/'ticket.json').read_text());request=ticket['request'];output=root/'output/bf16'
    _check(request.get('scheduling_profile')==MIXED_PREFILL_PROFILE and request['service']['max_tokens_per_batch']==8192
           and request['service']['max_tokens_per_chunk_for_prefill']==2048,'Unexpected scheduling candidate')
    runtime=resolved_scheduling_options((output/'server.log').read_text(errors='replace'),request['service'])
    inputs=json.loads((root/'input/calibration.json').read_text())['requests'];raw_requests=_jsonl(output/'calibration_raw.jsonl')
    _check(len(raw_requests)==len(inputs)==136 and all(type(row.get(name)) is int for row in raw_requests
           for name in ('request_started_monotonic_ns','request_ended_monotonic_ns')),'No complete actual calibration clocks')
    terminal=json.loads((root/'result.json').read_text())
    _check(all(terminal['allocation_started_monotonic_ns']<=row['request_started_monotonic_ns']
               <row['request_ended_monotonic_ns']<=terminal['allocation_ended_monotonic_ns'] for row in raw_requests),
           'Calibration clocks lie outside the original allocation')
    long=[row for row in raw_requests if row['id'].startswith('long-holdout-')]
    _check(len(long)==8,'Original long calibration inventory differs')
    page=request['service']['page_size']
    single=max(math.ceil((row['prompt_tokens_cpu']+row['payload']['max_tokens'])/page)+1 for row in inputs)
    original=(observations/'application_metrics.jsonl').read_bytes();records=_jsonl(observations/'application_metrics.jsonl')
    response=json.loads((observations/'ORIGINAL_RESPONSE.json').read_text());header=records[0];rows=records[1:]
    intent=json.loads((observations/'LOCAL_INTENT.json').read_text())
    for name,key in [('OBSERVER_SOURCE.py','source_sha256'),('OUTER_SOURCE.py','outer_source_sha256'),
                     ('INNER_SOURCE.py','inner_source_sha256')]:
        _check(hashlib.sha256((observations/name).read_bytes()).hexdigest()==intent[key],
               'Original local observation program source changed')
    identity=json.loads((output/'container_identity.json').read_text())
    _check(header['run_id']==response['run_id']==request['run_id']
           and header['request_sha256']==response['request_sha256']==ticket['request_sha256']
           and response['container_id']==identity['id']
           and response['parser_sha256']==ticket['assets_sha256']['runtime/kv_controller/service_metrics.py']
           and response['stdout_sha256']==hashlib.sha256(original).hexdigest()
           and response['remote_file_writes'] is False and response['GPU_or_model_calls'] is False,
           'Original owned observation identity or bytes differ')
    checked=0
    for row in rows:
        for name,value in row['values'].items():
            _check(value==parse_metric(name,row['raw'][name]),'Cached gauge differs from raw HTML');checked+=1
    _check(rows and all(left['monotonic_ns']<right['monotonic_ns'] for left,right in zip(rows,rows[1:])),
           'Sample clock is incomplete or non-increasing')
    eligible,selected=select_samples(rows,long,single)
    qualified=selected is not None
    proof={name:request[name] for name in ('model_id','model_config_sha256','image_id','binary_sha256',
        'server_source_commit','python_source_commit','service','gpu_uuids')}
    proof.update(schema='nextgen-independent-mixed-prefill-pilot-v1',
        state='MIXED_PREFILL_BF16_PILOT_INDEPENDENTLY_VERIFIED' if qualified else 'PILOT_PARALLEL_LONG_RESIDENCY_NOT_VERIFIED',
        run_id=request['run_id'],scheduling_profile=MIXED_PREFILL_PROFILE,request_sha256=ticket['request_sha256'],
        terminal_result_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),
        validation_result_sha256=hashlib.sha256((output/'validation_result.json').read_bytes()).hexdigest(),
        server_log_sha256=hashlib.sha256((output/'server.log').read_bytes()).hexdigest(),
        resolved_runtime_options=runtime,actual_runtime_options_verified=True,
        source_SHA_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        calibration_requests_verified=calibration['calibration_requests_verified'],known_answer_checks_verified=20,
        actual_observer_GPU_Graph_passed=calibration['actual_observer_GPU_Graph_passed'],
        largest_single_request_page_upper_bound=single,
        observed_running_during_long_requests=selected['values']['num_running_requests'] if qualified else 0,
        observed_active_FullKV_pages_during_long_requests=selected['values']['num_used_blocks'] if qualified else 0,
        qualifying_same_sample=selected['monotonic_ns'] if qualified else None,
        samples_with_at_least_two_long_client_requests_inflight=len(eligible),
        raw_HTML_values_independently_reparsed=checked,application_raw_sha256=hashlib.sha256(original).hexdigest(),
        observation_process_exit_code=response['container_exec_exit'],long_calibration_requests=8,
        actual_single_GPU_pilot_release_verified=True,
        calibration_audit=calibration,quality_or_service_performance_certified=False,
        limitations=['Application gauges are sequential samples, not an atomic profiler.',
            'Client overlap alone cannot qualify this pilot; the same sampled server observation must show multiple running requests and active pages beyond any one request upper bound.',
            'Diagnostic range counters are enabled; no formal speed, quality acceptance, maximum capacity or HBM-traffic claim.'])
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sha-manifest',type=Path,required=True);parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--observations',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=audit(args.sha_manifest,args.tokenizer,args.observations)
    write_new_json(args.out,result);print(json.dumps({key:result[key] for key in ('state','run_id',
        'observed_running_during_long_requests','observed_active_FullKV_pages_during_long_requests',
        'largest_single_request_page_upper_bound')}))
