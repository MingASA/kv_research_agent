"""Audit the original failed NVFP4 load, retaining its full32-request denominator.

This is a separate failure schema. It never qualifies a successful production
benchmark and never changes the original FAILED ticket/result or any raw bytes.
"""
from collections import Counter
from datetime import datetime
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.audit_native_results import _value,_close,_percentile
from scripts.audit_nextgen_validation import _mirror_paths,_stream
from scripts.controller.durable import digest,write_new_json
from scripts.controller.nextgen_performance import verify_frozen_case,aiperf_command,validate_cli_protocol_proof
from scripts.controller.official_tokenization import OfficialJSONTokenizer
from scripts.controller.production import audit_requests

EXPECTED='nextgen-nvfp4-qwen38-mixed8192-prefix32-r0-20261030yc'


def load(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def records(path):
    raw=path.read_bytes();assert raw.endswith(b'\n')
    return [json.loads(line) for line in raw.split(b'\n') if line]


def validate_inventory(rows,source):
    if len(source)!=32 or len(rows)!=32:raise ValueError('All32 offered requests must remain in denominator')
    keys=[(row['metadata']['conversation_id'],row['metadata']['turn_index']) for row in rows]
    ids=[row['metadata']['x_request_id'] for row in rows]
    if len(set(keys))!=32 or set(keys)!=set(source) or len(set(ids))!=32:
        raise ValueError('Missing/duplicated/unknown original request or session turn')
    if any(r['metadata']['benchmark_phase']!='profiling' or r['metadata']['was_cancelled'] is not False
           or r['metadata']['context_overflow_skip'] is not False for r in rows):
        raise ValueError('Unreviewed cancelled/skipped phase')
    bad=[r for r in rows if r.get('error')]
    if len(bad)!=1 or bad[0]['error'].get('type')!='TimeoutError':
        raise ValueError('This audit admits only the original one-timeout failure')
    return bad[0]


def parse_stream(row):
    done=0;finish=None;usage=None;content_times=[]
    for response in row['responses']:
        for packet in response['packets']:
            if packet.get('name')!='data':continue
            value=packet['value']
            if value=='[DONE]':done+=1;continue
            item=json.loads(value);assert not item.get('error')
            if item.get('usage'):usage=item['usage']
            for choice in item.get('choices',[]):
                finish=choice.get('finish_reason') or finish
                content=choice.get('delta',{}).get('content')
                if isinstance(content,str) and content:content_times.append(response['perf_ns'])
    assert all(a<=b for a,b in zip(content_times,content_times[1:]))
    gaps=[(b-a)/1e6 for a,b in zip(content_times,content_times[1:])]
    return done,finish,usage,content_times,gaps


def audit(manifest_path,tokenizer_path):
    manifest=load(manifest_path);base,root=_mirror_paths(manifest);assert manifest['run_id']==EXPECTED
    for item in manifest['files']:
        path=base/item['base_rel'];assert path.stat().st_size==item['size'] and sha(path)==item['remote_sha256']==item['snapshot_sha256']
    ticket=load(root/'ticket.json');request=ticket['request'];terminal=load(root/'result.json')
    assert ticket['request_sha256']==terminal['request_sha256']==digest(request)
    assert terminal['status']=='FAILED' and terminal['gpu_release_verified'] is True and terminal['live_owned_containers']==[]
    assert terminal['arm_results']==[] and request['arms']==['nvfp4'] and not (root/'COMPLETE').exists()
    for name,expected in ticket['assets_sha256'].items():assert sha(root/name)==expected
    assert request['scheduling_profile']=='mixed8192_chunk2048_v1' and request['performance_evaluation']['case']=='long-prefix-c16'
    output=root/'output/nvfp4';case=verify_frozen_case(root/'input/production','long-prefix-c16',model_id=request['model_id'])
    proof=load(root/'input/full_CLI_protocol_CPU_audit.json');validate_cli_protocol_proof(proof,require_session_limit=True)
    assert sha(root/'input/full_CLI_protocol_CPU_audit.json')==request['performance_evaluation']['full_CLI_protocol_CPU_audit_sha256']
    command=aiperf_command(request['model_id'],Path(request['remote_output'])/'input/production/long_prefix16x2.json',
        Path('/results/measured/artifacts'),requests=32,concurrency=16,sessions=16)
    assert load(output/'measured/command.json')==command
    accepted=load(output/'measured/aiperf_cli_CPU_acceptance.json')
    assert accepted['actual_Cyclopts_CLIConfig_accepted'] and accepted['aiperf_version']=='0.13.0'
    assert accepted['cli_config_sha256']==request['performance_evaluation']['aiperf_cli_config_sha256']
    assert (accepted['request_count'],accepted['conversation_num'],accepted['concurrency'],accepted['warmup_request_count'])==(32,16,16,None)
    source={(row['session_id'],i):payload for row in load(root/'input/production/long_prefix16x2.json')['data'] for i,payload in enumerate(row['payloads'])}
    metrics=records(output/'measured/artifacts/profile_export.jsonl');bad=validate_inventory(metrics,source)
    assert (bad['metadata']['conversation_id'],bad['metadata']['turn_index'])==('longdoc-gov_report-59',0)
    raw=records(output/'measured/artifacts/profile_export_raw.jsonl');assert len(raw)==32
    by_id={row['metadata']['x_request_id']:row for row in metrics};seen=set();success=[];failed=None;all_gaps=[]
    tokenizer=OfficialJSONTokenizer(tokenizer_path);expected_counts={key:tokenizer.count(payload) for key,payload in source.items()}
    for row in raw:
        meta=row['metadata'];identifier=meta['x_request_id'];assert identifier in by_id and identifier not in seen;seen.add(identifier)
        expected_meta=by_id[identifier]['metadata'];assert all(meta[k]==expected_meta[k] for k in ['conversation_id','turn_index','benchmark_phase'])
        key=(meta['conversation_id'],meta['turn_index']);assert row['payload']==source[key] and row['payload'].get('ignore_eos',False) is False
        done,finish,usage,times,gaps=parse_stream(row);all_gaps.extend(gaps)
        assert terminal['allocation_started_monotonic_ns']<=row['start_perf_ns']<terminal['allocation_ended_monotonic_ns']
        if by_id[identifier].get('error'):
            assert row.get('error',{}).get('type')=='TimeoutError' and done==0 and usage is None
            failed={'metadata':meta,'error':by_id[identifier]['error'],'HTTP_status':row['status'],
                'raw_stream_done_count':done,'final_usage_available':False,'nonempty_content_packets_observed':len(times),
                'actual_elapsed_seconds':(meta['request_end_ns']-meta['request_start_ns'])/1e9,
                'partial_output_not_counted_as_successful_completion':True}
            continue
        assert row.get('status')==200 and not row.get('error') and done==1 and finish in ('stop','length') and usage
        metric=by_id[identifier]['metrics'];prompt=_value(metric,'input_sequence_length');out=_value(metric,'output_token_count')
        assert prompt==expected_counts[key] and prompt.is_integer() and out.is_integer() and 1<=out<=source[key]['max_tokens']
        assert usage=={'prompt_tokens':int(prompt),'completion_tokens':int(out),'total_tokens':int(prompt+out)}
        for name,wanted in [('usage_prompt_tokens',prompt),('usage_completion_tokens',out),('usage_total_tokens',prompt+out)]:_close(_value(metric,name),wanted,'Successful raw SSE/metric usage differs')
        if out>1:_close(_value(metric,'inter_token_latency'),_value(metric,'decode_duration')/(out-1),'Conditional request ITL algebra differs')
        success.append({'metadata':meta,'metrics':metric,'finish':finish})
    assert seen==set(by_id) and len(success)==31 and failed is not None
    for session,turn in source:
        if turn:
            first=next(row for row in metrics if row['metadata']['conversation_id']==session and row['metadata']['turn_index']==0)
            second=next(row for row in metrics if row['metadata']['conversation_id']==session and row['metadata']['turn_index']==turn)
            assert first['metadata']['request_end_ns']<=second['metadata']['request_start_ns']
    summary=load(output/'measured/artifacts/profile_export_aiperf.json')
    assert summary['is_complete'] is True and summary['was_cancelled'] is False
    assert len(summary['error_summary'])==1 and summary['error_summary'][0]['count']==1 and summary['error_summary'][0]['error_details']['type']=='TimeoutError'
    duration=(datetime.fromisoformat(summary['end_time'])-datetime.fromisoformat(summary['start_time'])).total_seconds();assert duration>600
    warm=audit_requests(output/'warmup/artifacts',root/'input/production/warmup_long2.json',2,0,allow_multi_turn=True)
    validation=load(output/'validation_result.json');dataset=load(root/'input/validation.json');cases=dataset['cases'];known=records(output/'validation_raw.jsonl')
    counts={row['id']:row['prompt_tokens'] for row in load(root/'input/validation_tokenization_CPU.json')['cases']}
    expected=cases[:4]+cases[4:]+cases[:4];assert len(known)==len(expected)==20
    answers=[_stream(row,case,counts[case['id']],request['model_id']) for row,case in zip(known,expected)];assert answers[:4]==answers[-4:]
    server=(output/'server.log').read_text();assert 'NATIVE_KV_GRAPH_REPLAY' in server
    backend=[json.loads(line.split('NATIVE_KV_BACKEND ',1)[1]) for line in server.splitlines() if 'NATIVE_KV_BACKEND ' in line][-1]
    allocation=[json.loads(line.split('KV_ALLOCATION ',1)[1]) for line in server.splitlines() if 'KV_ALLOCATION ' in line][-1]
    assert backend==validation['native_backend'] and allocation==validation['native_allocation']
    assert backend['class']=='FlashInferNVFP4Backend' and backend['full_BF16_KV_shadow'] is False and backend['scale_views_alias_resident_pages'] is True
    assert allocation['token_capacity']==880320
    outputs=sum(int(_value(row['metrics'],'output_token_count')) for row in success)
    conditional={}
    for metric,label in [('time_to_first_token','ttft'),('request_latency','e2e'),('inter_token_latency','request_mean_itl')]:
        values=[_value(row['metrics'],metric) for row in success if metric!='inter_token_latency' or _value(row['metrics'],'output_token_count')>1]
        conditional[label+'_ms']={name:_percentile(values,q) for name,q in [('p50',.5),('p95',.95),('p99',.99)]}
    return {'schema':'independent-original-failed-load-audit-v1','run_id':EXPECTED,'arm':'nvfp4','original_status':'FAILED',
        'all_offered_requests_completed_successfully':False,'offered_requests':32,'successful_requests':31,'failed_requests':1,
        'success_fraction':31/32,'failure_type':'one600s_client_TimeoutError','failed_request':failed,
        'full_AIPerf_observation_seconds':duration,'successful_completions_per_full_observation_second':31/duration,
        'verified_successful_output_tokens':outputs,'verified_complete_output_tokens_per_full_observation_second':outputs/duration,
        'successful_response_only_conditional_quantiles':conditional,'conditional_quantiles_are_not_full_cohort_latency':True,
        'all32_payloads_and_session_turns_verified_once':True,'all31_successful_SSE_usage_and_official_token_counts_verified':True,
        'all_offered_prompt_tokens_from_official_JSON':sum(expected_counts.values()),'all32_denominator_retained':True,
        'warmup_separate_two_successful_requests_verified':warm['profiling_records'],
        'known20_ABA_Graph_and_native_NVFP4_model_verified':True,'native_allocation':allocation,
        'request_identity':{k:request[k] for k in ['model_id','model_config_sha256','image_id','binary_sha256','server_source_commit','python_source_commit','gpu_uuids','service','operators','graph_mode','prefix_cache']},
        'physical_slots_are_not_observed_active_residency':True,'auxiliary_application_samples_unavailable_after_SSH_reply_loss':True,
        'quality_snapshot_sha256':request['performance_evaluation']['quality_scores_sha256'],
        'source_SHA_manifest_sha256':sha(manifest_path),'original_result_sha256':sha(root/'result.json'),
        'raw_export_sha256':sha(output/'measured/artifacts/profile_export_raw.jsonl'),'auditor_source_sha256':sha(Path(__file__)),
        'gpu_released':True,'GPU_or_remote_calls':False,'retry_or_original_status_mutation':False,
        'production_acceptance_or_full_success_or_SLO_claim':False,
        'scope':'Observed fixed C16/600s load failed. Separate failure-aware rates include the full observation and32offered denominator; no failed partial output credited. Successful-response tails are conditional and cannot substitute for the missing successful request.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['sha-manifest','tokenizer','out']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=audit(args.sha_manifest,args.tokenizer);write_new_json(args.out,result)
    print(json.dumps({k:result[k] for k in ['run_id','offered_requests','successful_requests','failed_requests','full_AIPerf_observation_seconds','successful_completions_per_full_observation_second','verified_complete_output_tokens_per_full_observation_second']}))
