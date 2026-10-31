"""CPU-only original-model pressure accounting from frozen raw requests/gauges."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.audit_nextgen_performance import audit as audit_performance
from scripts.audit_nextgen_application_metrics import audit as audit_application
from scripts.audit_nextgen_validation import _mirror_paths
from scripts.controller.official_tokenization import OfficialJSONTokenizer
from scripts.controller.durable import digest,write_new_json


def load(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def records(path):return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def audit(manifest_path,performance_path,observations,tokenizer_path):
    perf=load(performance_path)
    recomputed=audit_performance(manifest_path,source_repo=Path('/home/mingasa/projects/xllm_quant_kvcache-agent-nextgen-native-kv-20261027'))
    assert recomputed==perf and perf['case']=='long-prefix-c16'
    manifest=load(manifest_path);_,root=_mirror_paths(manifest)
    ticket=load(root/'ticket.json');request=ticket['request'];arm=request['arms'][0]
    assert request['scheduling_profile']=='mixed8192_chunk2048_v1'
    output=root/'output'/arm;identity=load(output/'container_identity.json')
    observation=audit_application(observations,performance_path)
    response=load(observations/'ORIGINAL_RESPONSE.json');intent=load(observations/'LOCAL_INTENT.json')
    for name,key in [('OBSERVER_SOURCE.py','source_sha256'),('OUTER_SOURCE.py','outer_source_sha256'),('INNER_SOURCE.py','inner_source_sha256')]:
        assert sha(observations/name)==intent[key]
    assert response['container_id']==identity['id'] and response['request_sha256']==ticket['request_sha256']
    assert response['parser_sha256']==ticket['assets_sha256']['runtime/kv_controller/service_metrics.py']
    tokenizer=OfficialJSONTokenizer(tokenizer_path)
    source_path=root/'input/production/long_prefix16x2.json'
    source={(s['session_id'],i):p for s in load(source_path)['data'] for i,p in enumerate(s['payloads'])}
    expected={key:tokenizer.count(payload) for key,payload in source.items()}
    metric_path=output/'measured/artifacts/profile_export.jsonl';metric=records(metric_path)
    assert len(expected)==len(metric)==32
    for row in metric:
        meta=row['metadata'];key=(meta['conversation_id'],meta['turn_index'])
        value=row['metrics']['input_sequence_length']
        value=value['value'] if isinstance(value,dict) else value
        assert value==expected[key],('Official JSON token count differs',key,value,expected[key])
    page=request['service']['page_size'];single=max(math.ceil((expected[k]+p['max_tokens'])/page)+1 for k,p in source.items())
    start=perf['profile']['worker_observation_started_monotonic_ns'];end=perf['profile']['worker_observation_ended_monotonic_ns']
    samples=records(observations/'application_metrics.jsonl')[1:]
    during=[row for row in samples if start<=row['monotonic_ns']<=end]
    slots=perf['lifecycle']['Full_KV_physical_token_slots'];assert slots%page==0;physical=slots//page
    complete=[row for row in during if not row['errors']]
    assert complete and all(row['values']['num_used_blocks']<=physical for row in complete)
    qualifying=[row for row in complete if row['values']['num_running_requests']>=2 and row['values']['num_used_blocks']>single]
    selected=max(qualifying,key=lambda row:row['values']['num_used_blocks']) if qualifying else None
    peak=max(complete,key=lambda row:row['values']['num_used_blocks'])
    semantics=Path(__file__).parent/'active-FullKV-gauge-source-semantics-a1.json'
    points=[{'elapsed_since_measured_worker_start_seconds':(row['monotonic_ns']-start)/1e9,
             'monotonic_ns':row['monotonic_ns'],'values':row['values'],'errors':row['errors']} for row in during]
    return {'schema':'actual-mixed-pressure-residency-audit-v1','run_id':perf['run_id'],'arm':arm,
        'all32_actual_prompt_counts_match_official_JSON_tokenizer':True,
        'actual_prompt_tokens_total':sum(expected.values()),'actual_prompt_token_range':[min(expected.values()),max(expected.values())],
        'largest_single_request_pages_including_original_cap_and_padding':single,
        'physical_FullKV_pages':physical,'physical_FullKV_token_slots':slots,
        'multiple_request_residency_observed_in_same_sample':selected is not None,
        'qualifying_same_sample':None if selected is None else {'monotonic_ns':selected['monotonic_ns'],'values':selected['values']},
        'max_effective_used_FullKV_pages':peak['values']['num_used_blocks'],
        'values_at_max_used_pages_sample':peak['values'],
        'max_running_requests_independent_sample_peak':max(row['values']['num_running_requests'] for row in complete),
        'max_waiting_requests_independent_sample_peak':max(row['values']['num_waiting_requests'] for row in complete),
        'observed_used_pages_exceed_same_budget_BF16_physical3868_pages':peak['values']['num_used_blocks']>3868,
        'sample_points':points,'application_audit':observation,
        'request_identity':perf['request_identity'],'source_SHA_manifest_sha256':sha(manifest_path),
        'performance_audit_sha256':sha(performance_path),'actual_metric_raw_sha256':sha(metric_path),
        'source_payloads_sha256':sha(source_path),'tokenizer_json_sha256':sha(tokenizer_path/'tokenizer.json'),
        'source_semantics_proof_sha256':sha(semantics),'auditor_source_sha256':sha(Path(__file__)),
        'GPU_or_remote_calls':False,'max_admission_or_HBM_traffic_or_SLO_certified':False,
        'limitations':['Sequential2s sampled application gauges, not atomic snapshots; peak metrics may be from different times.',
            'Used pages include allocator padding and are distinct from composite prefix index entries or exact live payload tokens.',
            'Only displayed counter span is observed; four-token scientific-display rounding mismatch stays false in application audit.',
            'A capacity observation is bounded by this fixed budget/workload; it is not maximum successful service concurrency.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['sha-manifest','performance-audit','observations','tokenizer','out']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=audit(args.sha_manifest,args.performance_audit,args.observations,args.tokenizer)
    write_new_json(args.out,result)
    print(json.dumps({k:result[k] for k in ['run_id','all32_actual_prompt_counts_match_official_JSON_tokenizer',
        'physical_FullKV_pages','max_effective_used_FullKV_pages','max_running_requests_independent_sample_peak',
        'multiple_request_residency_observed_in_same_sample']}))


if __name__=='__main__':main()
