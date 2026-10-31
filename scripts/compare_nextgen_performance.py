"""Compare two independently audited actual-model production observations.

Counts, source, service and physical GPU must match. Different declared native
kernels remain explicit deployment choices; their effect is not dtype-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.controller.durable import write_new_json


def compare(reference: dict,candidate: dict) -> dict:
    schema='kv-nextgen-independent-production-performance-audit-v1'
    for audit in (reference,candidate):
        if (audit.get('schema')!=schema or audit.get('actual_model_and_Graph_and_raw_service_performance_verified') is not True
                or audit.get('all_offered_requests_completed') is not True
                or audit['offered_requests']!=audit['successful_requests']):
            raise ValueError('Both sides require independent complete actual-model/raw/Graph audits')
    if reference['arm']!='bf16' or candidate['arm'] not in ('fp8_e4m3','nvfp4'):
        raise ValueError('This comparison requires BF16 and an explicit native low-precision deployment')
    fields=('model_id','model_config_sha256','image_id','binary_sha256','server_source_commit','python_source_commit',
            'gpu_uuids','service','graph_mode','prefix_cache')
    for field in fields:
        if reference['request_identity'][field]!=candidate['request_identity'][field]:
            raise ValueError('Unmatched production deployment: '+field)
    for field in ('case','concurrency','input_manifest_sha256','offered_requests'):
        if reference[field]!=candidate[field]:raise ValueError('Different real workload/demand: '+field)
    for field in ('profile_session_ids_sha256','profile_prompt_counts_sha256'):
        if reference['profile'][field]!=candidate['profile'][field]:raise ValueError('Actual request IDs/input counts differ')
    def ratio(left,right):
        if left is None or right is None:return None
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (left,right)):
            raise ValueError('Nonpositive/nonfinite measured rate/latency/capacity')
        return right/left
    ref,cand=reference['profile'],candidate['profile'];deltas={}
    for key in ('request_rps','output_tps','osl_mean'):
        value=ratio(ref[key],cand[key]);deltas[key]={'BF16':ref[key],candidate['arm']:cand[key],
                                                  'candidate_over_BF16':value,'change_percent':100*(value-1)}
    for key in ('ttft_p50_ms','ttft_p95_ms','itl_p50_ms','itl_p95_ms','e2e_p50_ms','e2e_p95_ms'):
        value=ratio(ref[key],cand[key]);deltas[key]={'BF16':ref[key],candidate['arm']:cand[key],
            'candidate_over_BF16':value,'latency_reduction_percent':None if value is None else 100*(1-value)}
    storage={}
    for key in ('Full_KV_physical_token_slots','Full_KV_resident_bytes','GDN_state_resident_bytes','total_cache_resident_bytes'):
        left,right=reference['lifecycle'][key],candidate['lifecycle'][key]
        storage[key]={'BF16':left,candidate['arm']:right,'candidate_over_BF16':ratio(left,right)}
    if reference['lifecycle']['GDN_state_resident_bytes']!=candidate['lifecycle']['GDN_state_resident_bytes']:
        raise ValueError('GDN state capacity differs; review its independent admission constraint')
    return {'schema':'nextgen-native-KV-actual-production-pair-v1','model_id':reference['request_identity']['model_id'],
            'case':reference['case'],'concurrency':reference['concurrency'],'candidate_arm':candidate['arm'],
            'reference_run_id':reference['run_id'],'candidate_run_id':candidate['run_id'],
            'gpu_uuids':reference['request_identity']['gpu_uuids'],'service':reference['request_identity']['service'],
            'input_manifest_sha256':reference['input_manifest_sha256'],'actual_rates_and_tails':deltas,
            'resident_storage_and_physical_pool_capacity':storage,
            'operators':{'BF16':reference['request_identity']['operators'],candidate['arm']:candidate['request_identity']['operators']},
            'kv_scales':{'BF16':reference.get('kv_scales'),candidate['arm']:candidate.get('kv_scales')},
            'quality_snapshot_references':{'BF16':reference['quality_snapshot_reference_sha256'],
                                           candidate['arm']:candidate['quality_snapshot_reference_sha256']},
            'actual_phase_times':{'BF16':reference['actual_AIPerf_phase_times'],
                                  candidate['arm']:candidate['actual_AIPerf_phase_times']},
            'successful_requests_per_arm':reference['successful_requests'],'one_pair_has_statistical_variance_estimate':False,
            'different_native_kernels_are_dtype_only_causal_effect':False,'quality_accepted_or_SLO_gain_inferred':False,
            'maximum_successful_admission_capacity_inferred_from_physical_slots':False,
            'GPU_memory_utilization_is_HBM_bandwidth':False,
            'limitations':['Single complete observation per arm;no confidence interval or arrival-trace claim.',
                          'Output-token throughput and natural response-length changes are separate from request goodput.',
                          'Resident pool token slots are not an observed maximum successful concurrency.']}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','candidate','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();left=args.reference.read_bytes();right=args.candidate.read_bytes()
    result=compare(json.loads(left),json.loads(right))
    result.update(reference_audit_sha256=hashlib.sha256(left).hexdigest(),candidate_audit_sha256=hashlib.sha256(right).hexdigest())
    write_new_json(args.out,result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
