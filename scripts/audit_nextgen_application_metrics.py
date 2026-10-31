"""Independently reparse sampled application HTML and bind real workload clocks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.controller.durable import write_new_json
from scripts.controller.service_metrics import parse_metric


def audit(directory: Path,performance_path: Path) -> dict:
    raw=(directory/'application_metrics.jsonl').read_bytes()
    records=[json.loads(line) for line in raw.splitlines()]
    header,rows=records[0],records[1:]
    performance=json.loads(performance_path.read_text());response=json.loads((directory/'ORIGINAL_RESPONSE.json').read_text())
    if (header.get('schema')!='nextgen-owned-long-bvar-observation-v1' or not rows
            or header['run_id']!=performance['run_id'] or response['run_id']!=performance['run_id']
            or response['stdout_sha256']!=hashlib.sha256(raw).hexdigest()
            or header['request_sha256']!=response['request_sha256']
            or response['remote_file_writes'] is not False or response['GPU_or_model_calls'] is not False
            or performance.get('actual_model_and_Graph_and_raw_service_performance_verified') is not True):
        raise ValueError('No complete audited original workload or source observation identity')
    checked=0
    for row in rows:
        for name,value in row['values'].items():
            if value!=parse_metric(name,row['raw'][name]):raise ValueError('Stored application gauge differs from original HTML')
            checked+=1
    if any(left['monotonic_ns']>=right['monotonic_ns'] for left,right in zip(rows,rows[1:])):
        raise ValueError('Observation clock is not strictly increasing')
    profile=performance['profile'];start=profile['worker_observation_started_monotonic_ns'];end=profile['worker_observation_ended_monotonic_ns']
    during=[row for row in rows if start<=row['monotonic_ns']<=end and row['values']]
    if not during:raise ValueError('No gauge observation during original measured worker interval')
    names=sorted({key for row in during for key in row['values']})
    ranges={name:{'min':min(row['values'][name] for row in during if name in row['values']),
                  'max':max(row['values'][name] for row in during if name in row['values'])} for name in names}
    successful=[row for row in rows if not row['errors']]
    counters={}
    for name in ('prefix_cache_prompt_tokens_total','prefix_cache_hit_tokens_total'):
        values=[row['values'][name] for row in successful]
        if any(a>b for a,b in zip(values,values[1:])):raise ValueError('Application counter reset or dropped')
        counters[name]=values[-1]-values[0]
    prompts=int(round(profile['isl_mean']*performance['successful_requests']))
    offered=counters['prefix_cache_prompt_tokens_total'];hits=counters['prefix_cache_hit_tokens_total']
    return {'schema':'nextgen-independent-application-gauges-audit-v1','run_id':header['run_id'],
        'request_sha256':header['request_sha256'],'source_local_jsonl_sha256':hashlib.sha256(raw).hexdigest(),
        'source_bytes':len(raw),'total_samples':len(rows),'measured_worker_interval_samples':len(during),
        'raw_HTML_metric_values_independently_reparsed':checked,'error_samples':sum(bool(row['errors']) for row in rows),
        'utc_span':[rows[0]['utc'],rows[-1]['utc']],'ranges_during_measured_worker_interval':ranges,
        'counter_deltas_across_successful_observation_span':counters,
        'actual_offered_prompt_tokens_from_audited_API':prompts,
        'counter_span_covers_exact_offered_prompt_total':offered==prompts,
        'prefix_hit_fraction_for_observed_counter_span':hits/offered if offered else None,
        'performance_audit_path':str(performance_path),
        'performance_audit_sha256':hashlib.sha256(performance_path.read_bytes()).hexdigest(),
        'GPU_or_model_calls':False,'remote_file_writes':False,'max_capacity_or_HBM_traffic_claim':False,
        'observed_preemption_gauge_is_not_cumulative':True,'prefix_gauge_counts_include_composite_GDN_leaf':True,
        'limitations':['2-second sequential application sampling can miss peaks; errors remain recorded.',
            'Used Full-KV blocks and cached prefixes are different, overlapping allocator views.',
            'Composite prefix block counts sum Full-KV and GDN leaves; they are not Full-KV pages.',
            'Only an exactly matched counter span can describe full-workload prefix hit fraction.',
            'No maximum supported admission, cumulative preemption count or hardware traffic is certified.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations',type=Path,required=True)
    parser.add_argument('--performance-audit',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=audit(args.observations,args.performance_audit)
    write_new_json(args.out,result);print(json.dumps(result))
