"""CPU-only receive gaps between nonempty content packets in original raw SSE."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.audit_nextgen_validation import _mirror_paths
from scripts.audit_native_results import _percentile
from scripts.controller.durable import write_new_json


def audit(manifest_path,performance_path):
    manifest=json.loads(manifest_path.read_bytes());base,root=_mirror_paths(manifest)
    performance=json.loads(performance_path.read_bytes())
    assert performance['actual_model_and_Graph_and_raw_service_performance_verified'] is True
    assert performance['all_offered_requests_completed'] is True and performance['run_id']==manifest['run_id']
    arm=performance['arm'];path=root/'output'/arm/'measured/artifacts/profile_export_raw.jsonl'
    expected=next(row for row in manifest['files'] if (base/row['base_rel']).resolve(strict=True)==path.resolve(strict=True))
    raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    assert sha==expected['snapshot_sha256']==expected['remote_sha256'] and len(raw)==expected['size']
    rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(rows)==performance['successful_requests'];gaps=[];details=[]
    for row in rows:
        content_times=[]
        for response in row['responses']:
            timestamp=response['perf_ns'];assert type(timestamp) is int
            for packet in response['packets']:
                if packet['name']!='data' or packet['value']=='[DONE]':continue
                body=json.loads(packet['value'])
                for choice in body.get('choices',[]):
                    content=choice.get('delta',{}).get('content')
                    if isinstance(content,str) and content:content_times.append(timestamp)
        assert all(left<=right for left,right in zip(content_times,content_times[1:]))
        local=[(right-left)/1e6 for left,right in zip(content_times,content_times[1:])];gaps.extend(local)
        details.append({'x_request_id':row['metadata']['x_request_id'],
            'session':row['metadata']['conversation_id'],'turn':row['metadata']['turn_index'],
            'nonempty_content_packets':len(content_times),'observed_adjacent_gaps':len(local),
            'max_gap_ms':max(local) if local else None,'gaps_over_1second':sum(value>1000 for value in local)})
    assert gaps
    return {'schema':'original_raw_SSE_nonempty_content_packet_receive_gaps-v1','run_id':manifest['run_id'],
        'arm':arm,'case':performance['case'],'service':performance['request_identity']['service'],
        'requests':len(rows),'requests_without_two_nonempty_content_packets':sum(row['observed_adjacent_gaps']==0 for row in details),
        'all_original_requests_retained':True,'observed_adjacent_content_packet_gaps':len(gaps),
        'zero_gaps_due_to_same_receive_timestamp':sum(value==0 for value in gaps),
        'gap_ms':{'p50':_percentile(gaps,.5),'p95':_percentile(gaps,.95),'p99':_percentile(gaps,.99),'max':max(gaps)},
        'gaps_over_1second':sum(value>1000 for value in gaps),'per_request':details,
        'raw_sha256':sha,'performance_audit_sha256':hashlib.sha256(performance_path.read_bytes()).hexdigest(),
        'source_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'auditor_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'GPU_or_remote_calls':False,'individual_model_token_or_server_kernel_gap_measurement':False,
        'scope':'Adjacent nonempty content SSE packets at recorded client receive timestamps. Packets can contain different token counts or be coalesced in one read; zero gaps kept. Diagnostic, not per-token latency SLO acceptance.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['sha-manifest','performance-audit','out']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=audit(args.sha_manifest,args.performance_audit);write_new_json(args.out,result)
    print(json.dumps({k:result[k] for k in ['run_id','requests','observed_adjacent_content_packet_gaps','gap_ms','gaps_over_1second']}))
