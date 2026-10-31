#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Audit sealed FP8-KV/BF16-Q operator evidence without repeating GPU work."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.controller.durable import digest,write_new_json


def _sha(path: Path) -> str:
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):value.update(chunk)
    return value.hexdigest()


def _require(value: bool,message: str) -> None:
    if not value:raise ValueError(message)


def audit(mirror: Path,run: str,source: Path) -> dict:
    snapshot=json.loads((mirror/'SNAPSHOT.json').read_text())
    for row in snapshot['files']:
        path=mirror/row['relative_path']
        _require(not path.is_symlink() and path.resolve().is_relative_to(mirror.resolve())
                 and path.stat().st_size==row['size'] and _sha(path)==row['sha256'],'Mirrored bytes/SHA differ')
    root=mirror/'support/agent_work/runs'/run
    ticket=json.loads((root/'ticket.json').read_text());r=ticket['request'];terminal=json.loads((root/'result.json').read_text())
    _require(r['run_id']==run and terminal['run_id']==run and digest(r)==ticket['request_sha256']
        and terminal['request_sha256']==ticket['request_sha256'] and terminal['status']=='COMPLETE'
        and terminal['gpu_release_verified'] is True and terminal['live_owned_containers']==[]
        and json.loads((root/'COMPLETE').read_text())['result_sha256']==_sha(root/'result.json'),
        'No exact COMPLETE seal and verified driver release')
    _require(r['workload']['kind']=='native_bf16q_decode_operator' and r['arms']==['fp8_e4m3'],
        'Wrong operator scope')
    for name,expected in ticket['assets_sha256'].items():
        _require(_sha(root/name)==expected,'Frozen asset differs: '+name)
    for name,expected in r['backend_source_sha256'].items():
        _require(_sha(source/name)==expected,'Local independent source differs: '+name)
    input_path=root/'input/operator_manifest.json';manifest=json.loads(input_path.read_text())
    _require(_sha(input_path)==r['dataset_sha256'] and manifest['synthetic_tensors_only'] is True
        and manifest['geometry']=={'batch':128,'heads':40,'kv_heads':8,'head_dim':128,'page_size':128},
        'Input identity or actual-production envelope differs')
    out=root/'output/fp8_e4m3';m=json.loads((out/'measurement.json').read_text())
    _require(m['request_sha256']==ticket['request_sha256'] and m['measurement_success'] is True
        and m['operator_only'] is True and m['model_loaded'] is False and m['official_aiperf_complete'] is False
        and m['quality_or_service_result'] is False and terminal['arm_results']==[
            {'arm':'fp8_e4m3','result_sha256':_sha(out/'measurement.json')}],
        'Wrong measurement/terminal binding or model/service certification')
    for name,expected in m['evidence_sha256'].items():
        p=out/name;_require(p.resolve().is_relative_to(out.resolve()) and _sha(p)==expected,'Sealed evidence differs')
    _require(not (out/'server.log').exists() and not (out/'artifacts').exists(),'Unexpected model/AIPerf output')
    a=json.loads((out/'operator_acceptance.json').read_text());g=a['graph128'];n=a['native_operator'];b=g['backend']
    _require(a['classification']=='confirmed_BF16Q_FP8KV_operator_Graph128_only'
        and all(a[k] is False for k in ('model_loaded','AIPerf_ran','model_quality_validated','service_performance_measured'))
        and a['installed_sources_sha256']==manifest['primary_sources_sha256']==r['workload']['primary_sources_sha256'],
        'Actual installed primary source or output scope differs')
    required={'profile_scope':'production_bf16q_decode','planner_backend':'fa2','decode_impl':'flashinfer',
        'q_data_type':'torch.bfloat16','decode_internal_q_data_type':'torch.bfloat16','kv_data_type':'torch.float8_e4m3fn',
        'decode_tensor_cores':True,'q_scale':None,'fp8_views_alias_resident_pages':True,
        'resident_initialization_policy':'finite_full_initialization','full_BF16_KV_shadow':False,
        'k_scale':.125,'v_scale':.0625}
    _require(all(b.get(k)==v and n['backend'].get(k)==v for k,v in required.items()),'Actual Q/KV/scale/residency differs')
    _require('H800' in n['device'] and g==m['graph128'] and b==m['backend']
        and g['production_max_batch']==128 and g['reference_Q']=='BF16_without_FP8_QDQ'
        and g['relative_l2_limit']==.01 and all(g[k] is True for k in ('real_capture_and_replay',
            'writer_in_same_graph','noncontiguous_QKV','reserved_page0_unchanged','resident_storage_stable','rebind_preserves_live_cache')),
        'Actual H800 Graph128/writer/storage scope is incomplete')
    _require(all(n[k] is True for k in ('real_capture_and_replay','partial_page_and_replan',
        'noncontiguous_q_checked','padded_graph_bucket_checked','fresh_dirty_cache_initialized_to_zero')),
        'Missing original numerical/Graph/initialization checks')
    numerical={};large=[]
    for line in (out/'operator_acceptance.log').read_text().splitlines():
        if 'OPERATOR_REFERENCE ' in line:
            name,data=line.split('OPERATOR_REFERENCE ',1)[1].split(' ',1)
            _require(name not in numerical,'Duplicated numerical evidence');numerical[name]=ast.literal_eval(data)
        if 'BF16Q_GRAPH128 ' in line:large.append(ast.literal_eval(line.split('BF16Q_GRAPH128 ',1)[1]))
    _require(numerical==n['reference_comparisons'] and set(numerical)=={
        'prefill','capture_replay','changed_context','padded_graph'} and large==g['dynamic_live_rows_and_context']
        and [(x['bucket'],x['live_rows'],x['context']) for x in large]==[
            (128,128,257),(128,126,130),(128,1,130),(128,127,257)],'Actual logs and structured numerical rows differ')
    errors=list(numerical.values())+large
    _require(all(math.isfinite(x['relative_l2']) and 0<=x['relative_l2']<=.01
        and math.isfinite(x['max_abs']) and x['max_abs']>=0 for x in errors),'Actual error exceeds the fixed BF16-Q gate')
    return {'classification':'confirmed_operator_evidence_integrity_and_BF16Q_Graph128_contract_only',
        'run_id':run,'source_commit':r['python_source_commit'],'request_sha256':ticket['request_sha256'],
        'terminal_result_sha256':_sha(root/'result.json'),'files_SHA_verified':len(snapshot['files']),
        'backend_source_sha256':r['backend_source_sha256'],'binary_sha256':r['binary_sha256'],
        'image_id':r['image_id'],'operators':r['operators'],
        'bytes_SHA_verified':sum(x['size'] for x in snapshot['files']),
        'actual_numerical_Graph_reported_and_sealed':True,'numerical_execution_independently_repeated':False,
        'installed_sources_sha256':a['installed_sources_sha256'],'native_comparisons':numerical,'graph128':g,
        'model_loaded':False,'model_quality_certified':False,'new_service_benefit_certified':False,
        'next':'Explicit Graph128 quality contract and necessary full1719 candidate evaluation; retain modern BF16 reference'}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('mirror','source','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--run-id',required=True);a=parser.parse_args()
    result=audit(a.mirror,a.run_id,a.source);write_new_json(a.out,result)
    print(json.dumps({k:result[k] for k in ('classification','files_SHA_verified','source_commit')}))


if __name__=='__main__':main()
