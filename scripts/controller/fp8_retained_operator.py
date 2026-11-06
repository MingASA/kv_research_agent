"""Qualify retained FP8 changes after rejecting the mixed-split direction."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .durable import digest,write_new_json
from .upstream_operator import _fixture,_metadata,_writer


def reuse_query_component(request: dict,output: Path) -> None:
    from .next_stage_deployment import _completed_reference
    prior,old=_completed_reference(request,request['query_component_reference'])
    name='xllm/python/attention/query_quantization.py'
    if (old['workload']['kind']!='upstream_FP8_optimization_operator'
            or old['backend_source_sha256'][name]!=request['backend_source_sha256'][name]
            or any(old[k]!=request[k] for k in ('image_id','model_id','model_config_sha256'))):
        raise ValueError('Retained Query implementation differs from its successful component')
    root=prior/'output/fp8_e4m3';raw=(root/'operators/query.json').read_bytes()
    measurement=json.loads((root/'measurement.json').read_bytes())
    result=json.loads((prior/'result.json').read_bytes())
    acceptance=json.loads((root/'operators/acceptance.json').read_bytes())
    rows=json.loads(raw)
    if (digest(measurement)!=result['arm_results'][0]['result_sha256']
            or measurement['evidence_sha256']['operators/query.json']!=hashlib.sha256(raw).hexdigest()
            or acceptance.get('query_gate') is not True
            or [(r['tokens'],r['strided']) for r in rows]!=[(t,s) for t in (1,256,8192) for s in (False,True)]
            or any(r['bitwise_equal'] is not True or not 0<r['ratio']<=1.05 for r in rows)):
        raise ValueError('Original Query component is not the bound passed gate')
    write_new_json(output,{'query_gate':True,'exact_source_SHA_unchanged':True,
        'source_reference':request['query_component_reference'],'query_json_sha256':hashlib.sha256(raw).hexdigest(),
        'rows':rows,'new_query_microbenchmark_executed':False,'service_performance_claim':False})


def _mixed_metadata(table: object) -> object:
    import torch
    md=_metadata(3,129,128,'prefill',table)
    slots=md.slot_mapping.reshape(3,129)
    md.slot_mapping=torch.cat((slots[0,-1:],slots[1,-33:],slots[2,-1:]))
    md.qo_indptr=torch.tensor([0,1,34,35],device='cuda',dtype=torch.int32)
    md.q_cu_seq_lens=md.qo_indptr
    md.q_cu_seq_lens_host_values=[0,1,34,35]
    md.is_prefill,md.is_chunked_prefill,md.is_mixed=False,True,True
    return md


def _check_stored_bytes(backend: object,slots: object,expected_bits: list,oracle_pages: tuple) -> None:
    import torch
    for page,expected,oracle in zip(backend._pages[1],expected_bits,oracle_pages):
        observed=page.view(torch.uint8).reshape(-1,4,256).index_select(0,slots.long())
        if not torch.equal(observed,expected):
            raise ValueError('execute changed stored bytes away from independent Torch encoding')
        if not torch.equal(page.view(torch.uint8),oracle.view(torch.uint8)):
            raise ValueError('Same-value KV rewrite changed another cache slot')


def run(output: Path) -> dict:
    import torch
    import flashinfer
    from flashinfer.decode import xqa_batch_decode_with_kv_cache

    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability(0)!=(9,0):
        raise ValueError('Retained FP8 qualification requires one admitted SM90')
    reuse=json.loads((output.parent/'query_component_reuse.json').read_bytes())
    if reuse.get('query_gate') is not True or reuse.get('exact_source_SHA_unchanged') is not True:
        raise ValueError('Missing actual successful unchanged Query component')
    output.mkdir();torch.manual_seed(20261105);rows=[]
    for value_scale in (.0625,.5):
        backend,owners,state,table,key,value=_fixture(('fp8_e4m3','fa3','xqa'),3,129,value_scale)
        value[0,-1,0,0]=149.
        full=_metadata(3,129,128,'prefill',table)
        _writer(backend,key.flatten(0,1),value.flatten(0,1),full.slot_mapping)
        expected_bits=[]
        for source,page,scale in zip((key,value),backend._pages[1],(.125,value_scale)):
            expected=(source.flatten(0,1).float()/scale).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8)
            actual=page.view(torch.uint8).reshape(-1,4,256).index_select(0,full.slot_mapping.long())
            if not torch.equal(actual,expected):raise ValueError('FP8 writer scale bytes differ from independent Torch')
            expected_bits.append(expected)
        # This compressed unit-test snapshot has independently verified bytes;
        # direct operators never read pages rewritten by backend.execute().
        oracle_pages=tuple(page.clone() for page in backend._pages[1])
        if torch.count_nonzero(backend._decode_workspace[:8*1024*1024]).item():
            raise ValueError('Backend XQA first-use counters were not zero')
        before=state.conv.clone(),state.ssm.clone();layer=SimpleNamespace(layer_id=1)
        q=torch.randn((35,32,256),device='cuda',dtype=torch.bfloat16)[:,:24]
        k=torch.cat((key[0,-1:],key[1,-33:],key[2,-1:]))
        v=torch.cat((value[0,-1:],value[1,-33:],value[2,-1:]))
        md=_mixed_metadata(table);backend.prepare(md)
        if backend.native_kv_runtime_info().get('mixed_FP8_decode') is not None:
            raise ValueError('Rejected mixed split is still enabled')
        actual=backend.execute(q,k,v,layer)
        _check_stored_bytes(backend,full.slot_mapping,expected_bits,oracle_pages)
        workspace=torch.empty(128*2**20,device='cuda',dtype=torch.uint8)
        wrapper=flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace,'NHD',backend='fa3')
        wrapper.plan(torch.tensor([0,1,34,35],dtype=torch.int32),full.paged_kv_indptr_host,
            table.flatten(),full.paged_kv_last_page_len_host,24,4,256,128,causal=True,sm_scale=256**-.5,
            q_data_type=torch.float8_e4m3fn,kv_data_type=torch.float8_e4m3fn,o_data_type=torch.bfloat16)
        independent_q=q.contiguous().float().clamp(-448,448).to(torch.float8_e4m3fn)
        expected=wrapper.run(independent_q,oracle_pages,q_scale=1.,k_scale=.125,v_scale=value_scale)
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise ValueError('Retained mixed conformance requires finite outputs')
        mixed_error=(actual.reshape_as(expected).float()-expected.float()).abs().max().item()
        torch.testing.assert_close(actual.reshape_as(expected),expected,atol=.001,rtol=.001)
        md=_metadata(3,129,128,'decode',table);backend.prepare(md)
        q=torch.randn((3,32,256),device='cuda',dtype=torch.bfloat16)[:,:24]
        actual=backend.execute(q,key[:,-1],value[:,-1],layer)
        _check_stored_bytes(backend,full.slot_mapping,expected_bits,oracle_pages)
        oracle_workspace=torch.zeros_like(workspace)
        if torch.count_nonzero(oracle_workspace[:8*1024*1024]).item():raise ValueError('Oracle XQA counters not zero')
        pages=torch.zeros((3,2048),device='cuda',dtype=torch.int32);pages[:,:2].copy_(table)
        lengths=torch.full((3,),129,device='cuda',dtype=torch.uint32)
        expected=xqa_batch_decode_with_kv_cache(q.contiguous(),oracle_pages,oracle_workspace,pages,
            lengths,262144,bmm1_scale=256**-.5*.125,bmm2_scale=value_scale,kv_layout='NHD',enable_pdl=True)
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise ValueError('Retained pure-XQA conformance requires finite outputs')
        decode_error=(actual.reshape_as(expected).float()-expected.float()).abs().max().item()
        torch.testing.assert_close(actual.reshape_as(expected),expected,atol=.001,rtol=.001)
        if not torch.equal(state.conv,before[0]) or not torch.equal(state.ssm,before[1]):
            raise ValueError('Retained attention changed GDN state')
        row={'value_scale':value_scale,'writer_Torch_bitwise':True,'V_fixture_absmax':149.,
            'mixed_all_FA3_direct_pass':True,'mixed_max_abs_error':mixed_error,
            'pure_XQA_direct_pass':True,'decode_max_abs_error':decode_error,'GDN_unchanged':True,
            'direct_uses_independent_verified_snapshot':True,'finite_outputs':True,
            'atol':.001,'rtol':.001,'geometry':{'Q':24,'KV':4,'D':256,'page':128,'context':129}}
        write_new_json(output/('scale-'+str(value_scale)+'.json'),row);rows.append(row)
        del backend,owners,state,key,value,actual,expected,wrapper,workspace,oracle_workspace,oracle_pages,expected_bits
        torch.cuda.empty_cache()
    result={'schema':'FP8_retained_candidate_operator_v1','query_gate':True,'query_component_reused':True,
        'scale_rows':rows,'writer_two_scales_bitwise':True,'normal_mixed_all_FA3':True,'pure_XQA':True,
        'independent_snapshot_oracle':True,'finite_outputs':True,
        'rejected_mixed_split_present':False,'previous_mixed_performance_gate_passed':False,
        'new_operator_performance_claim':False,'model_loaded':False,'actual_model_Graph_qualification_required':True,
        'service_benchmark':False,'quality_acceptance':False}
    write_new_json(output/'acceptance.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
