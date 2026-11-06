"""Finite FP8 query fusion/mixed-phase attribution before serving experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from .durable import write_new_json
from .upstream_operator import _fixture, _metadata, _writer, _graph_us


def _mixed_metadata(table, context, lengths):
    import torch
    md = _metadata(len(lengths), context, 128, 'prefill', table)
    full_slots = md.slot_mapping.reshape(len(lengths), context)
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1]+length)
    md.slot_mapping = torch.cat([full_slots[i,-length:] for i,length in enumerate(lengths)])
    md.qo_indptr = torch.tensor(cumulative, dtype=torch.int32, device='cuda')
    md.q_cu_seq_lens = md.qo_indptr
    md.q_cu_seq_lens_host_values = cumulative
    md.is_prefill, md.is_chunked_prefill, md.is_mixed = False, True, True
    return md


def _control(backend):
    import torch
    from xllm.python.attention.flashinfer_native_kv import FlashInferNativeKVBackend
    result = FlashInferNativeKVBackend(24,4,256,256**-.5,0,torch.device('cuda',0),
        torch.bfloat16,'fp8_e4m3',.125,.0625,prefill_backend='fa3',decode_backend='flashinfer',
        max_context_tokens=262144)
    result.bind_kv_caches(backend._kv_caches)
    return result


def run(output: Path, baseline_query: Path):
    import torch
    import flashinfer
    from flashinfer.decode import xqa_batch_decode_with_kv_cache
    from xllm.python.attention.query_quantization import quantize_query_fp8

    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability(0)!=(9,0):
        raise ValueError('FP8 operator gate requires exactly one admitted SM90')
    output.mkdir()
    spec=importlib.util.spec_from_file_location('frozen_query_before_20261102',baseline_query)
    old=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=old
    spec.loader.exec_module(old)
    torch.manual_seed(20261105)
    query_rows=[]
    for tokens in (1,256,8192):
        for strided in (False,True):
            q=torch.randn((tokens,32 if strided else 24,256),device='cuda',dtype=torch.bfloat16)[:,:24]
            actual=quantize_query_fp8(q,1.)
            trusted=old.quantize_query_fp8(q,1.)
            independent=q.contiguous().float().clamp(-448,448).to(torch.float8_e4m3fn)
            if not torch.equal(actual.view(torch.uint8),trusted.view(torch.uint8)) or not torch.equal(actual.view(torch.uint8),independent.view(torch.uint8)):
                raise ValueError('Strided query fusion changed FP8 bytes')
            before=_graph_us(lambda:old.quantize_query_fp8(q,1.))
            after=_graph_us(lambda:quantize_query_fp8(q,1.))
            query_rows.append({'tokens':tokens,'strided':strided,'before_us':before,'after_us':after,'ratio':after/before,'bitwise_equal':True})
    write_new_json(output/'query.json',query_rows)

    backend,owners,state,table,key,value=_fixture(('fp8_e4m3','fa3','xqa'),3,129)
    full=_metadata(3,129,128,'prefill',table)
    _writer(backend,key.flatten(0,1),value.flatten(0,1),full.slot_mapping)
    md=_mixed_metadata(table,129,[1,33,1]);backend.prepare(md)
    q=torch.randn((35,32,256),device='cuda',dtype=torch.bfloat16)[:,:24]
    k=torch.cat([key[0,-1:],key[1,-33:],key[2,-1:]])
    v=torch.cat([value[0,-1:],value[1,-33:],value[2,-1:]])
    layer=SimpleNamespace(layer_id=1)
    state_before=state.conv.clone(),state.ssm.clone()
    actual=backend.execute(q,k,v,layer)
    pages=backend._pages[1]
    direct_table=torch.zeros((2,2048),dtype=torch.int32,device='cuda')
    direct_table[:,:2].copy_(table[[0,2]])
    direct_lengths=torch.full((2,),129,dtype=torch.uint32,device='cuda')
    workspace=torch.empty(128*2**20,dtype=torch.uint8,device='cuda')
    direct_decode=xqa_batch_decode_with_kv_cache(q[[0,34]].contiguous(),pages,workspace,
        direct_table,direct_lengths,262144,bmm1_scale=256**-.5*.125,bmm2_scale=.0625,
        kv_layout='NHD',enable_pdl=True)
    direct_prefill=flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty_like(workspace),'NHD',backend='fa3')
    direct_prefill.plan(torch.tensor([0,33],dtype=torch.int32),torch.tensor([0,2],dtype=torch.int32),
        table[1],torch.tensor([1],dtype=torch.int32),24,4,256,128,causal=True,sm_scale=256**-.5,
        q_data_type=torch.float8_e4m3fn,kv_data_type=torch.float8_e4m3fn,o_data_type=torch.bfloat16)
    independent_query=q[1:34].contiguous().float().clamp(-448,448).to(torch.float8_e4m3fn)
    expected_prefill=direct_prefill.run(independent_query,pages,q_scale=1.,k_scale=.125,v_scale=.0625)
    expected=torch.cat((direct_decode[:1],expected_prefill,direct_decode[1:])).reshape_as(actual)
    torch.testing.assert_close(actual,expected,atol=.001,rtol=.001)
    if not torch.equal(state.conv,state_before[0]) or not torch.equal(state.ssm,state_before[1]):
        raise ValueError('Quantized attention modified GDN state')
    conformance={'independent_direct_FI_conformance':True,'max_abs_error':(actual-expected).abs().max().item(),
                 'nonleading_decode_rows':[0,34],'GDN_unchanged':True,'mixed_service_CUDA_Graph_claim':False}
    write_new_json(output/'mixed_conformance.json',conformance)
    del backend,owners,state,table,key,value,actual,expected,direct_prefill,workspace
    torch.cuda.empty_cache()

    rows=[]
    for context,lengths in ((1024,[1]*8+[512]*4),(8192,[1]*31+[1024]),(32768,[1]*16+[4096]*2)):
        backend,_,_,table,key,value=_fixture(('fp8_e4m3','fa3','xqa'),len(lengths),context)
        control=_control(backend)
        full=_metadata(len(lengths),context,128,'prefill',table)
        _writer(backend,key.flatten(0,1),value.flatten(0,1),full.slot_mapping)
        md=_mixed_metadata(table,context,lengths)
        q=torch.randn((sum(lengths),32,256),device='cuda',dtype=torch.bfloat16)[:,:24]
        k=torch.cat([key[i,-length:] for i,length in enumerate(lengths)])
        v=torch.cat([value[i,-length:] for i,length in enumerate(lengths)])
        # Historical BF16 inputs are only fixture construction sources, not a
        # serving cache. Release them before timing/memory observations.
        del key,value,full
        measured={}
        for label,candidate in [('all_FA3',control),('split_XQA_FA3',backend)]:
            torch.cuda.synchronize()
            allocated_before=torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            candidate.prepare(md)
            device_us=_graph_us(lambda:candidate.execute(q,k,v,layer))
            peak_bytes=torch.cuda.max_memory_allocated()
            torch.cuda.synchronize();start=time.perf_counter_ns()
            for _ in range(10):candidate.prepare(md)
            torch.cuda.synchronize();prepare_us=(time.perf_counter_ns()-start)/10000
            measured[label]={'writer_attention_device_graph_us':device_us,'prepare_wall_us':prepare_us,
                             'per_layer_plus_prepare_amortized_16_us':device_us+prepare_us/16,
                             'allocator_bytes_before':allocated_before,'allocator_peak_bytes':peak_bytes,
                             'resident_KV_bytes':candidate.native_kv_runtime_info()['resident_KV_bytes']}
        ratio=measured['split_XQA_FA3']['per_layer_plus_prepare_amortized_16_us']/measured['all_FA3']['per_layer_plus_prepare_amortized_16_us']
        rows.append({'context':context,'query_lengths':lengths,'measurements':measured,'ratio':ratio,
                     'operator_capture_only_not_actual_mixed_service_graph':True})
        del backend,control,table,q,k,v,md
        torch.cuda.empty_cache()
    query_gate=all(r['ratio']<=1.05 for r in query_rows)
    mixed_geomean=math.exp(sum(math.log(r['ratio']) for r in rows)/len(rows))
    mixed_gate=mixed_geomean<=.97 and max(r['ratio'] for r in rows)<=1.15
    result={'schema':'FP8_finite_optimization_operator_v1','query':query_rows,'mixed':rows,
            'query_gate':query_gate,'mixed_gate':mixed_gate,'mixed_geomean':mixed_geomean,
            'mixed_conformance':conformance,'baseline_query_source_sha256':hashlib.sha256(baseline_query.read_bytes()).hexdigest(),
            'model_loaded':False,'service_performance_claim':False,'quality_acceptance_claim':False}
    write_new_json(output/'acceptance.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-query',type=Path,required=True)
    args=parser.parse_args();run(args.output,args.baseline_query)
