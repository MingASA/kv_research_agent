"""One bounded attribution of dense-reference error versus direct FlashInfer conformance."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .durable import write_new_json
from .upstream_operator import _fixture, _metadata, _decoded, _reference, _timings, _prefill_timings


def errors(actual, expected, atol, rtol):
    import torch
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    delta = (actual-expected).abs()
    close = torch.isclose(actual, expected, atol=atol, rtol=rtol)
    return {'atol': atol, 'rtol': rtol, 'pass': bool(close.all()),
            'max_abs_error': delta.max().item(), 'mean_abs_error': delta.mean().item(),
            'mismatched_elements': int((~close).sum()), 'elements': close.numel(),
            'bitwise_equal': bool(torch.equal(actual, expected))}


def run(output: Path):
    import torch
    import flashinfer
    from xllm.python.attention.query_quantization import quantize_query_fp8
    from flashinfer.decode import xqa_batch_decode_with_kv_cache

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (9, 0):
        raise ValueError('Diagnosis needs exactly one admitted SM90 device')
    output.mkdir()
    torch.manual_seed(20261102)
    backend, owners, state, table, key, value = _fixture(('fp8_e4m3', 'fa3', 'xqa'), 2, 129)
    before_state = (state.conv.clone(), state.ssm.clone())
    query = torch.randn((2,129,32,256), dtype=torch.bfloat16, device='cuda')[:,:,:24]
    reference_owners = tuple(torch.zeros_like(t) for t in owners)
    reference_pages = tuple(t.view(torch.float8_e4m3fn) for t in reference_owners)
    layer = SimpleNamespace(layer_id=1)
    direct_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128*2**20,dtype=torch.uint8,device='cuda'), 'NHD', backend='fa3')

    def independent_write(k, v, slots):
        for owner, tensor, scale in zip(reference_owners, (k,v), (.125,.0625)):
            quantized = (tensor.float()/scale).clamp(-448,448).to(torch.float8_e4m3fn)
            owner.view(-1,4,256).index_copy_(0, slots.long(), quantized.view(torch.uint8))

    records = {}
    for mode in ('prefill','chunked_prefill'):
        md = _metadata(2,129,128,mode,table)
        backend.prepare(md)
        q = query if mode == 'prefill' else query[:,-1:]
        k = key if mode == 'prefill' else key[:,-1:]
        v = value if mode == 'prefill' else value[:,-1:]
        q, k, v = q.reshape(-1,24,256), k.reshape(-1,4,256), v.reshape(-1,4,256)
        actual = backend.execute(q,k,v,layer)
        independent_write(k,v,md.slot_mapping)
        cache_equal = all(torch.equal(a,b) for a,b in zip(owners,reference_owners))
        q_fp8 = quantize_query_fp8(q,1.)
        independent_q = q.contiguous().float().clamp(-448,448).to(torch.float8_e4m3fn)
        q_equal = torch.equal(q_fp8.view(torch.uint8),independent_q.view(torch.uint8))
        direct_wrapper.plan(md.qo_indptr,md.paged_kv_indptr,md.paged_kv_indices,md.paged_kv_last_page_len,
            24,4,256,128,causal=True,sm_scale=256**-.5,window_left=-1,
            q_data_type=torch.float8_e4m3fn,kv_data_type=torch.float8_e4m3fn,o_data_type=torch.bfloat16)
        direct = direct_wrapper.run(independent_q,reference_pages,q_scale=1.,k_scale=.125,v_scale=.0625)
        dense = _reference(q.reshape(2,-1,24,256),_decoded(backend,table,129),
                           causal=mode=='prefill',fp8_query=True)
        records[mode] = {'cache_equals_independent_Torch_quantization':cache_equal,
            'query_equals_independent_Torch_quantization':q_equal,
            'native_vs_direct_FlashInfer':errors(actual,direct.reshape(-1,6144),.001,.001),
            'native_vs_dense_original_tolerance':errors(actual,dense,.02,.02),
            'direct_FlashInfer_vs_dense_original_tolerance':errors(direct.reshape(-1,6144),dense,.02,.02)}
        write_new_json(output/(mode+'.json'),records[mode])
        if not cache_equal or not q_equal or not records[mode]['native_vs_direct_FlashInfer']['pass']:
            raise ValueError('FP8 integration differs from independent quantization or direct FlashInfer')

    md = _metadata(2,129,128,'decode',table)
    backend.prepare(md,graph_mode=True)
    q = query[:,-1:].contiguous().reshape(2,24,256)
    k = key[:,-1:].contiguous().reshape(2,4,256)
    v = value[:,-1:].contiguous().reshape(2,4,256)
    reference_table = torch.zeros((2,2048),dtype=torch.int32,device='cuda')
    reference_table[:,:2].copy_(table)
    lengths = torch.full((2,),129,dtype=torch.uint32,device='cuda')
    workspace = torch.empty(128*2**20,dtype=torch.uint8,device='cuda')
    def direct_decode():
        return xqa_batch_decode_with_kv_cache(q,reference_pages,workspace,reference_table,lengths,262144,
            bmm1_scale=256**-.5*.125,bmm2_scale=.0625,kv_layout='NHD',enable_pdl=True)
    backend.execute(q,k,v,layer);direct_decode();torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = backend.execute(q,k,v,layer)
    graph.replay();initial=captured.clone()
    q.add_(.125);k.mul_(1.1);v.add_(.25);graph.replay()
    independent_write(k,v,md.slot_mapping)
    direct = direct_decode()
    dense = _reference(q.reshape(2,1,24,256),_decoded(backend,table,129),causal=False,fp8_query=False)
    records['changed_input_XQA_graph'] = {'graph_observed_changed_inputs':not bool(torch.equal(initial,captured)),
        'cache_equals_independent_Torch_quantization':all(torch.equal(a,b) for a,b in zip(owners,reference_owners)),
        'native_vs_direct_FlashInfer':errors(captured,direct.reshape(-1,6144),.001,.001),
        'native_vs_dense_original_tolerance':errors(captured,dense,.02,.02),
        'direct_FlashInfer_vs_dense_original_tolerance':errors(direct.reshape(-1,6144),dense,.02,.02)}
    write_new_json(output/'changed_input_XQA_graph.json',records['changed_input_XQA_graph'])
    if (not records['changed_input_XQA_graph']['graph_observed_changed_inputs']
            or not records['changed_input_XQA_graph']['cache_equals_independent_Torch_quantization']
            or not records['changed_input_XQA_graph']['native_vs_direct_FlashInfer']['pass']):
        raise ValueError('XQA integration/Graph differs from actual direct FlashInfer')
    if not torch.equal(state.conv,before_state[0]) or not torch.equal(state.ssm,before_state[1]):
        raise ValueError('FP8 operations changed GDN fixtures')
    timings = _timings(('fp8_e4m3','fa3','xqa'))
    prefill = _prefill_timings(('fp8_e4m3','fa3','xqa'))
    package = Path(flashinfer.__file__).parent
    names = ('data/include/flashinfer/attention/hopper/variants.cuh',
             'data/include/flashinfer/attention/hopper/quantization/mainloop_mma.cuh','xqa.py','decode.py')
    result = {'schema':'FP8-direct-library-attribution-v1','records':records,
        'direct_library_conformance_pass':True,'original_dense_tolerance_preserved':True,
        'all_original_dense_checks_pass':all(r['native_vs_dense_original_tolerance']['pass'] for r in records.values()),
        'timings':timings,'prefill_timings':prefill,'GDN_unchanged':True,
        'source_sha256':{n:hashlib.sha256((package/n).read_bytes()).hexdigest() for n in names if (package/n).is_file()},
        'FlashInfer_git_commit':getattr(flashinfer,'__git_commit__',getattr(flashinfer,'__git_version__',None)),
        'model_or_service_quality_claim':False,'formal_performance_claim':False}
    write_new_json(output/'diagnosis.json',result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
