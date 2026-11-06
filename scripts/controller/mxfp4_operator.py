"""Bounded SM90 MXFP4 writer, paged attention and changed-input Graph gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .durable import write_new_json


def _writer_edges(output: Path) -> None:
    """Exercise scale underflow, signed zero, rounding ties and nonfinite blocks."""
    import numpy as np
    import torch
    from xllm.python.attention.mxfp4_reference import encode_mxfp4
    from xllm.python.attention.mxfp4_writer import mxfp4_page_views, write_mxfp4_kv

    source=torch.zeros((8,4,256),device='cuda',dtype=torch.bfloat16)
    source[0,0,::2]=-0.
    source[1].fill_(2.**-133)
    source[2].fill_(2.**-127)
    source[3].fill_(2.**-126)
    source[4].fill_(torch.finfo(torch.bfloat16).max)
    source[5,0,0]=float('nan')
    source[6,1,31]=float('inf')
    ties=torch.tensor([0.,-0.,.25,-.25,.75,-.75,1.25,-1.25,1.75,-1.75,
                       2.5,-2.5,3.5,-3.5,5.,-5.,6.,-6.],device='cuda',dtype=torch.bfloat16)
    source[7,:, :18]=ties
    owners=tuple(torch.zeros((1,128,4,136),device='cuda',dtype=torch.uint8) for _ in range(2))
    views=tuple(mxfp4_page_views(t,256) for t in owners)
    slots=torch.arange(8,device='cuda',dtype=torch.int64)
    write_mxfp4_kv(source,source,slots,tuple(v[0] for v in views),tuple(v[1] for v in views))
    packed,sf=encode_mxfp4(source.float().cpu().numpy())
    for data,scales in views:
        np.testing.assert_array_equal(data[0,:8].cpu().numpy(),packed)
        np.testing.assert_array_equal(scales[0,:8].cpu().numpy(),sf)
    before=tuple(t.clone() for t in owners)
    invalid_slots=torch.tensor([-1,128]*4,device='cuda',dtype=torch.int64)
    write_mxfp4_kv(source,source,invalid_slots,tuple(v[0] for v in views),tuple(v[1] for v in views))
    if not all(torch.equal(a,b) for a,b in zip(before,owners)):
        raise ValueError('Invalid MXFP4 slots changed a physical page')
    write_new_json(output/'writer_edges.json',{'CPU_bitwise_equal':True,'tokens':8,
        'BF16_subnormals':True,'signed_zero':True,'rounding_ties':True,'nonfinite_block_NaN_scale':True,
        'negative_and_out_of_bounds_slots_unchanged':True})


def run(output: Path) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from xllm.python.attention.mxfp4_reference import encode_mxfp4, decode_mxfp4
    from xllm.python.attention.mxfp4_writer import mxfp4_page_views, write_mxfp4_kv
    from xllm.python.attention.mxfp4_flashinfer import make_mxfp4_wrapper

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (9, 0):
        raise ValueError('Operator qualification requires one admitted SM90')
    output.mkdir()
    _writer_edges(output)
    torch.manual_seed(20261105)
    batch, context, page, heads, kv_heads, dimension = 2, 129, 128, 24, 4, 256
    owners = tuple(torch.zeros((5, page, kv_heads, 136), dtype=torch.uint8, device='cuda') for _ in range(2))
    views = tuple(mxfp4_page_views(t, dimension) for t in owners)
    pages = tuple(v[0] for v in views)
    scales = tuple(v[1] for v in views)
    table = torch.tensor([[1, 2], [3, 4]], device='cuda', dtype=torch.int32)
    slots = (table[:, :, None] * page + torch.arange(page, device='cuda')).flatten(1)[:, :context].flatten().contiguous()
    values = tuple(torch.randn((batch * context, kv_heads, dimension), device='cuda', dtype=torch.bfloat16) for _ in range(2))
    write_mxfp4_kv(*values, slots, pages, scales)
    references = []
    for source, packed, sf in zip(values, pages, scales):
        cpu = source.float().cpu().numpy()
        ref_data, ref_sf = encode_mxfp4(cpu)
        flat = slots.cpu().numpy()
        actual_data = packed.cpu().numpy().reshape(-1, kv_heads, dimension // 2)[flat]
        actual_sf = sf.cpu().numpy().reshape(-1, kv_heads, dimension // 32)[flat]
        np.testing.assert_array_equal(actual_data, ref_data)
        np.testing.assert_array_equal(actual_sf, ref_sf)
        references.append(torch.from_numpy(decode_mxfp4(ref_data, ref_sf)).reshape(batch, context, kv_heads, dimension))
    write_new_json(output/'writer.json', {'CPU_bitwise_equal': True, 'storage_bytes': sum(t.numel() for t in owners),
        'physical_bytes_per_element': .53125, 'full_BF16_shadow': False, 'block_size': 32, 'scale': 'E8M0'})

    workspace = torch.empty(128 * 2**20, dtype=torch.uint8, device='cuda')
    wrapper = make_mxfp4_wrapper(workspace)
    host_indptr = torch.tensor([0, 2, 4], dtype=torch.int32)
    indptr = host_indptr.cuda()
    tails = torch.tensor([1, 1], dtype=torch.int32, device='cuda')
    results = {}
    for mode in ('prefill', 'decode'):
        query_len = context if mode == 'prefill' else 1
        query = torch.randn((batch * query_len, heads, dimension), dtype=torch.bfloat16, device='cuda')
        qo = torch.arange(batch + 1, dtype=torch.int32) * query_len
        wrapper.plan(qo, host_indptr, table.flatten(), tails.cpu(), heads, kv_heads, dimension, page,
                     causal=True, sm_scale=dimension ** -.5, q_data_type=torch.bfloat16,
                     kv_data_type=torch.uint8, o_data_type=torch.bfloat16)
        actual = wrapper.run(query, pages, kv_cache_sf=scales)
        q = query.float().cpu().reshape(batch, query_len, heads, dimension).transpose(1, 2)
        key, value = (r.repeat_interleave(6, dim=2).transpose(1, 2) for r in references)
        expected = functional.scaled_dot_product_attention(q, key, value, is_causal=mode == 'prefill').transpose(1, 2).reshape_as(actual)
        delta = (actual.float().cpu() - expected).abs()
        passed = torch.allclose(actual.float().cpu(), expected, atol=.02, rtol=.02)
        results[mode] = {'pass': passed, 'max_abs_error': delta.max().item(), 'mean_abs_error': delta.mean().item()}
        write_new_json(output/(mode+'.json'), results[mode])
        if not passed:
            raise ValueError('MXFP4 paged attention differs from independent decoded CPU reference')
        if mode == 'decode':
            # Capture actual writer + attention, then change inputs in place.
            k, v = (t.reshape(batch, context, kv_heads, dimension)[:, -1].contiguous() for t in values)
            last_slots = slots.reshape(batch, context)[:, -1].contiguous()
            graph = torch.cuda.CUDAGraph()
            for _ in range(3):
                write_mxfp4_kv(k, v, last_slots, pages, scales)
                wrapper.run(query, pages, kv_cache_sf=scales)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                write_mxfp4_kv(k, v, last_slots, pages, scales)
                captured = wrapper.run(query, pages, kv_cache_sf=scales)
            graph.replay()
            initial = captured.clone()
            query.add_(.125); k.mul_(1.25); v.add_(.25)
            graph.replay()
            flat=last_slots.cpu().numpy()
            for source,data,sf in zip((k,v),pages,scales):
                expected_data,expected_sf=encode_mxfp4(source.float().cpu().numpy())
                np.testing.assert_array_equal(data.cpu().numpy().reshape(-1,kv_heads,dimension//2)[flat],expected_data)
                np.testing.assert_array_equal(sf.cpu().numpy().reshape(-1,kv_heads,dimension//32)[flat],expected_sf)
            changed = not torch.equal(initial, captured)
            direct = wrapper.run(query, pages, kv_cache_sf=scales)
            if not changed or not torch.equal(captured, direct):
                raise ValueError('MXFP4 changed-input Graph differs from direct attention')
            results['Graph'] = {'changed_inputs': changed, 'bitwise_equal_direct': True}
    # Whole-page copy includes both E2M1 payload and E8M0 scales.
    copies = tuple(t.clone() for t in owners)
    for t in copies:
        t[0].copy_(t[1])
    if not all(torch.equal(t[0], original[1]) for t, original in zip(copies, owners)):
        raise ValueError('MXFP4 page copy lost scale or data bytes')
    result = {'schema':'MXFP4_SM90_operator_gate_v1','format':'OCP_MXFP4_E2M1_block32_E8M0',
              'writer_CPU_bitwise':True,'results':results,'whole_page_copy':True,'model_loaded':False,
              'service_benchmark':False,'geometry':{'Q':24,'KV':4,'D':256,'page':128},
              'BF16_history_workspace_bytes':0,'resident_full_KV_bytes':sum(t.numel() for t in owners)}
    write_new_json(output/'acceptance.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
