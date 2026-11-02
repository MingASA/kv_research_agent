"""Real SM90 native-cache qualification and bounded direct FlashInfer timings.

Run only inside the admitted single-GPU worker. The dense CPU reference is a
small correctness fixture; it is never a serving KV shadow or service benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

from .durable import write_new_json

CANDIDATES = (('fp8_e4m3', 'fa3', 'xqa'), ('fp8_e4m3', 'fa2', 'flashinfer'),
              ('nvfp4', 'fa2', 'flashinfer'))


def _metadata(batch: int, context: int, page: int, mode: str, table: object) -> object:
    import torch

    pages = (context + page - 1) // page
    query_length = context if mode == 'prefill' else 1
    host_indptr = torch.arange(batch + 1, dtype=torch.int32) * pages
    host_last = torch.full((batch,), (context - 1) % page + 1, dtype=torch.int32)
    if mode == 'prefill':
        positions = torch.arange(context, device='cuda').expand(batch, -1)
        slots = (table[:, :, None] * page + torch.arange(page, device='cuda')).reshape(batch, -1)[:, :context]
    else:
        positions = torch.full((batch, 1), context - 1, device='cuda')
        slots = table[:, -1:] * page + (context - 1) % page
    del positions
    query_indptr = torch.arange(batch + 1, dtype=torch.int32, device='cuda') * query_length
    return SimpleNamespace(
        slot_mapping=slots.reshape(-1).contiguous().to(torch.int32),
        paged_kv_indptr=host_indptr.cuda(), paged_kv_indices=table.flatten(),
        paged_kv_last_page_len=host_last.cuda(), paged_kv_indptr_host=host_indptr,
        paged_kv_last_page_len_host=host_last, qo_indptr=query_indptr,
        q_cu_seq_lens=query_indptr, kv_cu_seq_lens=torch.arange(batch + 1, device='cuda', dtype=torch.int32) * context,
        kv_seq_lens_host=torch.arange(batch + 1, dtype=torch.int32) * context,
        is_prefill=mode == 'prefill', is_chunked_prefill=mode == 'chunked_prefill',
    )


def _decoded(backend: object, table: object, context: int) -> tuple[object, object]:
    import torch

    values = []
    indices = table.cpu().long()
    for index, tensor in enumerate(backend._pages[1]):
        packed = tensor.cpu()
        if backend._cache_format == 'fp8_e4m3':
            decoded = packed.float() * backend._fixed_scales[index]
        else:
            codebook = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
            nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
            scales = backend._block_scales[1][index].cpu().float().repeat_interleave(16, dim=-1)
            decoded = codebook[nibbles.long()] * scales * backend._fixed_scales[index]
        values.append(decoded[indices].flatten(1, 2)[:, :context])
    return tuple(values)


def _reference(query: object, decoded: tuple[object, object], *, causal: bool, fp8_query: bool) -> object:
    import torch
    import torch.nn.functional as functional

    q = query.detach().cpu()
    q = q.to(torch.float8_e4m3fn).float() if fp8_query else q.float()
    key, value = (t.repeat_interleave(6, dim=2).transpose(1, 2) for t in decoded)
    out = functional.scaled_dot_product_attention(q.transpose(1, 2), key, value, is_causal=causal)
    return out.transpose(1, 2).flatten(0, 1).flatten(1)


def _close(actual: object, expected: object) -> dict:
    import torch

    observed = actual.detach().float().cpu()
    error = (observed - expected).abs()
    torch.testing.assert_close(observed, expected, atol=.02, rtol=.02)
    return {'atol': .02, 'rtol': .02, 'max_abs_error': error.max().item(),
            'mean_abs_error': error.mean().item(), 'finite': bool(torch.isfinite(observed).all())}


def _writer(backend: object, key: object, value: object, slots: object) -> None:
    import flashinfer.page
    from xllm.python.attention.native_kv_writer import write_fp8_kv

    if backend._cache_format == 'fp8_e4m3':
        write_fp8_kv(key, value, slots, backend._pages[1], backend._fixed_scales)
    else:
        flashinfer.page.nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
            key, value, slots, backend._pages[1], backend._block_scales[1], *backend._global_scales, kv_layout='NHD')


def _fixture(candidate: tuple[str, str, str], batch: int, context: int) -> tuple:
    import torch
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.flashinfer_native_kv import FlashInferNativeKVBackend

    fmt, prefill, decode = candidate
    page = 128
    pages = (context + page - 1) // page
    table = torch.arange(1, 1 + batch * pages, device='cuda', dtype=torch.int32).reshape(batch, pages)
    physical = 256 if fmt == 'fp8_e4m3' else 144
    owners = tuple(torch.zeros((2 + batch * pages, page, 4, physical), dtype=torch.uint8, device='cuda') for _ in range(2))
    state = LayerCache(None, None, conv=torch.arange(12, device='cuda', dtype=torch.bfloat16),
                       ssm=torch.arange(16, device='cuda', dtype=torch.float32))
    scales = (.125, .0625) if fmt == 'fp8_e4m3' else (1., 1.)
    backend = FlashInferNativeKVBackend(24, 4, 256, 256**-.5, 0, torch.device('cuda', 0),
                                       torch.bfloat16, fmt, *scales, prefill_backend=prefill,
                                       decode_backend=decode, max_context_tokens=262144)
    backend.bind_kv_caches([state, LayerCache(*owners)])
    key = torch.randn((batch, context, 4, 256), device='cuda', dtype=torch.bfloat16) * .8
    value = torch.randn_like(key)
    return backend, owners, state, table, key, value


def _qualify(candidate: tuple[str, str, str], directory: Path) -> dict:
    import torch

    backend, owners, state, table, key, value = _fixture(candidate, 2, 129)
    state_before = (state.conv.clone(), state.ssm.clone())
    addresses = tuple(t.data_ptr() for t in owners)
    query = torch.randn((2, 129, 32, 256), device='cuda', dtype=torch.bfloat16)[:, :, :24]
    layer = SimpleNamespace(layer_id=1)
    phases = {}
    for mode in ('prefill', 'chunked_prefill', 'decode'):
        metadata = _metadata(2, 129, 128, mode, table)
        backend.prepare(metadata)
        q = query if mode == 'prefill' else query[:, -1:]
        k = key if mode == 'prefill' else key[:, -1:]
        v = value if mode == 'prefill' else value[:, -1:]
        out = backend.execute(q.reshape(-1, 24, 256), k.reshape(-1, 4, 256), v.reshape(-1, 4, 256), layer)
        decoded = _decoded(backend, table, 129)
        phases[mode] = _close(out, _reference(q, decoded, causal=mode == 'prefill',
                                            fp8_query=mode != 'decode' and candidate[1] == 'fa3'))
    metadata = _metadata(2, 129, 128, 'decode', table)
    backend.prepare(metadata, graph_mode=True)
    q = query[:, -1:].contiguous().reshape(2, 24, 256)
    k = key[:, -1:].contiguous().reshape(2, 4, 256)
    v = value[:, -1:].contiguous().reshape(2, 4, 256)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            backend.execute(q, k, v, layer)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = backend.execute(q, k, v, layer)
    graph.replay()
    original = captured.clone()
    q.add_(.125)
    k.mul_(1.1)
    v.add_(.25)
    graph.replay()
    phases['changed_input_graph_replay'] = _close(captured, _reference(q.reshape(2, 1, 24, 256),
                                                                    _decoded(backend, table, 129), causal=False,
                                                                    fp8_query=False))
    if torch.equal(original, captured):
        raise ValueError('Graph replay did not observe changed inputs')
    saved = tuple(t.clone() for t in owners)
    ignored = torch.tensor([-1], dtype=torch.int32, device='cuda')
    _writer(backend, k[:1], v[:1], ignored)
    if not all(torch.equal(a, b) for a, b in zip(owners, saved)):
        raise ValueError('Negative slot modified persistent pages')
    for owner in owners:
        owner[-1].copy_(owner[1])
        if not torch.equal(owner[-1], owner[1]):
            raise ValueError('Whole-page copy did not preserve payload and scales')
    if (addresses != tuple(t.data_ptr() for t in owners) or not torch.equal(state.conv, state_before[0])
            or not torch.equal(state.ssm, state_before[1])):
        raise ValueError('Native operations changed storage addresses or GDN state')
    result = {'candidate': candidate, 'shape': {'heads': 24, 'kv_heads': 4, 'head_dim': 256, 'page_size': 128},
              'phases': phases, 'negative_slots_skipped': True, 'whole_page_copy': True,
              'GDN_unchanged': True, 'persistent_addresses_unchanged': True,
              'strided_projection_query_verified': True, 'query_fixture_strides': list(query.stride()),
              'native_runtime': backend.native_kv_runtime_info(), 'GPU_operator_qualification': True,
              'model_or_service_quality': False}
    write_new_json(directory / 'qualification.json', result)
    return result


def _graph_us(function: object) -> float:
    import torch

    for _ in range(3):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = function()
    graph.replay()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(50):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / 50


def _timings(candidate: tuple[str, str, str]) -> list[dict]:
    import torch

    rows = []
    for batch in (1, 16, 32):
        for context in (1024, 8192):
            backend, _, _, table, key, value = _fixture(candidate, batch, context)
            full = _metadata(batch, context, 128, 'prefill', table)
            _writer(backend, key.flatten(0, 1), value.flatten(0, 1), full.slot_mapping)
            metadata = _metadata(batch, context, 128, 'decode', table)
            backend.prepare(metadata, graph_mode=True)
            q = torch.randn((batch, 24, 256), dtype=torch.bfloat16, device='cuda')
            k, v = key[:, -1].contiguous(), value[:, -1].contiguous()
            layer = SimpleNamespace(layer_id=1)
            full_us = _graph_us(lambda: backend.execute(q, k, v, layer))
            writer_us = _graph_us(lambda: _writer(backend, k, v, metadata.slot_mapping))
            pages = backend._pages[1]
            if candidate[2] == 'xqa':
                direct = lambda: backend._active_xqa_plan.run(q, pages, backend._decode_workspace,
                                                            backend.scale, *backend._fixed_scales)
            else:
                options = {'kv_cache_sf': backend._block_scales[1]} if candidate[0] == 'nvfp4' else {}
                direct = lambda: backend._active_decode_wrapper.run(q, pages, k_scale=backend._fixed_scales[0],
                                                                   v_scale=backend._fixed_scales[1], **options)
            attention_us = _graph_us(direct)
            layout_comparison = None
            if candidate[0] == 'nvfp4':
                contiguous_pages = tuple(t.contiguous() for t in pages)
                contiguous_scales = tuple(t.contiguous() for t in backend._block_scales[1])
                contiguous_us = _graph_us(lambda: backend._active_decode_wrapper.run(
                    q, contiguous_pages, kv_cache_sf=contiguous_scales,
                    k_scale=backend._fixed_scales[0], v_scale=backend._fixed_scales[1]))
                actual = direct()
                expected = backend._active_decode_wrapper.run(
                    q, contiguous_pages, kv_cache_sf=contiguous_scales,
                    k_scale=backend._fixed_scales[0], v_scale=backend._fixed_scales[1])
                torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
                layout_comparison = {'same_direct_FlashInfer_with_contiguous_separate_planes_us': contiguous_us,
                                     'paged_owner_plane_aliases_us': attention_us}
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            for _ in range(50):
                backend.prepare(metadata, graph_mode=True)
            torch.cuda.synchronize()
            prepare_us = (time.perf_counter_ns() - started) / 50 / 1000
            rows.append({'batch': batch, 'context': context, 'native_decode_writer_attention_graph_us': full_us,
                         'writer_graph_us': writer_us, 'direct_FlashInfer_attention_graph_us': attention_us,
                         'device_only_graph_timing_not_service_latency': True,
                         'planning_outside_graph_timing': True, 'metadata_prepare_wall_us': prepare_us,
                         'declared_max_context': 262144, 'NVFP4_layout_comparison': layout_comparison, 'iterations': 50})
    return rows


def _prefill_timings(candidate: tuple[str, str, str]) -> list[dict]:
    import torch
    from xllm.python.attention.query_quantization import quantize_query_fp8

    rows = []
    for context in (129, 1024, 8192):
        backend, _, _, table, key, value = _fixture(candidate, 1, context)
        metadata = _metadata(1, context, 128, 'prefill', table)
        backend.prepare(metadata)
        query = torch.randn((context, 24, 256), dtype=torch.bfloat16, device='cuda')
        k, v = key.flatten(0, 1), value.flatten(0, 1)
        layer = SimpleNamespace(layer_id=1)
        full_us = _graph_us(lambda: backend.execute(query, k, v, layer))
        writer_us = _graph_us(lambda: _writer(backend, k, v, metadata.slot_mapping))
        fp8_q = candidate[1] == 'fa3'
        prepared_query = quantize_query_fp8(query, 1.) if fp8_q else query
        options = {'kv_cache_sf': backend._block_scales[1]} if candidate[0] == 'nvfp4' else {}
        if fp8_q:
            options['q_scale'] = 1.
        direct = lambda: backend._prefill_paged_wrapper.run(prepared_query, backend._pages[1],
            k_scale=backend._fixed_scales[0], v_scale=backend._fixed_scales[1], **options)
        direct_us = _graph_us(direct)
        q_us = _graph_us(lambda: quantize_query_fp8(query, 1.)) if fp8_q else 0.
        rows.append({'batch': 1, 'context': context, 'native_prefill_writer_query_attention_us': full_us,
                     'writer_us': writer_us, 'query_quantization_us': q_us,
                     'direct_FlashInfer_attention_us': direct_us, 'iterations': 50,
                     'stable_operator_graph_timing_only': True,
                     'production_prefill_in_Graph_claim': False, 'planning_excluded': True})
    return rows


def run(output: Path) -> dict:
    import torch
    import flashinfer

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (9, 0):
        raise ValueError('Operator qualification requires the single admitted SM90 GPU')
    torch.manual_seed(20261102)
    output.mkdir(exist_ok=False)
    results = []
    for candidate in CANDIDATES:
        directory = output / ('-'.join(candidate))
        directory.mkdir()
        try:
            result = _qualify(candidate, directory)
            result['timings'] = _timings(candidate)
            result['prefill_timings'] = _prefill_timings(candidate)
            write_new_json(directory / 'timings.json', result['timings'])
            write_new_json(directory / 'prefill_timings.json', result['prefill_timings'])
            result['status'] = 'PASS'
        except Exception as error:
            result = {'candidate': candidate, 'status': 'FAILED', 'GPU_operator_qualification': False,
                      'error': type(error).__name__+': '+str(error), 'traceback': traceback.format_exc()}
            write_new_json(directory/'failure.json', result)
            # A device fault can poison the context. In that case stop this
            # worker, rather than attributing cascading failures to other candidates.
            torch.cuda.synchronize()
        results.append(result)
    summary = {'schema': 'latest-upstream-native-SM90-operator-v1', 'candidates': results,
               'flashinfer_version': importlib.metadata.version('flashinfer-python'),
               'flashinfer_source': flashinfer.__file__, 'torch_version': torch.__version__,
               'GPU_name': torch.cuda.get_device_name(0), 'model_or_service_benchmark': False}
    package = Path(flashinfer.__file__).parent
    summary['flashinfer_python_source_sha256'] = {
        name: hashlib.sha256((package/name).read_bytes()).hexdigest()
        for name in ('page.py', 'prefill.py', 'decode.py', 'jit/attention/modules.py')
        if (package/name).is_file()}
    summary['all_candidates_pass'] = all(row['status'] == 'PASS' for row in results)
    summary['qualified_candidates'] = [row['candidate'] for row in results if row['status'] == 'PASS']
    write_new_json(output / 'acceptance.json', summary)
    if not summary['qualified_candidates']:
        raise ValueError('All frozen native operator candidates failed; original failures retained')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == '__main__':
    main()
