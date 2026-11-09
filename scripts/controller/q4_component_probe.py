# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Fixed-shape MX/NV operator diagnostics; call run() only in an admitted GPU job.

Import, --describe, and CPU tests never import Torch or initialize CUDA.
This is a FlashInfer operator proxy, not a vLLM engine or service benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import time
import traceback
from typing import Any, Callable


Q_HEADS, KV_HEADS, DIMENSION, PAGE_SIZE = 24, 4, 256, 128
WARMUP, REPEATS, INNER = 10, 40, 5
WRITER_GRAPH_BURST = 32
ATTENTION_GRAPH_BURST = 8
PLAN_WARMUP, PLAN_REPEATS = 3, 20
REFERENCE_ATOL, REFERENCE_RTOL = 0.02, 0.02
ROUTE_ATOL, ROUTE_RTOL = 0.001, 0.001
WORKSPACE_BYTES = 128 * 2**20
SEED = 20261106
BLOCKED_GRAPH_ROUTES = ("mx_prefill_wrapper", "nv_prefill_wrapper", "matched_nv_prefill_wrapper")
BLOCKED_GRAPH_REASON = "pinned_FA2_graph_ready_non_split_padding_mask_defect"
FI_COMMIT = "8bc3b578027791336c6ae87db5c9d76f82cef8bc"
FI_HASHES = {
    "page.py": "4a20f05410442ff415a44337c7c6246d90777e4c177852b2830fae3cf00eab04",
    "prefill.py": "2ad12a8387b3f6bff192e5945769b68d90cfcab00b9eb2a8a524af8dd71a29de",
    "decode.py": "d82d107a644596a9349780b839b34e690c50169ea4cea3d0ee02b78e9dc88c1a",
    "data/include/flashinfer/attention/prefill.cuh": "f93cbea96e3a723254c15f5a52d6dd4c13e3892e88eb9b9fed1c0c0048c7d710",
}
MX_HASHES = {
    "mxfp4_flashinfer": "282c76bac87b6256e4c98e5f0102a406517df843e3df803fd5c21a92d69a1d36",
    "mxfp4_writer": "c2ac430b94e9b75cb276bd7750fcc119451f15f652d4ad2ea292d05b284b5dca",
    "mxfp4_reference": "e657a7960cc8632ec50b92e6d8bedb989d57d40c503ab547e5dd2af5076b2671",
    "native_kv_layout": "ebcaa32cdbc56692cd309fec6cb6d10f23ab37dc97fc3998ff16e35a06be060d",
}
# Set only inside run(). Keeping imports cold makes CPU validation safe.
torch: Any = None
flashinfer: Any = None
mx_writer: Any = None
mx_backend: Any = None
mx_reference: Any = None
nv_layout: Any = None


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    context: int
    query_length: int


CASES = (
    Case("prefill_b1_s8192", 1, 8192, 8192),
    Case("decode_b32_s2048", 32, 2048, 1),
    Case("decode_b32_s16384", 32, 16384, 1),
)
PILOT_CASES = (Case("pilot_prefill_b1_s128", 1, 128, 128),
               Case("pilot_decode_b2_s129", 2, 129, 1))


def describe(phase: str = "main") -> dict[str, Any]:
    if phase not in ("pilot", "main"):
        raise ValueError("phase must be pilot or main")
    return {
        "schema": "q4_supported_component_probe_v3",
        "phase": phase, "pilot_timings_excluded_from_main_tables": phase == "pilot",
        "cases": [asdict(case) for case in (PILOT_CASES if phase == "pilot" else CASES)],
        "geometry": {"Q": Q_HEADS, "KV": KV_HEADS, "D": DIMENSION, "page": PAGE_SIZE},
        "dtype": "BF16", "seed": SEED,
        "timing": {"warmup": WARMUP, "repeats": REPEATS, "inner": INNER,
                   "writer_operations_per_graph_replay": WRITER_GRAPH_BURST,
                   "attention_operations_per_graph_replay": ATTENTION_GRAPH_BURST,
                   "plan_warmup": PLAN_WARMUP, "plan_repeats": PLAN_REPEATS},
        "reference_tolerance": {"atol": REFERENCE_ATOL, "rtol": REFERENCE_RTOL},
        "route_tolerance": {"atol": ROUTE_ATOL, "rtol": ROUTE_RTOL},
        "qualification_policy": {
            "same_fused_MX_NV_control": "hard gate at unchanged route tolerance",
            "prefill_vs_native_decode": "all strict results retained as diagnostics; equivalence may be false",
            "independent_reference": "hard gate for every measured eager route",
            "writer_Graph_and_native_decode_Graph": "hard gates for these supported components only",
            "blocked_graph_routes": list(BLOCKED_GRAPH_ROUTES),
            "blocked_graph_reason": BLOCKED_GRAPH_REASON,
            "blocked_Graph_wrapper_constructed": False,
            "all_component_gpu_activity_observed": "hard instrument gate in pilot and main",
            "CUPTI": "2 actual eager API calls per component, each in its own profiler context; after each case primary timings",
            "prior_failed_runs_preserved": ["upstream-fast-q4audit-20261106a-pilot-0001",
                                           "upstream-fast-q4audit-20261106b-pilot-0001"],
        },
        "sglang": {"status": "unsupported_Q24_KV4_D256", "timings": None},
        "proxy": "pinned FlashInfer operators; not full vLLM engine",
        "flashinfer_commit": FI_COMMIT, "flashinfer_version": "0.6.18.post1",
        "production_source_changes": False,
        "scale_unpack_independent_wall_percent": None,
        "full_prefill_reference": "fixed 16 query rows; all 8192 rows execute",
    }


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _summary(values: list[float]) -> dict[str, Any]:
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Timing samples must be finite nonnegative values")
    ordered = sorted(values)
    return {"median": statistics.median(values), "mean": statistics.mean(values),
            "min": ordered[0], "max": ordered[-1],
            "p10": ordered[int((len(values) - 1) * 0.1)],
            "p90": ordered[int((len(values) - 1) * 0.9)], "samples": values}


def _event_time(label: str, operation: Callable[[], Any], normalization: int = 1) -> dict[str, Any]:
    for _ in range(WARMUP):
        operation()
    torch.cuda.synchronize()
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(REPEATS)]
    torch.cuda.nvtx.range_push(label)
    try:
        for start, end in pairs:
            start.record()
            for _ in range(INNER):
                operation()
            end.record()
        pairs[-1][1].synchronize()
    finally:
        torch.cuda.nvtx.range_pop()
    return {"unit": "ms_per_call", "measurement": "CUDA event stream elapsed",
            "includes_possible_host_dispatch_gaps": True,
            "operations_per_callback": normalization,
            **_summary([start.elapsed_time(end) / (INNER * normalization) for start, end in pairs])}


def _host_time(label: str, operation: Callable[[], Any]) -> dict[str, Any]:
    for _ in range(PLAN_WARMUP):
        operation()
    torch.cuda.synchronize()
    api, drained = [], []
    torch.cuda.nvtx.range_push(label)
    try:
        for _ in range(PLAN_REPEATS):
            start = time.perf_counter_ns()
            operation()
            returned = time.perf_counter_ns()
            torch.cuda.synchronize()
            finished = time.perf_counter_ns()
            api.append((returned - start) / 1e3)
            drained.append((finished - start) / 1e3)
    finally:
        torch.cuda.nvtx.range_pop()
    return {"unit": "us", "cpu_api_return": _summary(api),
            "wall_including_async_drain": _summary(drained),
            "per_model_step_count": 1, "multiply_by_layers": False,
            "isolated_copy_latency": None}


def _memory() -> dict[str, int]:
    torch.cuda.synchronize()
    return {"allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "peak_allocated": torch.cuda.max_memory_allocated()}


def _tensor_sha(tensor: Any) -> str:
    digest = hashlib.sha256()
    row_bytes = tensor[0].numel() * tensor.element_size()
    rows = max(1, (4 * 2**20) // row_bytes)
    for first in range(0, tensor.shape[0], rows):
        digest.update(tensor[first:first + rows].contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _save_tensor(directory: Path, tensor: Any, *, full: bool = True,
                 sample: Any = None, statistics_required: bool = False) -> dict[str, Any]:
    """Pilot stores full tensors; main retains full byte SHA and fixed raw samples."""
    if not full and sample is None:
        raise ValueError("Main tensor archival requires explicit fixed samples")
    directory.mkdir()
    sha = _tensor_sha(tensor)
    stats = None
    if statistics_required:
        values = tensor.float()
        stats = {"finite": bool(torch.isfinite(values).all()), "max": values.max().item(),
                 "min": values.min().item(), "mean": values.mean().item(),
                 "max_abs": values.abs().max().item()}
        stats = {name: (str(value) if isinstance(value, float) and not math.isfinite(value) else value)
                 for name, value in stats.items()}
        del values
    saved = tensor if full else sample
    chunks = []
    for first in range(0, saved.shape[0], 32768):
        path = directory / f"rows-{first:08d}.pt"
        cpu = saved[first:first + 32768].detach().contiguous().cpu()
        with path.open("xb") as handle:
            torch.save(cpu, handle)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 2**20), b""):
                digest.update(block)
        chunks.append({"path": path.name, "first_row": first, "rows": cpu.shape[0],
                       "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    manifest = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "chunks": chunks,
                "full_tensor_body_retained": full, "full_tensor_byte_sha256": sha,
                "saved_shape": list(saved.shape), "full_tensor_statistics": stats}
    _write_json(directory / "manifest.json", manifest)
    return manifest


def _errors(actual: Any, expected: Any, atol: float, rtol: float) -> dict[str, Any]:
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise ValueError("Nonfinite attention/reference result")
    difference = actual.float() - expected.float()
    passed = bool(torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol))
    return {"pass": passed, "max_abs": difference.abs().max().item(),
            "rmse": difference.square().mean().sqrt().item(),
            "relative_l2": (difference.norm() / expected.float().norm().clamp_min(1e-30)).item(),
            "atol": atol, "rtol": rtol}


def _route_gates(case: Case, comparisons: dict[str, Any]) -> dict[str, bool]:
    same = "mx_prefill_wrapper__vs__matched_nv_prefill_wrapper"
    native = ["nv_prefill_wrapper__vs__nv_native_decode",
              "matched_nv_prefill_wrapper__vs__matched_nv_native_decode"] if case.query_length == 1 else []
    if set(comparisons) != {same, *native}:
        raise ValueError("Missing or unexpected strict route comparison")
    if any(type(check.get("pass")) is not bool for check in comparisons.values()):
        raise ValueError("Strict route comparison must retain its boolean outcome")
    return {"same_fused_MX_NV_control_passed": comparisons[same]["pass"],
            "strict_prefill_native_decode_equivalence_all_pass": all(comparisons[name]["pass"] for name in native),
            "strict_prefill_native_decode_equivalence_applicable": bool(native)}


def _acceptance_gates(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases or any(not case["routes"] for case in cases):
        raise ValueError("Supported-component admission requires completed case and route results")
    reference = all(case["primary_independent_reference_passed"] is True and
                    all(route["independent_decoded_reference"]["pass"] is True
                        for route in case["routes"].values()) for case in cases)
    same = all(case["route_comparison_gates"]["same_fused_MX_NV_control_passed"] is True for case in cases)
    writer_graph = all(set(case["writers"]) == {"mx", "nv"} and
                       all(writer["graph_correctness"][key] is True for writer in case["writers"].values()
                           for key in ("pass", "compressed_current_rows_changed", "captured_writer_bytes_equal_direct",
                                       "original_compressed_rows_restored")) for case in cases)
    native = []
    for case in cases:
        for name, route in case["routes"].items():
            if name in BLOCKED_GRAPH_ROUTES:
                if not (route["graph_status"] == "BLOCKED" and route["graph_blocked_reason"] == BLOCKED_GRAPH_REASON
                        and route["attention_graph"] is None and route["graph_ready_reference"] is None
                        and route["plan_info_graph"] is None):
                    raise ValueError("Blocked prefill Graph must remain explicitly unexecuted")
            elif name in ("nv_native_decode", "matched_nv_native_decode"):
                native.append((name, route))
            else:
                raise ValueError("Unexpected component route")
    native_graph = bool(native) and all(
        route["graph_status"] == "QUALIFIED_NATIVE_DECODE" and route["graph_ready_reference"]["pass"] is True and
        route["graph_direct_bitwise_equal"] is True and route["graph_replan_bitwise_equal"] is True and
        (name.startswith("matched_") or (route["graph_changed_inputs"] is not None and
         all(route["graph_changed_inputs"][key] is True for key in
             ("bitwise_equal_direct", "captured_writer_bytes_equal_direct", "compressed_current_rows_changed",
              "output_changed", "writer_and_attention_captured", "query_and_current_KV_changed"))))
        for name, route in native)
    activity = all(case["all_component_gpu_activity_observed"] is True for case in cases)
    gates = {"all_independent_reference_passed": reference, "same_fused_MX_NV_control_passed": same,
             "all_writer_graph_checks_passed": writer_graph, "all_native_decode_graph_checks_passed": native_graph,
             "all_component_gpu_activity_observed": activity,
             "strict_prefill_native_decode_equivalence_all_pass":
                 all(case["route_comparison_gates"]["strict_prefill_native_decode_equivalence_all_pass"] is True
                     for case in cases),
             "all_requested_graph_paths_supported": False, "blocked_graph_routes": list(BLOCKED_GRAPH_ROUTES),
             "blocked_graph_reason": BLOCKED_GRAPH_REASON, "production_graph_qualification": False}
    if not (reference and same and writer_graph and native_graph and activity):
        raise ValueError("Supported-component reference, control, Writer/native Graph and GPU-activity gates must pass")
    return gates


def _reference_rows(case: Case) -> list[int]:
    if case.query_length == 1:
        return [0]
    return [row for row in [0, 1, 2, 127, 128, 129, 255, 256, 511, 512, 1023, 2047, 4095, 6143, 8190, 8191]
            if row < case.query_length]


def _sample_output(value: Any, case: Case) -> Any:
    return value.reshape(case.batch, case.query_length, Q_HEADS, DIMENSION)[:, _reference_rows(case)]


class _Cache:
    def __init__(self, kind: str, page_count: int) -> None:
        self.kind = kind
        width = DIMENSION // 2 + DIMENSION // (32 if kind == "mx" else 16)
        self.owners = tuple(torch.zeros((page_count, PAGE_SIZE, KV_HEADS, width),
                                       device="cuda", dtype=torch.uint8) for _ in range(2))
        view_function = mx_writer.mxfp4_page_views if kind == "mx" else nv_layout.nvfp4_page_views
        views = tuple(view_function(owner, DIMENSION) for owner in self.owners)
        self.pages = tuple(view[0] for view in views)
        self.scales = tuple(view[1] for view in views)
        self.global_scales = tuple(torch.ones(1, device="cuda", dtype=torch.float32) for _ in range(2)) if kind == "nv" else ()

    def _write(self, values: tuple[Any, Any], slots: Any) -> None:
        if self.kind == "mx":
            mx_writer.write_mxfp4_kv(*values, slots, self.pages, self.scales)
        else:
            flashinfer.page.nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
                *values, slots, self.pages, self.scales, *self.global_scales, kv_layout="NHD")

    def _bytes(self) -> int:
        return sum(owner.numel() for owner in self.owners)


def _same_decoded_nv(mx: _Cache, valid_slots: Any) -> tuple[_Cache, dict[str, Any]]:
    nv = _Cache("nv", mx.owners[0].shape[0])
    valid = torch.zeros(mx.owners[0].shape[:2], device="cuda", dtype=torch.bool)
    valid[valid_slots // PAGE_SIZE, valid_slots % PAGE_SIZE] = True
    for mx_data, mx_sf, nv_data, nv_sf in zip(mx.pages, mx.scales, nv.pages, nv.scales):
        # Unwritten rows have zero payload; a unit NV scale preserves that zero.
        exponent = mx_sf.to(torch.int32) - 127
        power = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)
        power = torch.where(valid[:, :, None, None], power, 1.0)
        encoded = power.to(torch.float8_e4m3fn)
        if not bool(torch.isfinite(encoded.float()).all()) or not torch.equal(encoded.float(), power):
            raise ValueError("Fixed MX input produced scales outside exact E4M3 range; no scale search allowed")
        nv_data.copy_(mx_data)
        nv_sf.view(torch.uint8).copy_(encoded.view(torch.uint8).repeat_interleave(2, dim=-1))
        if bool(torch.count_nonzero(mx_data[~valid])) or bool(torch.count_nonzero(nv_data[~valid])):
            raise ValueError("Unwritten control rows must remain zero payload")
        if not torch.equal(nv_data, mx_data):
            raise ValueError("Matched control payload copy mismatch")
    return nv, {"payload_bitwise_equal": True, "scales_exact": True,
                "mapping": "one block32 power-of-two E8M0 scale -> two equal block16 E4M3 scales",
                "unwritten_zero_payload_rows_use_unit_NV_scale": True,
                "unwritten_rows_decode_zero_in_both_formats": True,
                "global_scales": [1.0, 1.0], "writer_benchmarked": False}


def _decode_sequence(cache: _Cache, table_row: Any, context: int) -> tuple[Any, Any]:
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                       device="cuda", dtype=torch.float32)
    values = []
    for data, scale in zip(cache.pages, cache.scales):
        packed = data.index_select(0, table_row.long()).reshape(-1, KV_HEADS, DIMENSION // 2)[:context]
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(context, KV_HEADS, DIMENSION)
        block = 32 if cache.kind == "mx" else 16
        sf = scale.view(torch.uint8).index_select(0, table_row.long()).reshape(-1, KV_HEADS, DIMENSION // block)[:context]
        if cache.kind == "mx":
            exponents = sf.to(torch.int32) - 127
            real_scale = torch.ldexp(torch.ones_like(exponents, dtype=torch.float32), exponents)
            real_scale = torch.where(sf == 255, float("nan"), real_scale).repeat_interleave(32, dim=-1)
        else:
            real_scale = sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
        values.append((lut[codes.long()] * real_scale).to(torch.bfloat16).float())
    return tuple(values)


def _reference(case: Case, query: Any, table: Any, cache: _Cache | None,
               original: tuple[Any, Any] | None = None) -> Any:
    rows = _reference_rows(case)
    outputs = []
    for batch in range(case.batch):
        if cache is not None:
            key, value = _decode_sequence(cache, table[batch], case.context)
        else:
            assert original is not None
            key, value = (tensor.reshape(case.batch, case.context, KV_HEADS, DIMENSION)[batch].float()
                          for tensor in original)
        q = query.reshape(case.batch, case.query_length, Q_HEADS, DIMENSION)[batch, rows].float()
        grouped = q.reshape(len(rows), KV_HEADS, Q_HEADS // KV_HEADS, DIMENSION)
        logits = torch.einsum("rhgd,shd->hgrs", grouped, key) * DIMENSION**-0.5
        if case.query_length > 1:
            allowed = torch.arange(case.context, device="cuda")[None, :] <= torch.tensor(rows, device="cuda")[:, None]
            logits.masked_fill_(~allowed[None, None], float("-inf"))
        probability = torch.softmax(logits, dim=-1)
        outputs.append(torch.einsum("hgrs,shd->rhgd", probability, value).reshape(len(rows), Q_HEADS, DIMENSION))
    return torch.stack(outputs)


def _writer_sample(case: Case, cache: _Cache, values: tuple[Any, Any], slots: Any) -> dict[str, Any]:
    indices = sorted(index for index in {0, 1, 127, 128, 129, case.batch * case.context - 1,
                      *[(batch + 1) * case.context - 1 for batch in range(case.batch)]}
                     if index < case.batch * case.context)
    sample = torch.tensor(indices, device="cuda", dtype=torch.int64)
    physical = slots[sample]
    if cache.kind == "mx":
        import numpy as np

        for source, data, scale in zip(values, cache.pages, cache.scales):
            packed, sf = mx_reference.encode_mxfp4(source[sample].float().cpu().numpy())
            np.testing.assert_array_equal(data[physical // PAGE_SIZE, physical % PAGE_SIZE].cpu().numpy(), packed)
            np.testing.assert_array_equal(scale[physical // PAGE_SIZE, physical % PAGE_SIZE].cpu().numpy(), sf)
    if any(bool(torch.count_nonzero(owner[0])) for owner in cache.owners):
        raise ValueError("Unused page 0 changed during native writer")
    return {"sample_rows": indices, "MX_independent_CPU_byte_equal": cache.kind == "mx",
            "NV_independent_CPU_byte_oracle": None,
            "NV_note": "actual compressed bytes independently decoded for attention reference; no claimed CPU writer byte oracle",
            "unused_page0_unchanged": True}


class _Route:
    def __init__(self, name: str, cache: _Cache, case: Case, table: Any, native_decode: bool) -> None:
        self.name, self.cache, self.case, self.native_decode = name, cache, case, native_decode
        self.host_indptr = torch.arange(case.batch + 1, dtype=torch.int32) * table.shape[1]
        self.host_tails = torch.full((case.batch,), (case.context - 1) % PAGE_SIZE + 1, dtype=torch.int32)
        self.host_qo = torch.arange(case.batch + 1, dtype=torch.int32) * case.query_length
        self.indices = table.flatten().contiguous()
        self.gpu_indptr = self.gpu_tails = self.gpu_qo = self.graph_indices = None
        if native_decode:
            self.gpu_indptr, self.gpu_tails, self.gpu_qo = (t.cuda() for t in (self.host_indptr, self.host_tails, self.host_qo))
            self.graph_indices = self.indices.clone()
        # No Graph wrapper, plan or output is constructed for the blocked prefill routes.
        self.workspaces = [torch.zeros(WORKSPACE_BYTES, device="cuda", dtype=torch.uint8)
                           for _ in range(2 if native_decode else 1)]
        self.eager = self._make_wrapper(False, self.workspaces[0])
        self.graph_wrapper = self._make_wrapper(True, self.workspaces[1]) if native_decode else None
        self._plan(False)
        if native_decode:
            self._plan(True)
        shape = (case.batch * case.query_length, Q_HEADS, DIMENSION)
        self.eager_out = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
        self.graph_out = torch.empty_like(self.eager_out) if native_decode else None

    def _make_wrapper(self, graph: bool, workspace: Any) -> Any:
        if graph and not self.native_decode:
            raise ValueError(BLOCKED_GRAPH_REASON)
        if self.native_decode:
            return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace, "NHD", use_tensor_cores=True, backend="fa2", use_cuda_graph=graph,
                paged_kv_indptr_buffer=self.gpu_indptr if graph else None,
                paged_kv_indices_buffer=self.graph_indices if graph else None,
                paged_kv_last_page_len_buffer=self.gpu_tails if graph else None)
        options = dict(use_cuda_graph=graph, qo_indptr_buf=self.gpu_qo if graph else None,
                       paged_kv_indptr_buf=self.gpu_indptr if graph else None,
                       paged_kv_indices_buf=self.graph_indices if graph else None,
                       paged_kv_last_page_len_buf=self.gpu_tails if graph else None)
        if self.cache.kind == "mx":
            return mx_backend.make_mxfp4_wrapper(workspace, **options)
        return flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2", **options)

    def _plan(self, graph: bool, fast: bool = False) -> None:
        if graph and not self.native_decode:
            raise ValueError(BLOCKED_GRAPH_REASON)
        wrapper = self.graph_wrapper if graph else self.eager
        options = dict(q_data_type=torch.bfloat16, kv_data_type=torch.uint8, sm_scale=DIMENSION**-0.5)
        arguments = (self.host_indptr, self.indices, self.host_tails,
                     Q_HEADS, KV_HEADS, DIMENSION, PAGE_SIZE)
        if self.native_decode:
            if fast:
                flashinfer.decode.fast_decode_plan(wrapper, *arguments,
                    global_override_indptr_cpu=self.host_indptr, **options)
            else:
                wrapper.plan(*arguments, **options)
        else:
            # Preserve and explicitly request the installed uint8 split-KV protection.
            wrapper.plan(self.host_qo, *arguments, causal=True, o_data_type=torch.bfloat16,
                         disable_split_kv=True, **options)

    def _stage(self) -> None:
        if not self.native_decode:
            raise ValueError(BLOCKED_GRAPH_REASON)
        self.gpu_indptr.copy_(self.host_indptr, non_blocking=True)
        self.gpu_tails.copy_(self.host_tails, non_blocking=True)
        if not self.native_decode:
            self.gpu_qo.copy_(self.host_qo, non_blocking=True)
        self.graph_indices.copy_(self.indices, non_blocking=True)

    def _attention(self, query: Any, graph: bool = False) -> Any:
        if graph and not self.native_decode:
            raise ValueError(BLOCKED_GRAPH_REASON)
        wrapper = self.graph_wrapper if graph else self.eager
        output = self.graph_out if graph else self.eager_out
        return wrapper.run(query, self.cache.pages, kv_cache_sf=self.cache.scales,
                           k_scale=1.0, v_scale=1.0, out=output)


def _capture(operation: Callable[[], Any], repeats: int = 1) -> Any:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            operation()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            operation()
    return graph


def _current_bytes(cache: _Cache, slots: Any) -> tuple[Any, ...]:
    return tuple(tensor.view(torch.uint8)[slots // PAGE_SIZE, slots % PAGE_SIZE].clone()
                 for tensor in (*cache.pages, *cache.scales))


def _require_native_graph_plan(plan: Any) -> list[int]:
    values = list(plan)
    if not (len(values) == 15 and all(type(values[index]) is int for index in (12, 13, 14))
            and values[13] == 1 and values[14] == 1 and values[12] > 0):
        raise ValueError("Unsafe native Graph plan: require 15 entries, Graph=1, split=1 and positive mask offset; got " + repr(values))
    return values


def _profile_native_route(route: _Route, query: Any, current: tuple[Any, Any], current_slots: Any,
                   expected: Any, output: Path, matched_control: bool, full_archive: bool) -> dict[str, Any]:
    actual = route._attention(query)
    conformance = _errors(_sample_output(actual, route.case), expected, REFERENCE_ATOL, REFERENCE_RTOL)
    if not conformance["pass"]:
        _write_json(output / (route.name + "-failed-reference.json"), conformance)
        raise ValueError(route.name + " differs from independent decoded reference")
    graph_plan_before = _require_native_graph_plan(route.graph_wrapper._plan_info)
    _write_json(output / (route.name + "-graph-plan-before-execution.json"), {"pass": True, "plan_info": graph_plan_before})
    graph_conformance = _errors(_sample_output(route._attention(query, True), route.case),
                                expected, REFERENCE_ATOL, REFERENCE_RTOL)
    if not graph_conformance["pass"]:
        raise ValueError(route.name + " graph-ready planner differs from reference")
    graph = _capture(lambda: route._attention(query, True))
    graph.replay()
    observed = route.graph_out.clone()
    if not torch.equal(observed, route._attention(query, True)):
        raise ValueError("Attention Graph/direct mismatch")
    del observed
    changed = None
    if not matched_control:
        old_query = query.clone()
        changed_values = tuple(value.clone() for value in current)

        def writer_attention() -> None:
            route.cache._write(changed_values, current_slots)
            route._attention(query, True)

        combined = _capture(writer_attention)
        combined.replay()
        initial = route.graph_out.clone()
        initial_bytes = _current_bytes(route.cache, current_slots)
        query.add_(0.125)
        changed_values[0].mul_(1.125)
        changed_values[1].add_(0.25)
        combined.replay()
        observed = route.graph_out.clone()
        captured_bytes = _current_bytes(route.cache, current_slots)
        route.cache._write(changed_values, current_slots)
        direct_bytes = _current_bytes(route.cache, current_slots)
        if all(torch.equal(a, b) for a, b in zip(initial_bytes, captured_bytes)):
            raise ValueError("Captured writer did not change compressed current rows")
        if not all(torch.equal(a, b) for a, b in zip(captured_bytes, direct_bytes)):
            raise ValueError("Captured writer bytes differ from direct writer")
        # Direct run uses the same rewritten packed cache; this is graph conformance,
        # separate from the independent numerical reference check above.
        direct = route._attention(query, True)
        if torch.equal(initial, observed) or not torch.equal(observed, direct):
            raise ValueError("Changed-Q/K/V Graph failed")
        graph_raw = _save_tensor(output / (route.name + "-graph-changed-output"), observed,
                                 full=full_archive, sample=_sample_output(observed, route.case),
                                 statistics_required=True)
        if not graph_raw["full_tensor_statistics"]["finite"]:
            raise ValueError("Changed-input Graph output is nonfinite")
        changed = {"output_changed": True, "bitwise_equal_direct": True,
                   "writer_and_attention_captured": True,
                   "compressed_current_rows_changed": True,
                   "captured_writer_bytes_equal_direct": True,
                   "captured_full_output_byte_sha256": graph_raw["full_tensor_byte_sha256"],
                   "direct_full_output_byte_sha256": _tensor_sha(direct),
                   "query_and_current_KV_changed": True, "dynamic_lengths_tested": False}
        query.copy_(old_query)
        route.cache._write(current, current_slots)
        if not all(torch.equal(a, b) for a, b in zip(initial_bytes, _current_bytes(route.cache, current_slots))):
            raise ValueError("Failed to restore original compressed input after Graph validation")
        del combined, old_query, changed_values, initial, observed, direct, initial_bytes, captured_bytes, direct_bytes
    host = {"eager_plan_host_metadata": _host_time(route.name + "/eager_plan", lambda: route._plan(False)),
            "graph_full_plan": _host_time(route.name + "/graph_plan", lambda: route._plan(True)),
            "explicit_metadata_staging": _host_time(route.name + "/metadata_copy", route._stage)}
    if route.native_decode:
        host["graph_fast_decode_plan"] = _host_time(route.name + "/fast_plan", lambda: route._plan(True, True))
    route._stage()
    route._plan(True, route.native_decode)
    graph_plan_after = _require_native_graph_plan(route.graph_wrapper._plan_info)
    _write_json(output / (route.name + "-graph-plan-after-fast-replan.json"), {"pass": True, "plan_info": graph_plan_after})
    # Plans are refreshed outside capture; verify the old captured graph remains correct.
    graph.replay()
    captured = route.graph_out.clone()
    direct = route._attention(query, True)
    if not torch.equal(captured, direct):
        raise ValueError("Graph changed after same-geometry replanning")
    del captured, direct
    timing_graph = _capture(lambda: route._attention(query, True), ATTENTION_GRAPH_BURST)
    result = {
        "independent_decoded_reference": conformance,
        "graph_ready_reference": graph_conformance,
        "graph_changed_inputs": changed,
        "graph_direct_bitwise_equal": True,
        "graph_replan_bitwise_equal": True,
        "graph_status": "QUALIFIED_NATIVE_DECODE", "graph_blocked_reason": None,
        "native_graph_plan_guards": {"before_first_execution": graph_plan_before,
                                     "after_fast_replan": graph_plan_after, "pass": True},
        "attention_eager": _event_time(route.name + "/attention_eager", lambda: route._attention(query)),
        "attention_graph": _event_time(route.name + "/attention_graph", timing_graph.replay, ATTENTION_GRAPH_BURST),
        "host_prepare": host,
        "memory_at_timing": _memory(),
        "BF16_history_workspace_bytes_during_timing": 0,
        "caller_output_preallocated": True,
        "split_policy": "native decode default planner" if route.native_decode else "uint8 disable_split_kv=True",
        "metadata_note": "fixed host indptr/tails, device indices; not complete engine prepare()",
        "explicit_staging_host_buffers_pinned": all(t.is_pinned() for t in (route.host_indptr, route.host_tails, route.host_qo)),
        "metadata_staging_bytes": {"H2D": (route.case.batch * 2 + 1 +
                (0 if route.native_decode else route.case.batch + 1)) * 4,
                "D2D_indices": route.indices.numel() * 4},
        "fused_scale_unpack_included": True,
        "independent_scale_unpack_ms": None,
        "plan_info_eager": list(route.eager._plan_info),
        "plan_info_graph": list(route.graph_wrapper._plan_info),
        "resolved_backend": getattr(route.eager, "_backend", "private MX FA2"),
    }
    _write_json(output / (route.name + ".json"), result)
    return result



def _profile_route(route: _Route, query: Any, current: tuple[Any, Any], current_slots: Any,
                   expected: Any, output: Path, matched_control: bool, full_archive: bool) -> dict[str, Any]:
    if route.native_decode:
        return _profile_native_route(route, query, current, current_slots, expected, output,
                                     matched_control, full_archive)
    actual = route._attention(query)
    conformance = _errors(_sample_output(actual, route.case), expected, REFERENCE_ATOL, REFERENCE_RTOL)
    if not conformance["pass"]:
        _write_json(output / (route.name + "-failed-reference.json"), conformance)
        raise ValueError(route.name + " differs from independent decoded reference")
    host = {"eager_plan_host_metadata": _host_time(route.name + "/eager_plan", lambda: route._plan(False)),
            "graph_full_plan": None, "explicit_metadata_staging": None,
            "blocked_graph_reason": BLOCKED_GRAPH_REASON}
    result = {"independent_decoded_reference": conformance, "graph_ready_reference": None,
              "graph_changed_inputs": None, "graph_direct_bitwise_equal": None,
              "graph_replan_bitwise_equal": None, "graph_status": "BLOCKED",
              "graph_blocked_reason": BLOCKED_GRAPH_REASON,
              "attention_eager": _event_time(route.name + "/attention_eager", lambda: route._attention(query)),
              "attention_graph": None, "host_prepare": host, "memory_at_timing": _memory(),
              "BF16_history_workspace_bytes_during_timing": 0, "caller_output_preallocated": True,
              "split_policy": "uint8 disable_split_kv=True", "fused_scale_unpack_included": True,
              "independent_scale_unpack_ms": None, "plan_info_eager": list(route.eager._plan_info),
              "plan_info_graph": None, "resolved_backend": getattr(route.eager, "_backend", "private MX FA2"),
              "metadata_note": "ordinary eager host-metadata plan; blocked Graph wrapper/plan/output were never created"}
    _write_json(output / (route.name + ".json"), result)
    return result


def _qualify_writer_graph(cache: _Cache, current: tuple[Any, Any], slots: Any,
                          output: Path, full_archive: bool) -> dict[str, Any]:
    initial_bytes = _current_bytes(cache, slots)
    changed = tuple(value.clone() for value in current)
    graph = _capture(lambda: cache._write(changed, slots))
    graph.replay()
    if not all(torch.equal(a, b) for a, b in zip(initial_bytes, _current_bytes(cache, slots))):
        raise ValueError("Writer Graph changed original bytes before input mutation")
    changed[0].mul_(1.125)
    changed[1].add_(0.25)
    graph.replay()
    observed = _current_bytes(cache, slots)
    if all(torch.equal(a, b) for a, b in zip(initial_bytes, observed)):
        raise ValueError("Writer Graph did not consume changed K/V")
    cache._write(changed, slots)
    direct = _current_bytes(cache, slots)
    if not all(torch.equal(a, b) for a, b in zip(observed, direct)):
        raise ValueError("Changed-input Writer Graph bytes differ from direct writer")
    rows = sorted(index for index in {0, 1, 127, 128, 129, slots.numel() - 1} if index < slots.numel())
    saved = []
    for index, tensor in enumerate(observed):
        saved.append(_save_tensor(output / (cache.kind + "-writer-graph-changed-" + str(index)), tensor,
                                  full=full_archive, sample=tensor[rows]))
    direct_sha = [_tensor_sha(tensor) for tensor in direct]
    cache._write(current, slots)
    if not all(torch.equal(a, b) for a, b in zip(initial_bytes, _current_bytes(cache, slots))):
        raise ValueError("Writer Graph qualification did not restore original cache rows")
    return {"pass": True, "compressed_current_rows_changed": True,
            "captured_writer_bytes_equal_direct": True, "original_compressed_rows_restored": True,
            "changed_KV_inputs": True, "attention_invoked": False,
            "captured_full_byte_sha256": [entry["full_tensor_byte_sha256"] for entry in saved],
            "direct_full_byte_sha256": direct_sha,
            "scope": "standalone Writer Graph/direct conformance; not Attention Graph support"}


def _valid_gpu_activity_invocation(entry: dict[str, Any]) -> bool:
    kernels = entry.get("kernels")
    return (entry.get("status") == "GPU_events_present" and isinstance(kernels, list) and bool(kernels)
            and all(type(kernel.get("duration_us")) in (int, float) and math.isfinite(kernel["duration_us"])
                    and kernel["duration_us"] > 0 for kernel in kernels)
            and type(entry.get("GPU_kernel_active_us")) in (int, float)
            and math.isfinite(entry["GPU_kernel_active_us"]) and entry["GPU_kernel_active_us"] > 0
            and entry["GPU_kernel_active_us"] == sum(kernel["duration_us"] for kernel in kernels))


def _gpu_activity_complete(case: Case, routes: dict[str, Any], receipt: dict[str, Any]) -> bool:
    expected = {case.name + "/writer_" + kind for kind in ("mx", "nv")} | {
        case.name + "/attention_" + name for name in routes}
    observed = receipt["components"]
    return (len(observed) == len(expected) and {entry.get("label") for entry in observed} == expected
            and all(entry.get("status") == "GPU_events_present" and len(entry.get("invocations", [])) == 2
                    and [call.get("invocation") for call in entry["invocations"]] == [1, 2]
                    and all(_valid_gpu_activity_invocation(call) for call in entry["invocations"])
                    and entry.get("GPU_kernel_active_us_per_operation") ==
                        sum(call["GPU_kernel_active_us"] for call in entry["invocations"]) / 2
                    for entry in observed))


def _run_case(case: Case, ordinal: int, output: Path, phase: str,
              trace_jobs: list[tuple[str, Callable[[], Any]]]) -> dict[str, Any]:
    output.mkdir()
    full_archive = phase == "pilot"
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda").manual_seed(SEED + ordinal)
    initial_rng_state = generator.get_state().tolist()
    pages_per_sequence = (case.context + PAGE_SIZE - 1) // PAGE_SIZE
    count = case.batch * pages_per_sequence
    table = torch.arange(count, 0, -1, device="cuda", dtype=torch.int32).reshape(case.batch, pages_per_sequence)
    slots = (table[:, :, None].to(torch.int64) * PAGE_SIZE + torch.arange(PAGE_SIZE, device="cuda"))
    slots = slots.reshape(case.batch, -1)[:, :case.context].flatten().contiguous()
    values = tuple(torch.randn((case.batch * case.context, KV_HEADS, DIMENSION), generator=generator,
                               device="cuda", dtype=torch.bfloat16) for _ in range(2))
    query = torch.randn((case.batch * case.query_length, Q_HEADS, DIMENSION), generator=generator,
                        device="cuda", dtype=torch.bfloat16)
    sample_rows = sorted(row for row in {0, 1, 127, 128, 129,
                  *[(batch + 1) * case.context - 1 for batch in range(case.batch)]}
                         if row < case.batch * case.context)
    _write_json(output / "rng-and-samples.json", {"seed": SEED + ordinal,
                "initial_generator_state_uint8": initial_rng_state,
                "post_initialization_generator_state_uint8": generator.get_state().tolist(),
                "KV_sample_rows": sample_rows, "query_rows_per_request": _reference_rows(case),
                "main_large_tensor_bodies_retained": False,
                "retention": "pilot full small tensors; main full byte SHA, RNG state, fixed lossless samples"})
    _save_tensor(output / "input-key", values[0], full=full_archive, sample=values[0][sample_rows])
    _save_tensor(output / "input-value", values[1], full=full_archive, sample=values[1][sample_rows])
    _save_tensor(output / "input-query", query, full=full_archive, sample=_sample_output(query, case))
    _save_tensor(output / "input-page-table", table)
    _save_tensor(output / "input-slots", slots, full=full_archive, sample=slots[sample_rows])
    caches = {kind: _Cache(kind, count + 1) for kind in ("mx", "nv")}
    writers = {}
    for kind, cache in caches.items():
        cache._write(values, slots)
        writers[kind] = _writer_sample(case, cache, values, slots)
    matched, match_receipt = _same_decoded_nv(caches["mx"], slots)
    original_reference = _reference(case, query, table, None, values)
    references = {kind: _reference(case, query, table, cache) for kind, cache in caches.items()}
    _save_tensor(output / "reference-original-BF16", original_reference)
    for kind, reference in references.items():
        _save_tensor(output / ("reference-decoded-" + kind), reference)
    quantization = {kind: _errors(value, original_reference, REFERENCE_ATOL, REFERENCE_RTOL)
                    for kind, value in references.items()}
    for metric in quantization.values():
        metric.pop("pass")
        metric["interpretation"] = "descriptive operator quantization loss; not library conformance or model quality gate"
    current = values if case.query_length > 1 else tuple(
        value.reshape(case.batch, case.context, KV_HEADS, DIMENSION)[:, -1].contiguous() for value in values)
    current_slots = slots if case.query_length > 1 else slots.reshape(case.batch, case.context)[:, -1].contiguous()
    memory_before_release = _memory()
    del values, original_reference
    gc.collect()
    torch.cuda.empty_cache()
    memory_after_release = _memory()
    _write_json(output / "inputs.json", {"seed": SEED + ordinal, "table_sha256": _tensor_sha(table),
                "query_sha256": _tensor_sha(query),
                "compressed_cache_sha256": {kind: [_tensor_sha(owner) for owner in cache.owners] for kind, cache in caches.items()},
                "matched_cache_sha256": [_tensor_sha(owner) for owner in matched.owners],
                "same_decoded_control": match_receipt, "quantization_loss": quantization,
                "memory_before_reference_release": memory_before_release,
                "memory_after_reference_release": memory_after_release,
                "BF16_current_activation_bytes": sum(v.numel() * v.element_size() for v in current),
                "BF16_history_workspace_bytes_during_timing": 0,
                "resident_bytes": {kind: cache._bytes() for kind, cache in caches.items()},
                "matched_control_extra_resident_bytes": matched._bytes()})
    configurations = [("mx_prefill_wrapper", caches["mx"], False, False),
                      ("nv_prefill_wrapper", caches["nv"], False, False)]
    if case.query_length == 1:
        configurations.append(("nv_native_decode", caches["nv"], True, False))
    configurations.append(("matched_nv_prefill_wrapper", matched, False, True))
    if case.query_length == 1:
        configurations.append(("matched_nv_native_decode", matched, True, True))
    results, samples, routes, primary_checks = {}, {}, {}, {}
    for name, cache, native, control in configurations:
        route = _Route(name, cache, case, table, native)
        routes[name] = route
        expected = references["mx" if control else cache.kind]
        samples[name] = route._attention(query).clone()
        raw = _save_tensor(output / (name + "-raw-output"), samples[name], full=full_archive,
                           sample=_sample_output(samples[name], case), statistics_required=True)
        if not raw["full_tensor_statistics"]["finite"]:
            raise ValueError("Full eager output is nonfinite")
        check = _errors(_sample_output(samples[name], case), expected, REFERENCE_ATOL, REFERENCE_RTOL)
        primary_checks[name] = check
        _write_json(output / (name + "-preflight.json"), {
            "independent_decoded_reference": check,
            "plan_info_eager": list(route.eager._plan_info),
            "plan_info_graph": list(route.graph_wrapper._plan_info) if route.native_decode else None,
            "graph_status": "PENDING_NATIVE_QUALIFICATION" if route.native_decode else "BLOCKED",
            "graph_blocked_reason": None if route.native_decode else BLOCKED_GRAPH_REASON,
            "resolved_backend": getattr(route.eager, "_backend", "private MX FA2"),
            "split_policy": "native decode default planner" if route.native_decode else "uint8 disable_split_kv=True",
            "mask_policy": "NON_CAUSAL final single query" if route.native_decode else "CAUSAL",
        })
        if not check["pass"]:
            _write_json(output / (name + "-preflight-failed.json"), check)
            raise ValueError("Reference gate failed before timing")
    comparisons = {}
    pairs = [("mx_prefill_wrapper", "matched_nv_prefill_wrapper")]
    if case.query_length == 1:
        pairs.extend([("nv_prefill_wrapper", "nv_native_decode"),
                      ("matched_nv_prefill_wrapper", "matched_nv_native_decode")])
    for left, right in pairs:
        check = _errors(samples[left], samples[right], ROUTE_ATOL, ROUTE_RTOL)
        check["required_for_component_admission"] = left == "mx_prefill_wrapper"
        check["interpretation"] = ("same fused implementation and same decoded KV" if left == "mx_prefill_wrapper"
                                   else "distinct prefill/native-decode routes; strict equivalence diagnostic")
        comparisons[left + "__vs__" + right] = check
    _write_json(output / "route-conformance-before-timing.json", comparisons)
    if any(not check["pass"] for check in comparisons.values()):
        _write_json(output / "route-conformance-failed.json", comparisons)
    route_gates = _route_gates(case, comparisons)
    _write_json(output / "route-comparison-gates.json", route_gates)
    if not route_gates["same_fused_MX_NV_control_passed"]:
        raise ValueError("Same-fused MX/NV matched-cache control failed before timing")
    del samples
    for kind, cache in caches.items():
        writers[kind]["graph_correctness"] = _qualify_writer_graph(cache, current, current_slots, output, full_archive)
        writers[kind]["timing"] = _event_time(case.name + "/writer/" + kind,
                                              lambda cache=cache: cache._write(current, current_slots))
        writer_graph = _capture(lambda cache=cache: cache._write(current, current_slots), WRITER_GRAPH_BURST)
        writers[kind]["graph_timing"] = _event_time(case.name + "/writer_graph/" + kind,
                                                   writer_graph.replay, WRITER_GRAPH_BURST)
        writers[kind]["eager_NV_scale_validation_D2H"] = kind == "nv"
        writers[kind]["interpretation"] = "eager API stream time includes wrapper work; graph replay bypasses eager scale validation through the normal capture contract"
        del writer_graph
        trace_jobs.append((case.name + "/writer_" + kind,
                           lambda cache=cache, current=current, slots=current_slots: cache._write(current, slots)))
    _write_json(output / "writers.json", writers)
    for name, cache, native, control in configurations:
        expected = references["mx" if control else cache.kind]
        results[name] = _profile_route(routes[name], query, current, current_slots, expected, output, control, full_archive)
        trace_jobs.append((case.name + "/attention_" + name,
                           lambda route=routes[name], query=query: route._attention(query)))
    activity = _trace_components(trace_jobs, output.parent / "component-cupti-trace" / case.name)
    observed = _gpu_activity_complete(case, results, activity)
    result = {"case": asdict(case), "routes": results, "writers": writers,
              "gpu_activity": activity, "all_component_gpu_activity_observed": observed,
              "route_conformance": comparisons,
              "route_comparison_gates": route_gates,
              "primary_independent_reference_passed": all(check["pass"] for check in primary_checks.values()),
              "primary_independent_references": primary_checks, "quantization_loss": quantization}
    _write_json(output / "case.json", result)
    if not observed:
        raise ValueError("CUPTI did not produce actual GPU kernels for every required supported component")
    return result


def _load_and_check_sources() -> dict[str, Any]:
    global torch, flashinfer, mx_writer, mx_backend, mx_reference, nv_layout
    torch = importlib.import_module("torch")
    flashinfer = importlib.import_module("flashinfer")
    importlib.import_module("flashinfer.page")
    importlib.import_module("flashinfer.decode")
    if flashinfer.__version__ != "0.6.18.post1":
        raise ValueError("Pinned FlashInfer version required; no automatic install/upgrade")
    modules = {name: importlib.import_module("xllm.python.attention." + name) for name in MX_HASHES}
    mx_writer, mx_backend = modules["mxfp4_writer"], modules["mxfp4_flashinfer"]
    mx_reference, nv_layout = modules["mxfp4_reference"], modules["native_kv_layout"]
    paths = {"flashinfer/" + path: (Path(flashinfer.__file__).parent / path, sha) for path, sha in FI_HASHES.items()}
    paths.update({name: (Path(modules[name].__file__), sha) for name, sha in MX_HASHES.items()})
    actual = {}
    for name, (path, expected) in paths.items():
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        actual[name] = {"path": str(path), "sha256": sha, "expected": expected}
        if sha != expected:
            raise ValueError("Source identity mismatch: " + name)
    return {"files": actual, "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "flashinfer_version": flashinfer.__version__, "flashinfer_commit": FI_COMMIT}


def _profiler_inventory() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for tool in ("nsys", "ncu"):
        executable = shutil.which(tool)
        result[tool] = {"path": executable, "version": None}
        if executable is not None:
            try:
                completed = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=10)
                result[tool].update({"returncode": completed.returncode,
                                     "version": (completed.stdout + completed.stderr)[-4000:]})
            except subprocess.TimeoutExpired:
                result[tool]["status"] = "version_probe_timeout"
    result["fused_scale_unpack_internal_wall_share"] = "unresolved; trace counters are not additive timing"
    return result


def _trace_size_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _trace_components(jobs: list[tuple[str, Callable[[], Any]]], output: Path) -> dict[str, Any]:
    """Two separately observed calls per component; reject missing invocation evidence."""
    output.mkdir(parents=True)
    receipts = []
    budget = 16 * 2**20
    for label, operation in jobs:
        name = label.replace("/", "__")
        receipt: dict[str, Any] = {"label": label, "repetitions": 2, "invocations": [],
                                  "phase": "after_this_case_primary_timings", "eager_API_trace": True,
                                  "one_profiler_context_per_invocation": True}
        for invocation in (1, 2):
            raw = output / (name + ".call-" + str(invocation) + ".trace.json")
            call: dict[str, Any] = {"invocation": invocation, "actual_API_calls_requested": 1}
            try:
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                             torch.profiler.ProfilerActivity.CUDA], record_shapes=False,
                             profile_memory=False, with_stack=False) as profiler:
                    with torch.profiler.record_function(label + "/call_" + str(invocation)):
                        operation()
                    torch.cuda.synchronize()
                profiler.export_chrome_trace(str(raw))
                call.update({"raw_trace": raw.name, "raw_bytes": raw.stat().st_size,
                             "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest()})
                payload = json.loads(raw.read_text())
                kernels = [{"name": event.get("name"), "duration_us": event["dur"],
                            "timestamp_us": event.get("ts"), "args": event.get("args", {})}
                           for event in payload.get("traceEvents", [])
                           if event.get("cat") == "kernel" and event.get("ph") == "X" and "dur" in event]
                copies = [{"name": event.get("name"), "duration_us": event["dur"],
                           "category": event.get("cat"), "args": event.get("args", {})}
                          for event in payload.get("traceEvents", [])
                          if "memcpy" in str(event.get("cat", "")).lower()
                          and event.get("ph") == "X" and "dur" in event]
                if not kernels or any(type(kernel["duration_us"]) not in (int, float)
                                      or not math.isfinite(kernel["duration_us"]) or kernel["duration_us"] <= 0
                                      for kernel in kernels):
                    raise ValueError("CUPTI invocation lacks finite positive GPU kernel durations")
                call.update({"status": "GPU_events_present", "kernels": kernels, "copies": copies,
                             "GPU_kernel_active_us": sum(kernel["duration_us"] for kernel in kernels)})
                if not _valid_gpu_activity_invocation(call):
                    raise ValueError("CUPTI invocation kernel-duration sum is invalid")
            except Exception as error:
                # Retain raw/error evidence; no second-call inference, retry or profiler install.
                call = {key: value for key, value in call.items()
                        if key not in ("kernels", "copies", "GPU_kernel_active_us")}
                call.update({"status": "profiler_failed", "error_type": type(error).__name__,
                             "error": str(error), "traceback": traceback.format_exc()})
            _write_json(output / (name + ".call-" + str(invocation) + ".json"), call)
            receipt["invocations"].append(call)
            if call["status"] != "GPU_events_present" or _trace_size_bytes(output.parent) > budget:
                break
        calls = receipt["invocations"]
        valid = len(calls) == 2 and all(_valid_gpu_activity_invocation(call) for call in calls)
        receipt.update({"status": "GPU_events_present" if valid else "incomplete_or_invalid_GPU_invocations",
                        "kernels": [dict(kernel, invocation=call["invocation"]) for call in calls for kernel in call.get("kernels", [])],
                        "copies": [dict(copy, invocation=call["invocation"]) for call in calls for copy in call.get("copies", [])],
                        "GPU_kernel_active_us_per_operation":
                            sum(call["GPU_kernel_active_us"] for call in calls) / 2 if valid else None,
                        "duration_aggregation": "mean of two separately verified sums of GPU kernel durations; not interval union",
                        "internal_scale_unpack_percent": None,
                        "interpretation": "GPU active event durations; not end-to-end wall-time shares"})
        _write_json(output / (name + ".json"), receipt)
        receipts.append(receipt)
        if _trace_size_bytes(output.parent) > budget:
            receipts.append({"status": "trace_budget_exceeded_stop", "budget_bytes": budget,
                             "existing_raw_preserved": True})
            break
        if not valid:
            break
    result = {"case": output.name, "optional": False, "repetitions": 2,
              "one_profiler_context_per_invocation": True,
              "components": receipts, "budget_bytes_all_cases": budget,
              "bytes_before_this_summary": _trace_size_bytes(output.parent),
              "primary_timing_results_unmodified": True,
              "only_current_case_caches_retained_for_this_trace": True,
              "callbacks_released_before_next_case": True}
    _write_json(output / "summary.json", result)
    if _trace_size_bytes(output.parent) > budget:
        raise ValueError("CUPTI trace budget exceeded after writing summary; existing raw preserved")
    return result


def run(output: Path, phase: str = "main") -> dict[str, Any]:
    output = Path(output)
    output.mkdir()
    _write_json(output / "plan.json", describe(phase))
    with (output / "probe-driver.py").open("xb") as handle:
        handle.write(Path(__file__).read_bytes())
    _write_json(output / "profiler-inventory.json", _profiler_inventory())
    identity = _load_and_check_sources()
    if not str(torch.__version__).startswith("2.10.") or torch.version.cuda != "13.0":
        raise ValueError("This probe requires the existing Torch2.10/CUDA13.0 image")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (9, 0):
        raise ValueError("Require exactly one externally admitted SM90 GPU")
    identity["device_name"] = torch.cuda.get_device_name(0)
    identity["device_properties"] = str(torch.cuda.get_device_properties(0))
    identity["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    _write_json(output / "identity.json", identity)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = []
    with torch.inference_mode():
        for ordinal, case in enumerate(PILOT_CASES if phase == "pilot" else CASES):
            trace_jobs: list[tuple[str, Callable[[], Any]]] = []
            torch.cuda.nvtx.range_push(case.name)
            try:
                results.append(_run_case(case, ordinal, output / case.name, phase, trace_jobs))
            finally:
                torch.cuda.nvtx.range_pop()
            trace_jobs.clear()
            gc.collect()
            torch.cuda.empty_cache()
    gates = _acceptance_gates(results)
    result = {"schema": "q4_supported_component_result_v3", "completed": True, "pass": True, **gates,
              "source_checked": True, "phase": phase, "cases": results, "sglang": describe(phase)["sglang"],
              "service_or_model_quality_claim": False, "full_vllm_engine_tested": False,
              "independent_fused_scale_unpack_percentage": None}
    _write_json(output / "result.json", result)
    artifact_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file()
                         and "component-cupti-trace" not in path.parts)
    if artifact_bytes > 64 * 2**20:
        raise ValueError("Diagnostic artifacts exceeded the predeclared 64MiB cap")
    _write_json(output / "acceptance.json", {"schema": "q4_supported_component_acceptance_v3",
                "phase": phase, "completed": True, "pass": True, "source_checked": True, **gates,
                "pilot_timings_excluded_from_main_tables": phase == "pilot",
                "artifact_bytes_before_acceptance": artifact_bytes,
                "model_loaded": False, "service_benchmark": False,
                "full_vllm_engine_tested": False, "sglang_target": "unsupported_NA"})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--phase", choices=("pilot", "main"), default="main")
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(describe(args.phase), indent=2))
    elif args.output is None:
        parser.error("--output is required unless --describe is used")
    else:
        output_preexisted = args.output.exists()
        try:
            run(args.output, args.phase)
        except Exception as error:
            # Process boundary: retain the error and re-raise; no fallback or retry.
            if not output_preexisted and args.output.is_dir() and not (args.output / "failure.json").exists():
                _write_json(args.output / "failure.json", {"type": type(error).__name__,
                            "message": str(error), "traceback": traceback.format_exc(),
                            "phase": args.phase, "pass": False})
            raise
