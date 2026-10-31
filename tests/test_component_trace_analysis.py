# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Synthetic CPU fixtures exercise attribution, never count as GPU evidence."""

import unittest

from scripts.analyze_component_trace import analyze_trace


def _event(name, start, duration, cat, correlation=None):
    return {'ph': 'X', 'name': name, 'ts': start, 'dur': duration, 'pid': 10, 'tid': 20,
            'cat': cat, 'args': {'correlation': correlation, 'device': 0}}


def _fixture():
    trace = {'deviceProperties': [{'CPU_fixture_only': True}], 'traceEvents': [
        _event('xllm.forward.prefill', 0, 100, 'user_annotation'),
        _event('xllm.component.fp8.kv_write', 0, 30, 'user_annotation'),
        _event('cudaLaunchKernel', 2, 1, 'cuda_runtime', 7),
        _event('cudaLaunchKernel', 4, 1, 'cuda_runtime', 8),
        _event('CPU_fixture_kernel_A', 101, 20, 'kernel', 7),
        _event('CPU_fixture_kernel_B', 111, 20, 'kernel', 8),
        _event('CPU_fixture_kernel_unmatched', 140, 10, 'kernel', 999)]}
    meta = {'steps': [{'prefill': True, 'chunked_prefill': False, 'sequences': 2, 'input_tokens': 4096}]}
    return trace, meta


class ComponentTraceAnalysisTests(unittest.TestCase):
    def test_cpu_time_kernel_sum_and_overlapping_union_stay_distinct(self):
        trace, meta = _fixture(); value = analyze_trace(trace, meta)
        writer = value['components']['xllm.component.fp8.kv_write']
        self.assertEqual(writer['cpu_scope_sum_ms'], .03)
        self.assertEqual(writer['cuda_kernel_sum_ms'], .04)
        self.assertEqual(writer['cuda_kernel_busy_union_ms'], .03)
        self.assertEqual(value['cuda_all_activity_busy_union_ms'], .04)
        self.assertEqual(value['components']['unassigned']['cuda_kernel_events'], 1)
        self.assertEqual(value['forward_batches'][0]['sequences'], 2)
        self.assertEqual(value['outside_recorded_forward_cuda_kernel_sum_ms'], .01)

    def test_graph_replay_has_no_invented_writer_or_attention_breakdown(self):
        trace, meta = _fixture()
        trace['traceEvents'][1]['name'] = 'xllm.component.graph_replay'
        value = analyze_trace(trace, meta)
        self.assertEqual(value['components']['xllm.component.graph_replay']['cuda_kernel_events'], 2)
        self.assertNotIn('xllm.component.fp8.kv_write', value['components'])

    def test_wrong_device_missing_kernel_and_batch_phase_drift_are_refused(self):
        for corruption in ('device', 'kernel', 'phase'):
            trace, meta = _fixture()
            if corruption == 'device': trace['traceEvents'][-1]['args']['device'] = 1
            elif corruption == 'kernel': trace['traceEvents'] = [e for e in trace['traceEvents'] if e['cat'] != 'kernel']
            else: meta['steps'][0]['prefill'] = False
            with self.subTest(corruption=corruption), self.assertRaises(ValueError): analyze_trace(trace, meta)

    def test_ambiguous_launch_scope_is_retained_unassigned(self):
        trace, meta = _fixture()
        trace['traceEvents'].append(_event('cudaLaunchKernel', 80, 1, 'cuda_runtime', 7))
        value = analyze_trace(trace, meta)
        self.assertEqual(value['components']['unassigned']['cuda_kernel_events'], 2)


if __name__ == '__main__':
    unittest.main()
