# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""CPU-only clocks distinguish visible content from usage/DONE completion."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_stream_completion_gap import analyze


class StreamCompletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(); self.base = Path(self.temp.name)
        self.audit = self.base/'audit.json'; self.mirror = self.base/'mirror'; self.source = self.base/'source'
        names = ('metrics/types/request_latency_metric.py', 'records/record_processor_service.py',
                 'common/models/record_models.py', 'endpoints/openai_chat.py')
        for name in names:
            p = self.source/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# CPU fixture\n')
        row = {'session_id': 's', 'logical_phase': 'measurement', 'success': True, 'output_tokens': 3,
               'request_start_ns': 1000000000, 'request_end_ns': 1008000000,
               'client_start_lag_ms': 1.0, 'e2e_ms': 2.0, 'offered_e2e_ms': 9.0,
               'ttft_ms': 1.0, 'offered_ttft_ms': 2.0, 'itl_ms': .5}
        self.audit.write_text(json.dumps({'classification': 'confirmed_trace_measurement_integrity', 'run_id': 'fixture',
            'sealed_arms': {'bf16': {'metrics': {'per_request': [row],
                'phases': {'measurement': {'elapsed_including_drain_seconds': 1.0}}}}}}))
        responses = []
        for timestamp, value in [(1001000, {'choices': [{'delta': {'content': 'a'}}]}),
                                  (2001000, {'choices': [{'delta': {'content': 'b'}}]}),
                                  (5001000, {'choices': [], 'usage': {'completion_tokens': 3}}),
                                  (8001000, '[DONE]')]:
            responses.append({'perf_ns': timestamp, 'packets': [{'name': 'data',
                'value': value if isinstance(value, str) else json.dumps(value)}]})
        self.raw = {'metadata': {'conversation_id': 's', 'request_start_ns': 1000000000, 'request_end_ns': 1008000000},
                    'start_perf_ns': 1000, 'responses': responses}
        self.raw_path = self.mirror/'support/agent_work/runs/fixture/output/bf16/artifacts/profile_export_raw.jsonl'
        self.raw_path.parent.mkdir(parents=True); self.raw_path.write_text(json.dumps(self.raw)+'\n')

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_content_free_terminal_tail_is_kept_in_completion_proxy(self) -> None:
        result = analyze(self.audit, self.mirror, self.source)['arms']['bf16']
        self.assertEqual(result['terminal_minus_content_ms']['p50'], 6.0)
        self.assertEqual(result['final_SSE_inclusive_per_server_output_token_ms']['p50'], 3.5)

    def test_timestamp_mismatch_cannot_be_explained_as_a_terminal_tail(self) -> None:
        self.raw['metadata']['request_end_ns'] += 1
        self.raw_path.write_text(json.dumps(self.raw)+'\n')
        with self.assertRaises(ValueError): analyze(self.audit, self.mirror, self.source)


if __name__ == '__main__':
    unittest.main()
