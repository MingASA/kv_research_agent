# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Replay validation rejects corrupt evidence while retaining genuine overload."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.controller.durable import digest
from scripts.controller.trace_serving import audit_trace, validate_inputs


class TraceAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.artifacts = self.root / 'artifacts'
        (self.artifacts / 'logs').mkdir(parents=True)
        self.origin = 1_000_000_000_000
        (self.artifacts / 'logs/aiperf.log').write_text(
            'PhaseRecordsStats(phase=CreditPhase.PROFILING, phase_index=0, start_ns='+str(self.origin)+', total_expected_requests=3)\n')
        self.source = self.root / 'payload.jsonl'; self.manifest_file = self.root / 'manifest.json'
        self.rows, self.metrics, self.raws, records = [], [], [], []
        for i, (timestamp, count, status) in enumerate(((0, 2, 200), (60000, 3, 200), (61000, 4, 429))):
            identity = 'fixture-'+str(i)
            payload = {'messages': [{'role': 'user', 'content': 'CPU fixture'}], 'max_tokens': count,
                       'ignore_eos': True, 'stream': True, 'stream_options': {'include_usage': True}}
            self.rows.append({'session_id': identity, 'timestamp': timestamp, 'payload': payload})
            records.append({'replay_id': identity, 'source_timestamp_ms': timestamp, 'source_output_tokens': count,
                            'rendered_chat_tokens_cpu': 20, 'payload_sha256': digest(payload),
                            'logical_phase': 'warm_cache' if i == 0 else 'measurement'})
            start = self.origin+timestamp*1_000_000+2_000_000_000
            meta = {'conversation_id': identity, 'x_request_id': 'r'+str(i), 'benchmark_phase': 'profiling',
                    'turn_index': 0, 'was_cancelled': False, 'context_overflow_skip': False,
                    'credit_issued_ns': start-1000000, 'request_start_ns': start, 'request_end_ns': start+100_000_000}
            values = {'output_token_count': count, 'input_sequence_length': 20,
                      'time_to_first_token': 50, 'inter_token_latency': 50, 'request_latency': 100}
            self.metrics.append({'metadata': meta, 'error': None if status == 200 else {'type': 'HTTP429'},
                                 'metrics': {k: {'value': v} for k,v in values.items()}})
            packet = {'choices': [{'finish_reason': 'length'}],
                      'usage': {'prompt_tokens': 20, 'completion_tokens': count, 'total_tokens': 20+count}}
            self.raws.append({'metadata': meta, 'payload': payload, 'status': status, 'error': None if status == 200 else 'HTTP429',
                              'responses': [{'packets': [{'name': 'data', 'value': json.dumps(packet)},
                                                         {'name': 'data', 'value': '[DONE]'}]}] if status == 200 else []})
        self.manifest = {'selected_requests': 3, 'records': records, 'arrival_scaling': 1,
                         'first_retained_timestamp_ms': 0, 'window_begin_ms': 0, 'window_end_ms': 300000}
        (self.artifacts / 'profile_export_aiperf.json').write_text(json.dumps({
            'aiperf_version': '0.13.0', 'is_complete': True, 'was_cancelled': False, 'error_summary': {'HTTP429': 1}}))
        self._freeze()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _freeze(self) -> None:
        self.source.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        self.manifest_file.write_text(json.dumps(self.manifest))

    def _audit(self) -> dict:
        (self.artifacts / 'profile_export.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.metrics))
        (self.artifacts / 'profile_export_raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.raws))
        return audit_trace(self.artifacts, self.source, self.manifest_file)

    def test_rejections_retained_and_offered_latency_includes_client_delay(self) -> None:
        result = self._audit()
        self.assertEqual(result['http_status_counts'], {'200': 2, '429': 1})
        self.assertEqual(result['phases']['measurement']['attempts'], 2)
        self.assertEqual(result['phases']['measurement']['successes'], 1)
        self.assertEqual(result['phases']['measurement']['errors'], 1)
        self.assertEqual(result['phases']['measurement']['offered_ttft_ms']['p50'], 2050)
        self.assertEqual(result['phases']['measurement']['actual_output_tokens'], 3)

    def test_missing_raw_attempt_cannot_be_silently_discarded(self) -> None:
        self.raws.pop()
        with self.assertRaisesRegex(ValueError, 'incomplete'): self._audit()

    def test_length_or_wire_mismatch_prevents_valid_replay(self) -> None:
        self.raws[1]['payload'] = {**self.raws[1]['payload'], 'ignore_eos': False}
        with self.assertRaisesRegex(ValueError, 'wire payload'): self._audit()

    def test_arrival_scaling_must_match_unchanged_source_clock(self) -> None:
        self.rows[1]['timestamp'] += 1; self._freeze()
        with self.assertRaisesRegex(ValueError, 'timestamps changed'):
            validate_inputs(self.source, self.manifest_file)

    def test_missing_exact_phase_clock_prevents_invented_arrival_lag(self) -> None:
        (self.artifacts / 'logs/aiperf.log').write_text('')
        with self.assertRaisesRegex(ValueError, 'exact AIPerf'): self._audit()


if __name__ == '__main__':
    unittest.main()
