# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""CPU fixtures for real native-tool framing, full attempts and admission gates."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.controller.durable import canonical_bytes, digest
from scripts.controller.native_tools import audit_native_tools, parse_sse_packets, validate_native_inputs, parse_buffered_chat_response
from scripts.controller.policy import BF16_XQA_OPERATORS, NATIVE_TOOL_SOURCES, validate_request
from scripts.controller.production import run_service
from test_execution import fixture


def _packets():
    data = [{'choices': [{'index': 0, 'delta': {'content': 'Checking.', 'tool_calls': [
        {'index': 0, 'id': 'CPU_fixture', 'type': 'function',
         'function': {'name': 'lookup', 'arguments': '{"city":'}}]}, 'finish_reason': None}]},
        {'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0,
            'function': {'arguments': '"Paris"}'}}]}, 'finish_reason': 'tool_calls'}]},
        {'choices': [], 'usage': {'prompt_tokens': 2, 'completion_tokens': 4, 'total_tokens': 6}}]
    return [{'name': 'data', 'value': json.dumps(item)} for item in data] + [{'name': 'data', 'value': '[DONE]'}]


def _artifacts(root):
    source, manifest, artifacts = root/'inputs.json', root/'manifest.json', root/'artifacts'
    artifacts.mkdir()
    data, records, metric_rows, raw_rows = [], [], [], []
    for i in range(274):
        session = 'CPU_only_' + str(i); phase = 'warmup' if i < 16 else 'profiling'
        payload = {'model': 'CPU_fixture', 'messages': [{'role': 'user', 'content': str(i)}],
            'tools': [{'type': 'function', 'function': {'name': 'lookup'}}], 'tool_choice': 'auto',
            'ignore_eos': False, 'temperature': 0, 'top_p': 1, 'max_tokens': 1024,
            'stream': True, 'stream_options': {'include_usage': True}}
        data.append({'session_id': session, 'payloads': [payload]})
        records.append({'session_id': session, 'phase': 'warmup' if i < 16 else 'measurement',
                        'payload_sha256': digest(payload)})
        meta = {'x_request_id': 'CPU_request_' + str(i), 'conversation_id': session,
                'benchmark_phase': phase, 'turn_index': 0, 'was_cancelled': False, 'context_overflow_skip': False,
                'request_start_ns': 1000000000 + i * 100000000, 'request_end_ns': 1050000000 + i * 100000000}
        metrics = {k: {'value': v} for k,v in {'input_sequence_length': 2, 'output_token_count': 4,
            'usage_total_tokens': 6, 'time_to_first_token': 5, 'request_latency': 20}.items()}
        metric_rows.append({'metadata': meta, 'metrics': metrics, 'error': None})
        raw_rows.append({'metadata': meta, 'payload': payload, 'status': 200, 'error': None,
                         'responses': [{'packets': _packets()}]})
    source.write_bytes(canonical_bytes({'data': data}))
    manifest.write_text(json.dumps({'records': records, 'dataset_sha256': digest({'data': data})}))
    for name, rows in [('profile_export.jsonl', metric_rows), ('profile_export_raw.jsonl', raw_rows)]:
        (artifacts/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (artifacts/'profile_export_aiperf.json').write_text(json.dumps({'is_complete': True, 'was_cancelled': False,
        'aiperf_version': '0.13.0', 'start_time': '2026-10-27T00:00:00', 'end_time': '2026-10-27T00:01:00',
        'error_summary': None}))
    return source, manifest, artifacts, metric_rows, raw_rows


class NativeToolTests(unittest.TestCase):
    def test_buffered_full_tool_JSON_has_complete_clock_without_streaming_TTFT(self):
        with tempfile.TemporaryDirectory() as d:
            source, manifest, artifacts, metric_rows, raws = _artifacts(Path(d))
            data = json.loads(source.read_text()); m = json.loads(manifest.read_text())
            m['response_mode'] = 'buffered'
            response = {'object': 'chat.completion', 'choices': [{'index': 0, 'finish_reason': 'tool_calls',
                'message': {'content': None, 'tool_calls': [{'id': 'CPU_buffered', 'type': 'function',
                    'function': {'name': 'lookup', 'arguments': '{"city":"Paris"}'}}]}}],
                'usage': {'prompt_tokens': 2, 'completion_tokens': 4, 'total_tokens': 6}}
            for session, record, raw in zip(data['data'], m['records'], raws):
                payload = session['payloads'][0]; payload['stream'] = False; payload.pop('stream_options')
                record['payload_sha256'] = digest(payload); raw['payload'] = copy.deepcopy(payload)
                raw['responses'] = [{'text': json.dumps(response), 'content_type': 'application/json'}]
            m['dataset_sha256'] = digest(data); source.write_bytes(canonical_bytes(data)); manifest.write_text(json.dumps(m))
            (artifacts/'profile_export_raw.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in raws))
            result = audit_native_tools(artifacts, source, manifest)
            self.assertEqual(result['measurement']['successes'], 258)
            self.assertEqual(result['response_mode'], 'buffered')
            self.assertIsNone(result['requests'][16]['ttft_ms'])
            self.assertEqual(result['requests'][16]['complete_request_ms'], 50)
            bad = copy.deepcopy(response); bad['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = 'notJSON'
            with self.assertRaises(ValueError): parse_buffered_chat_response(bad)

    def test_split_arguments_mixed_content_and_tool_only_keep_real_native_call(self):
        for content in (True, False):
            packets = _packets()
            if not content:
                item = json.loads(packets[0]['value']); item['choices'][0]['delta'].pop('content')
                packets[0]['value'] = json.dumps(item)
            result = parse_sse_packets(packets)
            self.assertEqual(json.loads(result['tool_calls'][0]['function']['arguments']), {'city': 'Paris'})
            self.assertEqual(result['finish_reason'], 'tool_calls')
            self.assertEqual(result['usage']['completion_tokens'], 4)

    def test_changed_identity_missing_done_usage_and_ambiguous_json_rejected(self):
        for corruption in ('id', 'done', 'usage', 'duplicates'):
            packets = _packets()
            if corruption == 'done': packets.pop()
            elif corruption == 'usage': packets.pop(2)
            else:
                item = json.loads(packets[1]['value']); piece = item['choices'][0]['delta']['tool_calls'][0]
                if corruption == 'id': piece['id'] = 'changed_CPU_id'
                else: piece['function']['arguments'] = '"Paris","city":"London"}'
                packets[1]['value'] = json.dumps(item)
            with self.subTest(corruption=corruption), self.assertRaises(ValueError): parse_sse_packets(packets)

    def test_all_full_original_attempts_are_retained_including_HTTP200_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            source, manifest, artifacts, metrics, raws = _artifacts(Path(d))
            baseline = audit_native_tools(artifacts, source, manifest)
            self.assertEqual(baseline['measurement']['successes'], 258)
            metrics[16]['error'] = {'message': 'CPU_fixture_admission_rejection'}
            raws[16]['responses'] = [{'packets': [{'name': 'CPU concurrent limit', 'value': None}]}]
            for name, rows in [('profile_export.jsonl', metrics), ('profile_export_raw.jsonl', raws)]:
                (artifacts/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
            value = audit_native_tools(artifacts, source, manifest)
            self.assertEqual(value['measurement'], {'attempts': 258, 'failures': 1, 'successes': 257})
            self.assertEqual(len(value['requests']), 274)
            self.assertEqual(value['requests'][16]['http_status'], 200)
            self.assertFalse(value['requests'][16]['service_success'])
            self.assertFalse(value['task_quality_scored'])

    def test_missing_wire_attempt_and_payload_tool_drift_are_integrity_errors(self):
        for corruption in ('missing', 'tool_choice'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as d:
                source, manifest, artifacts, metrics, raws = _artifacts(Path(d))
                if corruption == 'missing': raws.pop()
                else: raws[16]['payload'] = {**raws[16]['payload'], 'tool_choice': 'required'}
                (artifacts/'profile_export_raw.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in raws))
                with self.assertRaises(ValueError): audit_native_tools(artifacts, source, manifest)

    def test_native_operator_failure_prevents_loading_model(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source, manifest, _, _, _ = _artifacts(root)
            request = {'workload': {'kind': 'native_tools_xqa_bf16_reference', 'requests': 258,
                'warmup': 16, 'input_manifest': str(manifest)}, 'operators': BF16_XQA_OPERATORS,
                'python_source_repo': str(root), 'dataset_path': str(source)}
            help_result = subprocess.CompletedProcess([], 0, '--export-level --custom-dataset-type --use-server-token-count')
            with patch('scripts.controller.production.subprocess.run', side_effect=[help_result, subprocess.CalledProcessError(1, ['CPU_operator'])]):
                with patch('scripts.controller.production.subprocess.Popen') as model:
                    with self.assertRaises(subprocess.CalledProcessError): run_service({'request': request}, root, root, 'bf16', {})
                    model.assert_not_called()

    def test_native_scope_rejects_forced_EOS_changed_cohort_parser_or_legacy_operators(self):
        approval, request, _ = fixture()
        request.update(schema='kv-production-run-v1', prefix_cache=True, concurrency=128,
            max_wall_seconds=3600, max_output_gib=3, python_source_repo=request['remote_output']+'/source_repo',
            python_source_commit='c'*40, backend_source_sha256={name: 'a'*64 for name in NATIVE_TOOL_SOURCES},
            operators=copy.deepcopy(BF16_XQA_OPERATORS), service={'max_seqs_per_batch': 128,
                'max_tokens_per_batch': 10240, 'max_concurrent_requests': 200, 'memory_utilization': .8},
            workload={'kind': 'native_tools_xqa_bf16_reference', 'requests': 258, 'warmup': 16, 'ignore_eos': False,
                'tool_call_parser': 'qwen25', 'source_commit': '6ea57973c7a6097fd7c5915698c54c17c5b1b6c8',
                'content_scope': 'original_BFCL_live_simple_messages_native_tools', 'source_content_sha256': 'b'*64,
                'arrival_mode': 'closed_loop_concurrency', 'input_manifest': request['remote_output']+'/input/production_manifest.json'})
        validate_request(request, approval)
        for key, value in [('requests', 256), ('ignore_eos', True), ('tool_call_parser', 'unknown')]:
            changed = copy.deepcopy(request); changed['workload'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): validate_request(changed, approval)
        request['operators']['bf16'].pop('decode')
        with self.assertRaises(ValueError): validate_request(request, approval)


if __name__ == '__main__':
    unittest.main()
