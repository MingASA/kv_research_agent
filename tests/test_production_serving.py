"""CPU fixtures for real-request audit and modern production scope; no GPU claims."""

import copy
import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess

from scripts.controller.policy import PRODUCTION_SOURCES, QUALITY_SOURCES, XQA_SOURCES, XQA_OPERATORS, BF16_XQA_OPERATORS, validate_request
from scripts.controller.production import audit_requests, verify_backend, candidate_quality_command, run_candidate_quality, run_service
from test_execution import fixture


class ProductionPolicyTests(unittest.TestCase):
    def request(self):
        approval, request, _ = fixture()
        request.update(schema="kv-production-run-v1", prefix_cache=True, concurrency=128,
                       python_source_repo=request["remote_output"] + "/source_repo", python_source_commit="c" * 40,
                       backend_source_sha256={name: "a" * 64 for name in PRODUCTION_SOURCES},
                       operators={"bf16": {"planner": "fa3", "q": "bf16"}, "fp8_e4m3": {"planner": "fa3", "q": "fp8_e4m3", "q_scale": 1.0}},
                       service={"max_seqs_per_batch": 128, "max_tokens_per_batch": 10240, "max_concurrent_requests": 200, "memory_utilization": .8},
                       workload={"kind": "sharegpt_natural", "ignore_eos": False, "requests": 512, "warmup": 16, "source_content_sha256": "b" * 64})
        return approval, request

    def test_explicit_modern_path_and_genuine_natural_requests_are_allowed(self):
        a, r = self.request(); validate_request(r, a)

    def test_declared_xqa_natural_serving_still_requires_natural_eos(self):
        a, r = self.request()
        r['workload']['kind']='sharegpt_natural_xqa'
        r['operators']=copy.deepcopy(XQA_OPERATORS)
        r['backend_source_sha256']={name:'a'*64 for name in XQA_SOURCES}
        validate_request(r,a)
        r['workload']['ignore_eos']=True
        with self.assertRaises(ValueError): validate_request(r,a)

    def test_bf16_xqa_reference_is_explicit_and_cannot_change_fp8_or_internal_precision(self):
        a, r = self.request()
        r['workload']['kind'] = 'sharegpt_natural_xqa_bf16_reference'
        r['operators'] = copy.deepcopy(BF16_XQA_OPERATORS)
        r['backend_source_sha256'] = {name: 'a'*64 for name in XQA_SOURCES}
        validate_request(r, a)
        for arm, key, wrong in [('bf16', 'decode_internal_q', 'fp8_e4m3'), ('bf16', 'planner', 'fa2'),
                                ('fp8_e4m3', 'planner', 'fa3'), ('bf16', 'enable_pdl', False)]:
            bad = copy.deepcopy(r); bad['operators'][arm][key] = wrong
            with self.subTest(arm=arm, key=key), self.assertRaises(ValueError): validate_request(bad, a)
        r['workload']['kind'] = 'sharegpt_natural_xqa'
        with self.assertRaises(ValueError): validate_request(r, a)

    def test_bf16_xqa_service_operator_failure_prevents_loading_the_model(self):
        _, request = self.request()
        request['operators'] = copy.deepcopy(BF16_XQA_OPERATORS)
        request['workload']['kind'] = 'sharegpt_natural_xqa_bf16_reference'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'input.json'; request['dataset_path'] = str(source)
            source.write_text(json.dumps({'data': [{'session_id': str(i), 'payloads': [
                {'messages': [{'role': 'user', 'content': 'fixture'}], 'max_tokens': 1024, 'ignore_eos': False}]
                } for i in range(528)]}))
            help_result = subprocess.CompletedProcess([], 0, '--export-level --custom-dataset-type --use-server-token-count')
            with patch('scripts.controller.production.subprocess.run', side_effect=[help_result, subprocess.CalledProcessError(1, ['operator'])]) as probe:
                with patch('scripts.controller.production.subprocess.Popen') as model:
                    with self.assertRaises(subprocess.CalledProcessError):
                        run_service({'request': request}, root, root, 'bf16', {})
                    model.assert_not_called()
                env = probe.call_args.kwargs['env']
                self.assertEqual(env['XLLM_FLASHINFER_NATIVE_PLANNER'], 'fa3')
                self.assertEqual(env['XLLM_FLASHINFER_NATIVE_DECODE'], 'xqa')
                self.assertNotIn('XLLM_FLASHINFER_FP8_QUERY', env)

    def test_actual_bf16_xqa_backend_marker_requires_declared_dtype_and_graph(self):
        backend = {'class': 'FlashInferBackend', 'attention_impl': 'flashinfer', 'planner_backend': 'fa3',
            'paged_prefill': True, 'graph_enabled': True, 'profile_scope': 'production_xqa',
            'quality_transform': False, 'kv_cache_dtype': 'auto', 'q_data_type': 'torch.bfloat16',
            'kv_data_type': 'torch.bfloat16', 'decode_impl': 'xqa', 'decode_internal_q_data_type': 'torch.bfloat16',
            'xqa_enable_pdl': True, 'xqa_unused_kv_policy': 'finite_full_initialization',
            'xqa_max_context': 32768, 'xqa_graph_padding': 'reserved_page0_one_token_discard',
            'graph_captured_batches': [1, 2, 4]}
        replay = {'class': 'FlashInferBackend', 'captured_batch_size': 2}
        def log(row):
            return 'NATIVE_KV_BACKEND '+json.dumps(row)+'\nNATIVE_KV_GRAPH_REPLAY '+json.dumps(replay)
        self.assertEqual(verify_backend(log(backend), 'bf16', BF16_XQA_OPERATORS['bf16']), backend)
        for key, value in [('decode_internal_q_data_type', 'torch.float8_e4m3fn'), ('planner_backend', 'fa2'),
                           ('xqa_enable_pdl', False), ('graph_captured_batches', [])]:
            bad = dict(backend); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_backend(log(bad), 'bf16', BF16_XQA_OPERATORS['bf16'])

    def test_forced_eos_disable_and_legacy_planner_cannot_enter_production(self):
        a, r = self.request(); r["workload"]["ignore_eos"] = True
        with self.assertRaises(ValueError): validate_request(r, a)
        a, r = self.request(); r["operators"]["fp8_e4m3"]["planner"] = "fa2"
        with self.assertRaises(ValueError): validate_request(r, a)

    def test_unbounded_concurrency_or_nonisolated_checkout_are_rejected(self):
        a, r = self.request(); r["concurrency"] = 2000
        with self.assertRaises(ValueError): validate_request(r, a)
        a, r = self.request(); r["python_source_repo"] = r["source_repo"]
        with self.assertRaises(ValueError): validate_request(r, a)

    def quality_request(self):
        a, r = self.request()
        r['concurrency'] = 8
        r['backend_source_sha256'].update({name: 'a' * 64 for name in QUALITY_SOURCES})
        r['workload'] = {'kind': 'native_quality_fa3', 'ignore_eos': False, 'cohort': 'pilot', 'samples_per_arm': 80,
                         'source_content_sha256': 'b' * 64,
                         'scorer_dependencies': a['approved_workspace'] + '/experiments/joint-kv-14b/longbench-deps-py312',
                         'longbench_repo': r['remote_output'] + '/input/scorer',
                         'gsm8k_shots': r['remote_output'] + '/input/five_train_examples.json',
                         'approved_data_manifest': r['remote_output'] + '/input/approved_data_manifest.json'}
        return a, r

    def test_production_quality_pins_query_profile_tools_and_readonly_dependencies(self):
        a, r = self.quality_request(); validate_request(r, a)
        command = candidate_quality_command({'request': r}, Path('/fixture'), 'fp8_e4m3')
        self.assertEqual(command[command.index('--native-planner') + 1], 'fa3')
        self.assertEqual(command[command.index('--fp8-q-scale') + 1], '1.0')
        self.assertEqual(command[command.index('--runtime-source-commit') + 1], r['python_source_commit'])
        self.assertEqual(command[command.index('--max-memory-utilization') + 1], '0.8')
        r['backend_source_sha256'].pop(QUALITY_SOURCES[0])
        with self.assertRaises(ValueError): validate_request(r, a)

    def full_reference_request(self):
        a, r = self.quality_request()
        r['arms'] = ['bf16']
        r['operators'] = copy.deepcopy(BF16_XQA_OPERATORS)
        r['backend_source_sha256'] = {name: 'a'*64 for name in XQA_SOURCES + QUALITY_SOURCES}
        r['workload'].update(kind='native_quality_xqa_bf16_reference', cohort='full_reference', samples_per_arm=1719)
        return a, r

    def test_full_bf16_reference_cannot_generate_fp8_or_change_cohort(self):
        a, r = self.full_reference_request()
        validate_request(r, a)
        for arms in (['fp8_e4m3'], ['bf16', 'fp8_e4m3']):
            bad = copy.deepcopy(r); bad['arms'] = arms
            with self.subTest(arms=arms), self.assertRaises(ValueError): validate_request(bad, a)
        for key, value in [('cohort', 'pilot'), ('samples_per_arm', 1639), ('kind', 'native_quality_xqa')]:
            bad = copy.deepcopy(r); bad['workload'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): validate_request(bad, a)

    def test_full_reference_command_uses_bf16_xqa_and_full_scope_without_fp8_q(self):
        _, r = self.full_reference_request()
        command = candidate_quality_command({'request': r}, Path('/fixture'), 'bf16')
        for flag, value in [('--native-planner', 'fa3'), ('--native-decode', 'xqa'),
                            ('--cohort-scope', 'full'), ('--arms', 'auto')]:
            self.assertEqual(command[command.index(flag)+1], value)
        self.assertNotIn('--fp8-q-scale', command)

    def test_production_batch_quality_is_explicit_full_paired_c128_only(self):
        a, r = self.full_reference_request()
        r.update(arms=['bf16', 'fp8_e4m3'], concurrency=128)
        r['workload'].update(kind='native_quality_xqa_production_batch', cohort='full_production_batch')
        validate_request(r, a)
        for key, value in [('concurrency', 8), ('concurrency', 64), ('arms', ['bf16'])]:
            bad = copy.deepcopy(r); bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): validate_request(bad, a)
        bad = copy.deepcopy(r); bad['workload']['samples_per_arm'] = 80
        with self.assertRaises(ValueError): validate_request(bad, a)

    def test_production_batch_commands_preserve_distinct_deployable_profiles_and_full_counts(self):
        _, r = self.full_reference_request()
        r.update(arms=['bf16', 'fp8_e4m3'], concurrency=128)
        r['workload'].update(kind='native_quality_xqa_production_batch', cohort='full_production_batch')
        for arm, planner in [('bf16', 'fa3'), ('fp8_e4m3', 'fa2')]:
            command = candidate_quality_command({'request': r}, Path('/fixture'), arm)
            for flag, value in [('--native-planner', planner), ('--native-decode', 'xqa'),
                                ('--cohort-scope', 'full'), ('--concurrency', '128')]:
                self.assertEqual(command[command.index(flag)+1], value)
            self.assertNotIn('--fp8-q-scale', command)

    def test_full_reference_operator_failure_blocks_model_with_exact_probe_configuration(self):
        _, r = self.full_reference_request()
        with tempfile.TemporaryDirectory() as directory:
            with patch('scripts.controller.production.subprocess.run', side_effect=[
                    subprocess.CompletedProcess([], 0), subprocess.CalledProcessError(1, ['operator'])]) as probe:
                with patch('scripts.controller.production.subprocess.Popen') as model:
                    with self.assertRaises(subprocess.CalledProcessError):
                        run_candidate_quality({'request': r}, Path(directory), Path(directory), 'bf16', {'PYTHONPATH': '/fixture'})
                    model.assert_not_called()
                self.assertEqual(probe.call_args.kwargs['env']['XLLM_FLASHINFER_NATIVE_PLANNER'], 'fa3')
                self.assertEqual(probe.call_args.kwargs['env']['XLLM_FLASHINFER_NATIVE_DECODE'], 'xqa')

    def test_full_bf16_reference_requires_padded_graph_and_initialization_evidence(self):
        _, r = self.full_reference_request()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'operator_acceptance.json').write_text(json.dumps({'real_capture_and_replay': True,
                'partial_page_and_replan': True, 'noncontiguous_q_checked': True, 'model_loaded': False}))
            with patch('scripts.controller.production.subprocess.run'):
                with patch('scripts.controller.production.subprocess.Popen') as model:
                    with self.assertRaises(ValueError):
                        run_candidate_quality({'request': r}, root, root, 'bf16', {'PYTHONPATH': '/fixture'})
                    model.assert_not_called()

    def test_quality_cannot_silently_use_full_cohort_or_alternate_dependencies(self):
        for field, wrong in [('cohort', 'full'), ('samples_per_arm', 1719), ('scorer_dependencies', '/shared')]:
            a, r = self.quality_request(); r['workload'][field] = wrong
            with self.assertRaises(ValueError): validate_request(r, a)

    def test_production_scorer_failure_prevents_model_start(self):
        _, request = self.quality_request()
        with tempfile.TemporaryDirectory() as directory:
            with patch('scripts.controller.production.subprocess.run', side_effect=subprocess.CalledProcessError(1, ['fixture'])) as cpu:
                with patch('scripts.controller.production.subprocess.Popen') as model:
                    with self.assertRaises(subprocess.CalledProcessError):
                        run_candidate_quality({'request': request}, Path(directory), Path(directory), 'bf16', {'PYTHONPATH': '/fixture'})
                    model.assert_not_called()
                self.assertEqual(cpu.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')

    def test_completion_fragment_requires_pinned_parent_and_never_claims_full(self):
        a, r = self.quality_request()
        r['workload'].update(cohort='completion_fragment', samples_per_arm=1639)
        with self.assertRaises(ValueError): validate_request(r, a)
        r['workload'].update(reuse_run_id='completed-parent', parent_result_sha256='c' * 64, parent_audit_sha256='d' * 64)
        validate_request(r, a)
        command = candidate_quality_command({'request': r}, Path('/fixture'), 'bf16')
        self.assertEqual(command[command.index('--cohort-scope') + 1], 'fragment')

    def test_xqa_pilot_pins_prefill_internal_decode_precision_and_new_source(self):
        a, r = self.quality_request()
        r['workload']['kind'] = 'native_quality_xqa'
        r['operators'] = copy.deepcopy(XQA_OPERATORS)
        r['backend_source_sha256'] = {name: 'a' * 64 for name in XQA_SOURCES + QUALITY_SOURCES}
        validate_request(r, a)
        command = candidate_quality_command({'request': r}, Path('/fixture'), 'fp8_e4m3')
        self.assertEqual(command[command.index('--native-planner') + 1], 'fa2')
        self.assertEqual(command[command.index('--native-decode') + 1], 'xqa')
        self.assertNotIn('--fp8-q-scale', command)
        r['operators']['fp8_e4m3']['decode_internal_q'] = 'bf16'
        with self.assertRaises(ValueError): validate_request(r, a)
        r['operators'] = copy.deepcopy(XQA_OPERATORS)
        r['backend_source_sha256'].pop('xllm/python/attention/xqa_decode.py')
        with self.assertRaises(ValueError): validate_request(r, a)

    def test_xqa_operator_failure_never_loads_a_model(self):
        _, r = self.quality_request(); r['workload']['kind'] = 'native_quality_xqa'
        with tempfile.TemporaryDirectory() as directory:
            with patch('scripts.controller.production.subprocess.run', side_effect=[None, subprocess.CalledProcessError(1, ['operator-fixture'])]) as run:
                with patch('scripts.controller.production.subprocess.Popen') as model:
                    with self.assertRaises(subprocess.CalledProcessError):
                        run_candidate_quality({'request': r}, Path(directory), Path(directory), 'fp8_e4m3', {'PYTHONPATH': '/fixture'})
                    model.assert_not_called()
                self.assertEqual(run.call_args.kwargs['env']['XLLM_FLASHINFER_NATIVE_DECODE'], 'xqa')

    def test_xqa_completion_requires_its_verified_parent_contract(self):
        a, r = self.quality_request()
        r['workload'].update(kind='native_quality_xqa', cohort='completion_fragment', samples_per_arm=1639)
        r['operators'] = copy.deepcopy(XQA_OPERATORS)
        r['backend_source_sha256'] = {name: 'a' * 64 for name in XQA_SOURCES + QUALITY_SOURCES}
        with self.assertRaises(ValueError): validate_request(r, a)
        r['workload'].update(reuse_run_id='verified-xqa-pilot', parent_result_sha256='c' * 64, parent_audit_sha256='d' * 64)
        validate_request(r, a)
        command = candidate_quality_command({'request': r}, Path('/fixture'), 'fp8_e4m3')
        self.assertEqual(command[command.index('--cohort-scope') + 1], 'fragment')
        self.assertEqual(command[command.index('--native-decode') + 1], 'xqa')
        r['arms'] = ['bf16']
        with self.assertRaises(ValueError): validate_request(r, a)
        r['workload']['recovery'] = {'prior_run_id': 'recovered-fp8-fragment', 'verified_arm': 'fp8_e4m3',
            'prior_terminal_result_sha256': 'e'*64, 'arm_measurement_sha256': 'f'*64, 'recovered_audit_sha256': '0'*64}
        validate_request(r, a)
        r['arms'] = ['fp8_e4m3']
        with self.assertRaises(ValueError): validate_request(r, a)

    def test_xqa_operator_exit0_without_graph_and_initialization_evidence_cannot_load_model(self):
        _, r = self.quality_request(); r['workload']['kind'] = 'native_quality_xqa'
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'operator_acceptance.json').write_text('{"real_capture_and_replay": false}')
            with patch('scripts.controller.production.subprocess.run'):
                with patch('scripts.controller.production.subprocess.Popen') as model:
                    with self.assertRaisesRegex(ValueError, 'Operator exit0'):
                        run_candidate_quality({'request': r}, Path(directory), Path(directory), 'fp8_e4m3', {'PYTHONPATH': '/fixture'})
                    model.assert_not_called()

    def test_anonymous_trace_cannot_enter_as_natural_semantic_requests(self):
        a, r = self.request()
        r['operators'] = copy.deepcopy(XQA_OPERATORS)
        r['backend_source_sha256'] = {name: 'a'*64 for name in XQA_SOURCES}
        r['concurrency'] = 1024
        r['dataset_path'] = r['remote_output']+'/input/payload_trace.jsonl'
        r['dataset_archive'] = {'relative_path': 'input/payload_trace.jsonl.gz', 'uncompressed_size_bytes': 1000}
        r['workload'] = {'kind': 'anonymous_trace_xqa', 'ignore_eos': True, 'requests': 1025, 'warmup': 0,
            'arrival_mode': 'fixed_schedule_continuous_warm_cache', 'content_scope': 'anonymous_arrival_length_prefix_emulation',
            'family': 'conversation', 'arrival_scaling': 1, 'trace_manifest': r['remote_output']+'/input/trace_manifest.json',
            'source_content_sha256': 'b'*64}
        validate_request(r, a)
        modern = copy.deepcopy(r)
        modern['workload']['kind'] = 'anonymous_trace_xqa_bf16_reference'
        modern['operators'] = copy.deepcopy(BF16_XQA_OPERATORS)
        validate_request(modern, a)
        modern['operators']['bf16']['decode_internal_q'] = 'fp8_e4m3'
        with self.assertRaises(ValueError): validate_request(modern, a)
        for key, value in (('ignore_eos', False), ('warmup', 8), ('arrival_mode', 'closed_loop'),
                           ('arrival_scaling', float('inf')), ('trace_manifest', '/unowned')):
            bad = copy.deepcopy(r); bad['workload'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): validate_request(bad, a)


class ProductionRawAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.artifacts = self.root / "artifacts"; self.artifacts.mkdir()
        self.source = self.root / "input.json"
        sessions, self.rows, self.raws = [], [], []
        origin = datetime.datetime(2026, 10, 26, tzinfo=datetime.timezone.utc).timestamp()
        for index, (phase, tokens) in enumerate((("warmup", 2), ("profiling", 1), ("profiling", 2))):
            session = "source-" + str(index)
            payload = {"messages": [{"role": "user", "content": "fixture only"}], "max_tokens": 1024}
            sessions.append({"session_id": session, "payloads": [payload]})
            meta = {"benchmark_phase": phase, "x_request_id": "r" + str(index), "conversation_id": session,
                    "was_cancelled": False, "context_overflow_skip": False, "turn_index": 0,
                    "request_start_ns": int((origin + index * .1) * 1e9), "request_end_ns": int((origin + index * .1 + .011) * 1e9)}
            values = {"output_token_count": tokens, "input_sequence_length": 30, "usage_completion_tokens": tokens,
                      "usage_prompt_tokens": 30, "usage_total_tokens": 30 + tokens,
                      "request_latency": 11, "time_to_first_token": 10, "decode_duration": tokens - 1,
                      "inter_token_latency": 1 if tokens > 1 else 0}
            self.rows.append({"metadata": meta, "metrics": {name: {"value": value} for name, value in values.items()}, "error": None})
            response = {"choices": [{"finish_reason": "stop"}], "usage": {"prompt_tokens": 30, "completion_tokens": tokens}}
            self.raws.append({"metadata": meta, "payload": payload, "status": 200, "error": None,
                              "responses": [{"packets": [{"name": "data", "value": json.dumps(response)}, {"name": "data", "value": "[DONE]"}]}]})
        self.source.write_text(json.dumps({"data": sessions}))
        summary = {"aiperf_version": "0.13.0", "is_complete": True, "was_cancelled": False, "error_summary": {},
                   "start_time": "2026-10-26T00:00:00+00:00", "end_time": "2026-10-26T00:00:01+00:00",
                   "output_token_throughput": {"avg": 3}, "request_throughput": {"avg": 2}}
        (self.artifacts / "profile_export_aiperf.json").write_text(json.dumps(summary))
        (self.root / "gpu_telemetry.csv").write_text("utc,memory_used_mib,compute_util_pct,memory_util_pct\n2026-10-26T00:00:00.150000+00:00,1000,50,20\n")

    def tearDown(self): self.temp.cleanup()

    def run_audit(self):
        (self.artifacts / "profile_export.jsonl").write_text("\n".join(json.dumps(row) for row in self.rows))
        (self.artifacts / "profile_export_raw.jsonl").write_text("\n".join(json.dumps(row) for row in self.raws))
        return audit_requests(self.artifacts, self.source, 2, 1)

    def test_natural_one_token_answer_is_valid_and_has_no_invented_itl(self):
        result = self.run_audit()
        self.assertEqual(result["one_token_answers"], 1)
        self.assertEqual(result["itl_p50_ms"], 1)
        self.assertEqual(result["finish_reason_counts"], {"stop": 2})
        self.assertTrue(result["http_payload_and_sse_verified"])

    def test_wire_payload_drift_is_rejected(self):
        self.raws[1]["payload"] = {**self.raws[1]["payload"], "ignore_eos": True}
        with self.assertRaises(ValueError): self.run_audit()

    def test_missing_stream_done_is_rejected(self):
        self.raws[1]["responses"][0]["packets"].pop()
        with self.assertRaises(ValueError): self.run_audit()

    def test_duplicate_or_missing_raw_record_is_rejected(self):
        self.raws.pop()
        with self.assertRaises(ValueError): self.run_audit()

    def test_zero_usage_cannot_be_counted_as_success(self):
        self.rows[1]["metrics"]["output_token_count"]["value"] = 0
        with self.assertRaises(ValueError): self.run_audit()

    def test_raw_phase_or_prompt_usage_drift_cannot_pass(self):
        self.raws[1]['metadata'] = dict(self.raws[1]['metadata'], benchmark_phase='warmup')
        with self.assertRaises(ValueError): self.run_audit()
        self.raws[1]['metadata']['benchmark_phase'] = 'profiling'
        packet = self.raws[1]['responses'][0]['packets'][0]
        value = json.loads(packet['value']); value['usage']['prompt_tokens'] = 99
        packet['value'] = json.dumps(value)
        with self.assertRaises(ValueError): self.run_audit()


if __name__ == "__main__": unittest.main()
