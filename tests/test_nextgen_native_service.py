"""Admit calibrated new-binary native formats without weakening GPU guards."""

import copy
import json
import unittest
from unittest.mock import patch

from scripts.controller.approval import ApprovalError
from scripts.controller.nextgen import GDN_OVERLAY, QUALITY_KIND, SERVICE_KIND, native_python_environment
from scripts.controller.nextgen_service import (SERVICE_SOURCE_FILES, _calibration_request,
                                                _operator_environment, native_server_command)
from scripts.controller.policy import validate_request
from test_nextgen_contract import _fixture


def _service(arm='bf16'):
    approval, request, _ = _fixture()
    request.update(arms=[arm], max_wall_seconds=3600, startup_seconds=900, server_binary_execution='required',
                   backend_source_sha256={name: 'a'*64 for name in SERVICE_SOURCE_FILES},
                   workload={'kind': SERVICE_KIND,
                             'scope': 'format_correctness_lifecycle_and_optional_untimed_calibration'},
                   service={'max_batch': 8, 'max_tokens_per_batch': 2048, 'linear_state_slots': 16,
                            'max_context': 131072, 'max_memory_utilization': .85, 'page_size': 64},
                   range_observation=arm == 'bf16', calibration_count=136, calibration_concurrency=4,
                   calibration_sha256='c'*64,
                   runtime_dependency_overlay={'overlay_name': GDN_OVERLAY, 'overlay_manifest_sha256': 'd'*64,
                                               'CPU_import_evidence_sha256': 'e'*64},
                   operators={arm: {'planner': 'fa3' if arm == 'bf16' else 'fa2',
                                    'decode': 'flashinfer' if arm == 'nvfp4' else 'xqa'}})
    if arm != 'bf16':
        request.update(kv_scales={'key': .03125, 'value': .015625}, scale_calibration_audit_sha256='f'*64)
    return approval, request


class NewBinaryNativeServiceTests(unittest.TestCase):
    def test_core_quality_is_distinct_observer_free_and_bounded_for_real_1720_inputs(self):
        approval,request=_service()
        request.update(range_observation=False,max_wall_seconds=7200,
            workload={'kind':QUALITY_KIND,'scope':'natural_stop_core_quality_generations_not_service_performance'},
            service={**request['service'],'max_batch':16,'linear_state_slots':32,'max_memory_utilization':.9},
            quality_evaluation={'requests':1720,'client_concurrency':16,'client_token_budget':160000,
                                'request_timeout_seconds':600,'max_seconds':4800,
                                'requests_sha256':'a'*64,'source_manifest_sha256':'b'*64})
        validate_request(request,approval)
        for name,value in [('range_observation',True),('max_wall_seconds',10000),
                           ('quality_evaluation',{**request['quality_evaluation'],'client_token_budget':512000})]:
            with self.subTest(name=name),self.assertRaises(ApprovalError):
                validate_request({**request,name:value},approval)

    def test_all_three_explicit_formats_keep_typed_production_options(self):
        for arm in ('bf16', 'fp8_e4m3', 'nvfp4'):
            with self.subTest(arm=arm):
                approval, request = _service(arm)
                validate_request(request, approval)
                command = native_server_command(request, arm)
                self.assertIn('--kv_cache_backend=flashinfer', command)
                self.assertIn('--block_size=64', command)
                self.assertIn('--kv_cache_dtype=' + ('auto' if arm == 'bf16' else arm), command)
                self.assertEqual(sum(value.startswith('--kv_cache_dtype=') for value in command), 1)

    def test_no_binary_identity_exception_or_old_recovery_is_available(self):
        approval, request = _service()
        for name, value in [('server_binary_execution', 'none'), ('startup_recovery', {'prior': 'x'}),
                            ('environment_recovery', {'prior': 'x'}), ('backend_source_sha256', {}),
                            ('python_source_repo', '/another/tree'), ('arms', ['bf16', 'nvfp4'])]:
            with self.subTest(name=name), self.assertRaises(ApprovalError):
                validate_request({**request, name: value}, approval)

    def test_wrong_native_kernels_and_quantized_observation_are_refused(self):
        approval, request = _service('nvfp4')
        for changes in [{'range_observation': True}, {'operators': {'nvfp4': {'planner': 'fa3', 'decode': 'flashinfer'}}},
                        {'operators': {'nvfp4': {'planner': 'fa2', 'decode': 'xqa'}}}, {'kv_scales': {'key': 0, 'value': 1}},
                        {'scale_calibration_audit_sha256': ''}]:
            with self.subTest(changes=changes), self.assertRaises(ApprovalError):
                validate_request({**request, **changes}, approval)

    def test_calibration_and_dependency_file_identities_remain_required(self):
        approval, request = _service()
        for name, value in [('calibration_count', 0), ('calibration_concurrency', 8),
                            ('calibration_sha256', ''), ('runtime_dependency_overlay', {}),
                            ('startup_seconds', 1500), ('max_wall_seconds', 10000), ('graph_mode', 'off')]:
            with self.subTest(name=name), self.assertRaises(ApprovalError):
                validate_request({**request, name: value}, approval)
        changed = copy.deepcopy(request)
        changed['runtime_dependency_overlay']['overlay_name'] = 'install-something-else'
        with self.assertRaises(ApprovalError):
            validate_request(changed, approval)

    def test_independent_overlay_preserves_cuda_libraries_and_does_not_mutate_caller(self):
        _, request = _service()
        environment = {'LD_LIBRARY_PATH': '/existing/cuda', 'PYTHONPATH': '/existing/runtime'}
        result = native_python_environment(request, environment)
        self.assertEqual(result['LD_LIBRARY_PATH'], environment['LD_LIBRARY_PATH'])
        self.assertTrue(result['PYTHONPATH'].endswith(':/existing/runtime'))
        self.assertIn('/'+GDN_OVERLAY+'/site-packages', result['PYTHONPATH'])
        self.assertEqual(environment['PYTHONPATH'], '/existing/runtime')

    def test_tiny_operator_cannot_bind_the_full_model_observation_configuration(self):
        original = {'XLLM_KV_OBSERVATION_CONFIG': '/results/full-model-observer.json',
                    'PYTHONPATH': '/private/verified-overlay', 'LD_LIBRARY_PATH': '/existing/cuda'}
        result = _operator_environment(original)
        self.assertNotIn('XLLM_KV_OBSERVATION_CONFIG', result)
        self.assertIn('XLLM_KV_OBSERVATION_CONFIG', original)
        self.assertEqual(result['LD_LIBRARY_PATH'], original['LD_LIBRARY_PATH'])

    def test_calibration_requires_actual_new_tokenizer_usage_and_preserves_response_bytes(self):
        response = {'usage': {'prompt_tokens': 123, 'completion_tokens': 1},
                    'choices': [{'finish_reason': 'stop', 'message': {'content': '42'}}]}
        class Response:
            def __enter__(self):return self
            def __exit__(self, *_args):return False
            def read(self, _maximum):return json.dumps(response).encode()
        row = {'id': 'calibration-fixture', 'payload': {'messages': [], 'max_tokens': 1024},
               'prompt_tokens_cpu': 123}
        with patch('urllib.request.urlopen', return_value=Response()):
            self.assertTrue(_calibration_request(row, 'Qwen3.8-27B')['contract_valid'])
            timed=_calibration_request(row,'Qwen3.8-27B')
            self.assertGreater(timed['request_ended_monotonic_ns'],timed['request_started_monotonic_ns'])
            response['usage']['prompt_tokens'] = 2
            result = _calibration_request(row, 'Qwen3.8-27B')
            self.assertFalse(result['contract_valid'])
            self.assertTrue(result['raw_HTTP_body_base64'])


if __name__ == '__main__':
    unittest.main()
