import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.controller.approval import ApprovalError
from scripts.controller import q4_component_profile as profile
from scripts.controller.idle_admission import FAST_ADMISSION, idle_parameters
from scripts.controller.durable import digest


class ReadonlyComponentAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        raw = b'# immutable CPU fixture; no GPU import\n'
        (root/'q4_component_probe.py').write_bytes(raw)
        self.module_file = str(root/'q4_component_profile.py')
        self.approval = {'identity': 'same authorization object'}
        self.request = {
            'workload': {'kind': profile.KIND, 'scope': profile.SCOPE},
            'profile_protocol': copy.deepcopy(profile.PROTOCOL), 'profile_phase': 'pilot',
            'graph_qualification_failure_reference': copy.deepcopy(profile.GRAPH_FAILURE_REFERENCE),
            'python_source_commit': profile.SOURCE_COMMIT,
            'next_stage_profile': 'ocp_mxfp4_20261105', 'server_binary_execution': 'none',
            'probe_driver_sha256': hashlib.sha256(raw).hexdigest(), 'max_wall_seconds': 1800,
            'gpu_uuids': ['unchanged-single-device'], 'quota_identity': 'unchanged-budget',
            'opaque_source_identity': {'digest': 'preserve'},
        }

    def validate(self, request, guard):
        with patch.object(profile, '__file__', self.module_file), patch(
                'scripts.controller.native_next_stage.validate_request', guard), patch.object(
                profile, 'PROBE_DRIVER_SHA256', self.request['probe_driver_sha256']):
            profile.validate_request(request, self.approval)

    def test_delegates_all_identity_authority_and_resource_fields_unchanged(self):
        calls = []
        self.validate(self.request, lambda request, approval: calls.append((request, approval)))
        self.assertEqual(len(calls), 1)
        common, approval = calls[0]
        self.assertIs(approval, self.approval)
        self.assertEqual(common['workload']['kind'], 'upstream_next_stage_native_operator')
        self.assertEqual({k: v for k, v in common.items() if k != 'workload'},
                         {k: v for k, v in self.request.items() if k != 'workload'})

    def test_unsupported_shapes_sources_driver_and_modes_fail_before_shared_guard(self):
        changes = [
            ('profile_protocol', dict(profile.PROTOCOL, head_dim=512)),
            ('profile_protocol', dict(profile.PROTOCOL, KV_heads=1)),
            ('profile_protocol', dict(profile.PROTOCOL, formats=['mxfp4', 'sglang'])),
            ('profile_protocol', dict(profile.PROTOCOL, candidate_runtime_source_mutations=True)),
            ('profile_phase', 'unbounded_search'), ('python_source_commit', '0'*40),
            ('probe_driver_sha256', '0'*64), ('server_binary_execution', 'required'),
            ('max_wall_seconds', 1799), ('max_wall_seconds', 3601),
            ('next_stage_profile', None),
            ('profile_phase', 'main'),
            ('graph_qualification_failure_reference', None),
            ('graph_qualification_failure_reference', dict(profile.GRAPH_FAILURE_REFERENCE, result_sha256='0'*64)),
        ]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                request = copy.deepcopy(self.request)
                request[key] = value
                with self.assertRaises(ApprovalError):
                    self.validate(request, lambda *_: self.fail('Invalid probe reached shared guard'))

    def test_shared_guard_rejection_is_never_bypassed(self):
        def reject(*_):
            raise ApprovalError('foreign GPU or exhausted authority')
        with self.assertRaisesRegex(ApprovalError, 'foreign GPU'):
            self.validate(self.request, reject)

    def test_readonly_profile_uses_the_same_single_gpu_fast_admission_policy(self):
        request = dict(self.request, resource_admission=dict(FAST_ADMISSION))
        self.assertEqual(idle_parameters(request), (0, 5))
        for changes in ({'gpu_uuids': ['one', 'two']},
                        {'resource_admission': dict(FAST_ADMISSION, foreign_gpu_action='ignore')}):
            with self.assertRaises(ApprovalError):
                idle_parameters(dict(request, **changes))

    def test_main_needs_an_immutable_pilot_identity_before_shared_admission(self):
        request = dict(self.request, profile_phase='main',
                       profile_pilot_reference={'run_id': 'qualified-pilot', 'result_sha256': 'a'*64})
        calls = []
        self.validate(request, lambda *args: calls.append(args))
        self.assertEqual(len(calls), 1)
        request['profile_pilot_reference']['run_id'] = '../different-run'
        with self.assertRaises(ApprovalError):
            self.validate(request, lambda *_: self.fail('Unsafe reference reached shared guard'))


class DiagnosticCompletionTest(unittest.TestCase):
    def setUp(self):
        self.acceptance = {
            'schema': 'q4_supported_component_acceptance_v3', 'phase': 'pilot',
            'completed': True, 'pass': True, 'source_checked': True,
            'all_independent_reference_passed': True, 'same_fused_MX_NV_control_passed': True,
            'all_writer_graph_checks_passed': True, 'all_native_decode_graph_checks_passed': True,
            'all_component_gpu_activity_observed': True,
            'strict_prefill_native_decode_equivalence_all_pass': False,
            'all_requested_graph_paths_supported': False, 'production_graph_qualification': False,
            'blocked_graph_routes': list(profile.BLOCKED_GRAPH_ROUTES),
            'blocked_graph_reason': profile.BLOCKED_GRAPH_REASON,
            'model_loaded': False, 'service_benchmark': False, 'full_vllm_engine_tested': False,
            'sglang_target': 'unsupported_NA', 'artifact_bytes_before_acceptance': 1024,
        }
        self.result = {'schema': 'q4_supported_component_result_v3',
                       'phase': 'pilot', 'completed': True, 'source_checked': True,
                       **{key: self.acceptance[key] for key in (
                           'all_independent_reference_passed', 'same_fused_MX_NV_control_passed',
                           'all_writer_graph_checks_passed', 'all_native_decode_graph_checks_passed',
                           'all_component_gpu_activity_observed', 'strict_prefill_native_decode_equivalence_all_pass',
                           'all_requested_graph_paths_supported', 'production_graph_qualification',
                           'blocked_graph_routes', 'blocked_graph_reason')},
                       'service_or_model_quality_claim': False,
                       'cases': [{'case': {'name': name}} for name in
                                 ('pilot_prefill_b1_s128', 'pilot_decode_b2_s129')]}

    def test_complete_pilot_is_a_diagnostic_only(self):
        profile.validate_acceptance(self.acceptance, self.result, 'pilot')

    def test_false_pass_wrong_phase_missing_cases_and_oversized_evidence_are_rejected(self):
        variants = [(dict(self.acceptance, **{'pass': False}), self.result),
                    (dict(self.acceptance, same_fused_MX_NV_control_passed=False), self.result),
                    (dict(self.acceptance, all_independent_reference_passed=False), self.result),
                    (dict(self.acceptance, all_route_conformance_passed=True), self.result),
                    (dict(self.acceptance, all_graph_checks_passed=True), self.result),
                    (dict(self.acceptance, production_graph_qualification=True), self.result),
                    (dict(self.acceptance, all_requested_graph_paths_supported=True), self.result),
                    (dict(self.acceptance, all_component_gpu_activity_observed=False), self.result),
                    (dict(self.acceptance, blocked_graph_routes=[]), self.result),
                    (dict(self.acceptance, strict_prefill_native_decode_equivalence_all_pass=None), self.result),
                    (dict(self.acceptance, artifact_bytes_before_acceptance=65*2**20), self.result),
                    (self.acceptance, dict(self.result, phase='main')),
                    (self.acceptance, dict(self.result, strict_prefill_native_decode_equivalence_all_pass=True)),
                    (self.acceptance, dict(self.result, strict_prefill_native_decode_equivalence_all_pass=0)),
                    (self.acceptance, dict(self.result, all_independent_reference_passed=1)),
                    (self.acceptance, dict(self.result, cases=self.result['cases'][:1]))]
        for acceptance, result in variants:
            with self.subTest(acceptance=acceptance, result=result), self.assertRaises(ValueError):
                profile.validate_acceptance(acceptance, result, 'pilot')

    def test_main_rejects_a_different_driver_even_after_pilot_run_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/'new-output'
            output.mkdir()
            pilot = root/'output/mxfp4/profile'
            pilot.mkdir(parents=True)
            (pilot/'acceptance.json').write_text(json.dumps(self.acceptance))
            (pilot/'result.json').write_text(json.dumps(self.result))
            request = {'profile_phase': 'main', 'probe_driver_sha256': 'a'*64,
                       'profile_pilot_reference': {'run_id': 'pilot', 'result_sha256': 'b'*64}}
            old = dict(request, profile_phase='pilot')
            with patch('scripts.controller.next_stage_deployment._completed_reference', return_value=(root, old)):
                profile.verify_pilot(request, output)
            old['probe_driver_sha256'] = 'c'*64
            with patch('scripts.controller.next_stage_deployment._completed_reference', return_value=(root, old)):
                with self.assertRaises(ValueError):
                    profile.verify_pilot(request, output)


class OriginalGraphRejectionTest(unittest.TestCase):
    def test_released_failure_is_linked_as_failure_and_unknown_cannot_authorize_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = {'python_source_commit': profile.SOURCE_COMMIT, 'image_id': 'pinned-image',
                   'probe_driver_sha256': 'a'*64, 'backend_source_sha256': {'source': 'b'*64},
                   'workload': {'kind': profile.KIND}, 'profile_phase': 'pilot'}
            result = {'run_id': 'old-pilot', 'status': 'FAILED', 'gpu_release_verified': True,
                      'live_owned_containers': [], 'arm_results': [], 'request_sha256': digest(old)}
            failure = {'type': 'AcceleratorError', 'message': 'CUDA error: an illegal memory access was encountered'}
            prior = root/'agent_work/runs/old-pilot'
            (prior/'output/mxfp4/profile').mkdir(parents=True)
            raw = json.dumps(result).encode()
            failure_raw = json.dumps(failure).encode()
            (prior/'result.json').write_bytes(raw)
            (prior/'ticket.json').write_text(json.dumps({'request': old, 'request_sha256': digest(old)}))
            (prior/'output/mxfp4/profile/failure.json').write_bytes(failure_raw)
            reference = {'run_id': 'old-pilot', 'result_sha256': hashlib.sha256(raw).hexdigest(),
                         'failure_sha256': hashlib.sha256(failure_raw).hexdigest(), 'driver_sha256': 'a'*64}
            request = dict(old, approved_workspace=str(root), graph_qualification_failure_reference=reference)
            output = root/'new-output'
            output.mkdir()
            with patch.object(profile, 'GRAPH_FAILURE_REFERENCE', reference):
                profile.verify_original_graph_rejection(request, output)
            proof = json.loads((output/'original_graph_rejection.json').read_bytes())
            self.assertEqual(proof['original_status'], 'FAILED')
            self.assertFalse(proof['production_graph_qualification'])
            result['status'] = 'UNKNOWN'
            raw = json.dumps(result).encode()
            (prior/'result.json').write_bytes(raw)
            reference['result_sha256'] = hashlib.sha256(raw).hexdigest()
            with patch.object(profile, 'GRAPH_FAILURE_REFERENCE', reference):
                with self.assertRaises(ValueError):
                    profile.verify_original_graph_rejection(request, output)


if __name__ == '__main__':
    unittest.main()
