"""A natural long-document service run cannot enable forced lengths or probes."""

import copy
import unittest
from pathlib import Path

from scripts.controller.approval import ApprovalError
from scripts.controller.policy import BF16_XQA_OPERATORS, XQA_SOURCES, validate_request
from scripts.controller.production import aiperf_service_command
from test_range_probe import _range_request


def _service_request() -> tuple:
    approval, request = _range_request()
    request.update(arms=['bf16', 'fp8_e4m3'], max_output_gib=5,
        backend_source_sha256={name:'a'*64 for name in XQA_SOURCES},
        operators=copy.deepcopy(BF16_XQA_OPERATORS),
        dataset_sha256='1a36f83244b09b3edd85895586ec324624640ade004eb3ece672e94ee5dcfe5a',
        workload={'kind':'long_document_QA_natural_xqa_bf16_reference', 'requests':128,
            'warmup':0, 'ignore_eos':False, 'content_scope':'heldout_original_LongBench_long_document_QA',
            'arrival_mode':'closed_loop_concurrency_finite_cohort', 'official_generation_caps_preserved':True,
            'aiperf_zero_warmup_policy':'omit_explicit_option', 'activation_diagnostic_enabled':False,
            'input_manifest':request['remote_output']+'/input/production_manifest.json',
            'aiperf_cli_config_sha256':'2be937b185ddfe3a19e8090aa4411f1015a2e00899d29db3055d3f7e19bdd42b',
            'source_content_sha256':'cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64'})
    return approval, request


class LongQAServiceTests(unittest.TestCase):
    def test_same_modern_paired_profile_accepts_natural_full128(self) -> None:
        approval, request = _service_request()
        validate_request(request, approval)
        command = aiperf_service_command(request, Path('/results'))
        self.assertNotIn('--warmup-request-count', command)
        self.assertEqual(command[command.index('--request-count')+1], '128')
        self.assertEqual(command[command.index('--custom-dataset-type')+1], 'inputs_json')

    def test_forced_length_or_observer_or_changed_scope_rejected(self) -> None:
        approval, request = _service_request()
        for field, value in (('ignore_eos', True), ('activation_diagnostic_enabled', True),
                             ('official_generation_caps_preserved', False), ('requests', 256),
                             ('warmup', 16), ('arrival_mode', 'fixed_schedule_continuous_warm_cache')):
            with self.subTest(field=field):
                changed = copy.deepcopy(request)
                changed['workload'][field] = value
                with self.assertRaises(ApprovalError):
                    validate_request(changed, approval)

    def test_changed_inputs_scale_or_unbounded_job_rejected(self) -> None:
        approval, request = _service_request()
        for field, value in (('dataset_sha256', 'd'*64), ('max_wall_seconds', 3600),
                             ('fp8_scales', {'key':.125,'value':.125}), ('max_output_gib', 10)):
            with self.subTest(field=field):
                changed = {**request, field:value}
                with self.assertRaises(ApprovalError):
                    validate_request(changed, approval)


if __name__ == '__main__':
    unittest.main()
