"""P1 evidence gates reject partial, mislabeled and cross-format qualification."""

import copy
import unittest

from scripts.controller.idle_admission import FAST_ADMISSION, idle_parameters
from scripts.controller.kv_p1_operator import (
    KIND, MX_GATES, NV_GATES, PROTOCOL, operator_acceptance_passed,
)


class P1AcceptanceTest(unittest.TestCase):
    def acceptance(self, mx=False):
        return {
            'schema': 'kv_p0_p1_operator_acceptance_v1', 'phase': 'main',
            'completed': True, 'pass': True, 'source_checked': True, 'model_loaded': False,
            'runtime_source_commit': '1'*40, 'tool_sha256': '2'*64, 'tool_receipt_sha256': '3'*64,
            'protocol': copy.deepcopy(PROTOCOL),
            'qualified_formats': ['mxfp4'] if mx else ['bf16', 'fp8_e4m3', 'nvfp4'],
            'gates': {key: True for key in (MX_GATES if mx else NV_GATES)},
        }

    def test_same_nv_component_receipt_requires_all_actual_format_gates(self):
        value = self.acceptance()
        for arm in ('bf16', 'fp8_e4m3', 'nvfp4'):
            self.assertTrue(operator_acceptance_passed(value, arm))
        self.assertFalse(operator_acceptance_passed(value, 'mxfp4'))
        for gate in NV_GATES:
            partial = copy.deepcopy(value)
            del partial['gates'][gate]
            self.assertFalse(operator_acceptance_passed(partial, 'nvfp4'))

    def test_every_graph_and_replanning_gate_is_required(self):
        value = self.acceptance(mx=True)
        self.assertTrue(operator_acceptance_passed(value, 'mxfp4'))
        for gate in MX_GATES:
            partial = copy.deepcopy(value)
            partial['gates'][gate] = False
            self.assertFalse(operator_acceptance_passed(partial, 'mxfp4'))

    def test_capture_or_old_source_receipt_is_not_new_qualification(self):
        for field, changed in [('schema', 'q4_supported_component_acceptance_v3'),
                               ('source_checked', False), ('completed', False),
                               ('runtime_source_commit', 'unverified'), ('tool_receipt_sha256', ''),
                               ('qualified_formats', ['nvfp4']), ('model_loaded', True)]:
            value = self.acceptance()
            value[field] = changed
            self.assertFalse(operator_acceptance_passed(value, 'nvfp4'))

    def test_new_operator_keeps_exact_user_five_second_policy(self):
        request = {'workload': {'kind': KIND}, 'gpu_uuids': ['one'],
                   'resource_admission': dict(FAST_ADMISSION)}
        self.assertEqual(idle_parameters(request), (0, 5))
        request['resource_admission']['foreign_gpu_action'] = 'ignore'
        with self.assertRaises(ValueError):
            idle_parameters(request)


if __name__ == '__main__':
    unittest.main()
