"""Quantized model jobs must bind to an actual completed calibration identity."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.controller.approval import ApprovalError
from scripts.controller.durable import digest
from scripts.controller.nextgen import MODEL
from scripts.controller.nextgen_service import verify_scale_prerequisites
from test_nextgen_native_service import _service


class NativeScalePrerequisiteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        _, self.request = _service('fp8_e4m3')
        self.request.update(remote_output=str(self.base/'agent_work/runs/new'), run_id='new')
        self.path = Path(self.request['remote_output'])/'input/scale_calibration_audit.json'
        self.path.parent.mkdir(parents=True)
        self.prior = self.base/'agent_work/runs/calibration'
        (self.prior/'output/bf16').mkdir(parents=True)
        old = {**self.request, 'run_id': 'calibration', 'arms': ['bf16'], 'range_observation': True}
        self.old_ticket = {'request': old, 'request_sha256': digest(old)}
        self.result = {'status': 'COMPLETE', 'gpu_release_verified': True, 'live_owned_containers': [],
                       'request_sha256': self.old_ticket['request_sha256']}
        self.snapshot = {'FP8_E4M3_candidate_scales': self.request['kv_scales'],
                         'request_sha256': self.old_ticket['request_sha256']}
        self.audit = {'state': 'BF16_CALIBRATION_INDEPENDENTLY_VERIFIED', 'model_id': MODEL,
                      'run_id': 'calibration', 'actual_observer_GPU_Graph_passed': True,
                      'candidate_scales': {'fp8_e4m3': self.request['kv_scales']},
                      'calibration_request_sha256': self.old_ticket['request_sha256']}
        (self.prior/'output/bf16/observer_Graph.json').write_text('{}')
        self._freeze()

    def _freeze(self):
        for name, value in [('ticket.json', self.old_ticket), ('result.json', self.result),
                            ('output/bf16/kv_observer_snapshot.json', self.snapshot)]:
            (self.prior/name).write_text(json.dumps(value))
        self.audit.update(terminal_result_sha256=hashlib.sha256((self.prior/'result.json').read_bytes()).hexdigest(),
                          snapshot_sha256=hashlib.sha256((self.prior/'output/bf16/kv_observer_snapshot.json').read_bytes()).hexdigest())
        self.path.write_text(json.dumps(self.audit))
        self.request['scale_calibration_audit_sha256'] = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def test_exact_completed_BF16_calibration_identity_and_scales_are_admitted(self):
        verify_scale_prerequisites(self.request, self.base, os.getuid())

    def test_modified_audit_or_requested_scales_are_rejected(self):
        self.path.write_text('{}')
        with self.assertRaises(ApprovalError):
            verify_scale_prerequisites(self.request, self.base, os.getuid())
        self._freeze()
        self.request['kv_scales'] = {'key': 1.0, 'value': 1.0}
        with self.assertRaises(ApprovalError):
            verify_scale_prerequisites(self.request, self.base, os.getuid())

    def test_calibration_cannot_be_borrowed_from_running_or_unreleased_job(self):
        for mutation in ('running', 'unreleased', 'live_container'):
            with self.subTest(mutation=mutation):
                self.result.update(status='COMPLETE', gpu_release_verified=True, live_owned_containers=[])
                if mutation == 'running':self.result['status'] = 'RUNNING'
                elif mutation == 'unreleased':self.result['gpu_release_verified'] = False
                else:self.result['live_owned_containers'] = ['still-running']
                self._freeze()
                with self.assertRaises(ApprovalError):
                    verify_scale_prerequisites(self.request, self.base, os.getuid())

    def test_changed_ticket_cannot_hide_behind_its_previous_digest(self):
        self.old_ticket['request']['binary_sha256'] = '0'*64
        self._freeze()
        with self.assertRaises(ApprovalError):
            verify_scale_prerequisites(self.request, self.base, os.getuid())


if __name__ == '__main__':
    unittest.main()
