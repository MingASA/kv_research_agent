"""Completion cannot reuse unverified predictions or change the production engine."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.controller.durable import digest
from scripts.controller.policy import PRODUCTION_SOURCES, XQA_SOURCES
from scripts.controller.supervisor import UnsafeState, verify_quality_parent


class QualityParentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.parent = self.base / 'agent_work/runs/fixture-parent'
        self.root = self.base / 'agent_work/runs/fixture-completion'
        self.parent.mkdir(parents=True); (self.root / 'input').mkdir(parents=True)
        self.request = {key: 'fixture' for key in ('binary_sha256', 'model_config_sha256', 'operators', 'service', 'graph_mode', 'fp8_scales', 'concurrency')}
        self.request['backend_source_sha256'] = {name: 'a' * 64 for name in PRODUCTION_SOURCES}
        self.request['workload'] = {'cohort': 'completion_fragment', 'reuse_run_id': 'fixture-parent'}
        prior = copy.deepcopy(self.request); prior['workload'] = {'cohort': 'pilot'}
        self._write(self.parent / 'ticket.json', {'request': prior})
        result = {'status': 'COMPLETE', 'gpu_release_verified': True, 'live_owned_containers': [],
                  'arm_results': [{}, {}], 'request_sha256': digest(prior)}
        result_sha = self._write(self.parent / 'result.json', result)
        self._write(self.parent / 'COMPLETE', {'result_sha256': result_sha})
        audit_sha = self._write(self.root / 'input/parent_quality_audit.json', {
            'terminal_result_sha256': result_sha, 'classification': 'confirmed_pilot_prediction_integrity_and_rescoring',
            'samples_per_arm': 80, 'run_id': 'fixture-parent'})
        self.request['workload'].update(parent_result_sha256=result_sha, parent_audit_sha256=audit_sha)

    def tearDown(self):
        self.temp.cleanup()

    def _write(self, path, value):
        data = (json.dumps(value) + '\n').encode(); path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def _check(self):
        verify_quality_parent(self.request, self.base, self.root, os.getuid())

    def test_verified_complete_parent_with_matching_engine_is_accepted(self):
        self._check()

    def test_missing_completion_or_changed_result_is_rejected(self):
        (self.parent / 'COMPLETE').unlink()
        with self.assertRaises(FileNotFoundError): self._check()

    def test_fake_independent_audit_cannot_unlock_completion(self):
        path = self.root / 'input/parent_quality_audit.json'
        self.request['workload']['parent_audit_sha256'] = self._write(path, {'classification': 'proposal'})
        with self.assertRaises(UnsafeState): self._check()

    def test_changed_operator_or_backend_cannot_reuse_old_quality(self):
        self.request['backend_source_sha256'][PRODUCTION_SOURCES[0]] = 'b' * 64
        with self.assertRaises(UnsafeState): self._check()

    def test_parent_path_traversal_is_rejected(self):
        self.request['workload']['reuse_run_id'] = '../foreign'
        with self.assertRaises(UnsafeState): self._check()

    def test_xqa_completion_cannot_reuse_changed_xqa_code(self):
        prior = json.loads((self.parent / 'ticket.json').read_text())['request']
        prior['workload']['kind'] = 'native_quality_xqa'
        prior['backend_source_sha256'] = {name: 'a' * 64 for name in XQA_SOURCES}
        self._write(self.parent / 'ticket.json', {'request': prior})
        result = json.loads((self.parent / 'result.json').read_text())
        result['request_sha256'] = digest(prior)
        result_sha = self._write(self.parent / 'result.json', result)
        self._write(self.parent / 'COMPLETE', {'result_sha256': result_sha})
        audit_path = self.root / 'input/parent_quality_audit.json'
        audit = json.loads(audit_path.read_text()); audit['terminal_result_sha256'] = result_sha
        self.request['workload'].update(kind='native_quality_xqa', parent_result_sha256=result_sha,
            parent_audit_sha256=self._write(audit_path, audit))
        self.request['backend_source_sha256'] = copy.deepcopy(prior['backend_source_sha256'])
        self._check()
        self.request['backend_source_sha256']['xllm/python/attention/xqa_decode.py'] = 'b' * 64
        with self.assertRaises(UnsafeState): self._check()


if __name__ == '__main__':
    unittest.main()
