"""Admission guards for reuse of our unsuccessful CPU-only build intermediates."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.native_cpu_build_rpc import _incremental_parent


class IncrementalCPUAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.previous = self.base / 'agent_work/builds/nextgen-native-kv-12345678-20261027d'
        (self.previous / 'source_repo').mkdir(parents=True)
        (self.previous / 'build').mkdir()
        self.source = '1' * 40
        self.values = {
            'manifest.json': {'source_commit': self.source},
            'build_result.json': {'source_commit': self.source, 'success': False, 'GPU_job': False},
            'CPU_ONLY_ADMISSION.json': {'GPU_device_nodes': []},
        }
        self.specification = {'build_id': self.previous.name, 'source_commit': self.source,
                              'container_id': '2' * 64, 'immutable_files_sha256': {}}
        self.manifest = {'build_id': 'nextgen-native-kv-87654321-20261027e',
                         'incremental_reuse': self.specification}
        self.container = {'Config': {'Labels': {'kvra.cpu_build_id': self.previous.name}},
                          'State': {'Status': 'exited', 'Pid': 0, 'Running': False, 'OOMKilled': False}}
        self.dirty = ''
        self._freeze()

    def _freeze(self):
        for name, value in self.values.items():
            data = json.dumps(value).encode()
            (self.previous / name).write_bytes(data)
            self.specification['immutable_files_sha256'][name] = hashlib.sha256(data).hexdigest()

    def _output(self, command, **kwargs):
        if command[0] == 'docker':
            return json.dumps([self.container])
        return self.dirty if 'status' in command else self.source + '\n'

    def _check(self):
        with patch('scripts.native_cpu_build_rpc.subprocess.check_output', side_effect=self._output):
            return _incremental_parent(self.base, self.manifest)

    def test_only_exact_failed_exited_cpu_build_is_admitted(self):
        self.assertEqual(self._check(), self.previous)

    def test_running_foreign_or_oom_parent_is_rejected(self):
        for key, value in [('Running', True), ('Pid', 123), ('OOMKilled', True)]:
            with self.subTest(key=key):
                old = self.container['State'][key]
                self.container['State'][key] = value
                with self.assertRaises(ValueError):
                    self._check()
                self.container['State'][key] = old
        self.container['Config']['Labels']['kvra.cpu_build_id'] = 'someone-else'
        with self.assertRaises(ValueError):
            self._check()

    def test_successful_build_or_already_linked_binary_is_preserved(self):
        self.values['build_result.json']['success'] = True
        self._freeze()
        with self.assertRaises(ValueError):
            self._check()
        self.values['build_result.json']['success'] = False
        self._freeze()
        (self.previous / 'build/bin').mkdir()
        (self.previous / 'build/bin/xllm').touch()
        with self.assertRaises(ValueError):
            self._check()

    def test_changed_failure_proof_or_dirty_source_is_rejected(self):
        (self.previous / 'build_result.json').write_text('{}')
        with self.assertRaises(ValueError):
            self._check()
        self._freeze()
        self.dirty = ' M CMakeLists.txt\n'
        with self.assertRaises(ValueError):
            self._check()


if __name__ == '__main__':
    unittest.main()
