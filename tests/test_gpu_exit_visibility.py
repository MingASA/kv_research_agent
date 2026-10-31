"""Ownership-bound NVML exit-race recovery fixtures; never access a GPU."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.controller.supervisor import DockerBackend


GPU = 'GPU-fixture-only'
CID = 'a' * 64


class GPUExitVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.backend = DockerBackend({'request': {'gpu_uuids': [GPU]}}, Path(self.temp.name))
        self.snapshot = {GPU: {'index': 0, 'pids': [123], 'memory_mib': 100, 'util_pct': 0}}
        self.empty = {GPU: {'index': 0, 'pids': [], 'memory_mib': 0, 'util_pct': 0}}
        self.proof = {'pid': 123, 'uid': os.getuid(), 'start_ticks': 456, 'cgroup': '/docker-' + CID + '.scope', 'boot_id': 'fixture'}

    def tearDown(self):
        self.temp.cleanup()

    def _observe_owned(self):
        with patch('scripts.controller.supervisor.process_identity', return_value=self.proof):
            self.assertFalse(self.backend.foreign_gpu(self.snapshot, [CID]))

    def test_only_previously_proven_owned_exit_gets_one_fresh_read(self):
        self._observe_owned()
        with patch('scripts.controller.supervisor.process_identity', side_effect=FileNotFoundError):
            with patch.object(self.backend, 'snapshot', return_value=self.empty) as fresh:
                self.assertFalse(self.backend.foreign_gpu(self.snapshot, [CID]))
                fresh.assert_called_once()
        self.assertEqual(self.backend.last_gpu_observation['retired_owned_pid'], self.proof)

    def test_unknown_exited_pid_or_prelaunch_race_still_blocks(self):
        for preobserve, owned in [(False, [CID]), (True, [])]:
            if preobserve: self._observe_owned()
            with patch('scripts.controller.supervisor.process_identity', side_effect=FileNotFoundError):
                with patch.object(self.backend, 'snapshot') as fresh:
                    self.assertTrue(self.backend.foreign_gpu(self.snapshot, owned))
                    fresh.assert_not_called()
                    self.assertEqual(self.backend.last_gpu_observation['unresolved_gpu_pid'],123)
                    self.assertIn('previous_process_proof',self.backend.last_gpu_observation)

    def test_reappearing_or_new_foreign_process_is_never_ignored(self):
        self._observe_owned()
        foreign = dict(self.proof, uid=self.proof['uid'] + 1, cgroup='/foreign')
        with patch('scripts.controller.supervisor.process_identity', side_effect=[FileNotFoundError, foreign]):
            another = {GPU: dict(self.snapshot[GPU], pids=[789])}
            with patch.object(self.backend, 'snapshot', return_value=another):
                self.assertTrue(self.backend.foreign_gpu(self.snapshot, [CID]))
        self.assertEqual(self.backend.last_gpu_observation['reason'], 'live_unowned_gpu_process')

    def test_pid_still_reported_or_second_unknown_race_stays_blocked(self):
        self._observe_owned()
        for fresh in [self.snapshot, {GPU: dict(self.snapshot[GPU], pids=[789])}]:
            with patch('scripts.controller.supervisor.process_identity', side_effect=FileNotFoundError):
                with patch.object(self.backend, 'snapshot', return_value=fresh):
                    self.assertTrue(self.backend.foreign_gpu(self.snapshot, [CID]))

    def test_only_same_previously_owned_zombie_can_wait_for_actual_driver_release(self):
        self._observe_owned()
        retired = dict(self.proof, cgroup='/')
        lifecycle = {k:retired[k] for k in ('pid','uid','start_ticks')}; lifecycle['state'] = 'Z'
        with patch('scripts.controller.supervisor.process_identity', return_value=retired), \
             patch('scripts.controller.supervisor.process_lifecycle', return_value=lifecycle), \
             patch('scripts.controller.supervisor.time.sleep'), \
             patch.object(self.backend, 'snapshot', side_effect=[self.snapshot, self.empty]) as fresh:
            self.assertFalse(self.backend.foreign_gpu(self.snapshot, [CID]))
            self.assertEqual(fresh.call_count, 2)

    def test_live_cgroup_change_reused_pid_or_unresolved_zombie_stays_blocked(self):
        self._observe_owned()
        retired = dict(self.proof, cgroup='/')
        for change in ('live', 'reused', 'still_reported'):
            lifecycle = {k:retired[k] for k in ('pid','uid','start_ticks')}; lifecycle['state'] = 'S' if change == 'live' else 'Z'
            if change == 'reused': lifecycle['start_ticks'] += 1
            with self.subTest(change=change), \
                 patch('scripts.controller.supervisor.process_identity', return_value=retired), \
                 patch('scripts.controller.supervisor.process_lifecycle', return_value=lifecycle), \
                 patch('scripts.controller.supervisor.time.sleep'), \
                 patch.object(self.backend, 'snapshot', return_value=self.snapshot) as fresh:
                self.assertTrue(self.backend.foreign_gpu(self.snapshot, [CID]))
                self.assertEqual(fresh.call_count, 3 if change == 'still_reported' else 0)

    def test_known_kernel_exiting_running_task_requires_real_driver_removal(self):
        self._observe_owned()
        retired = dict(self.proof, cgroup='/')
        lifecycle = {k: retired[k] for k in ('pid', 'uid', 'start_ticks')}
        lifecycle.update(state='R', kernel_flags=4194308)
        for fresh_values, expected in (([self.snapshot, self.empty], False), ([self.snapshot]*3, True)):
            with self.subTest(expected=expected), \
                 patch('scripts.controller.supervisor.process_identity', return_value=retired), \
                 patch('scripts.controller.supervisor.process_lifecycle', return_value=lifecycle), \
                 patch('scripts.controller.supervisor.time.sleep'), \
                 patch.object(self.backend, 'snapshot', side_effect=fresh_values) as fresh:
                self.assertEqual(self.backend.foreign_gpu(self.snapshot, [CID]), expected)
                self.assertEqual(fresh.call_count, len(fresh_values))

    def test_running_or_unknown_or_reused_pid_cannot_claim_kernel_exit_grace(self):
        self._observe_owned()
        retired = dict(self.proof, cgroup='/')
        for change in ('no_exit_flag', 'boolean_flag', 'reused', 'uid', 'unknown'):
            lifecycle = {k: retired[k] for k in ('pid', 'uid', 'start_ticks')}
            lifecycle.update(state='R', kernel_flags=4194308)
            proof = dict(retired)
            if change == 'no_exit_flag': lifecycle['kernel_flags'] = 4194304
            if change == 'boolean_flag': lifecycle['kernel_flags'] = True
            if change == 'reused': lifecycle['start_ticks'] += 1
            if change == 'uid': proof['uid'] += 1
            if change == 'unknown': proof['start_ticks'] += 1
            with self.subTest(change=change), \
                 patch('scripts.controller.supervisor.process_identity', return_value=proof), \
                 patch('scripts.controller.supervisor.process_lifecycle', return_value=lifecycle), \
                 patch.object(self.backend, 'snapshot') as fresh:
                self.assertTrue(self.backend.foreign_gpu(self.snapshot, [CID]))
                fresh.assert_not_called()

    def test_process_lifecycle_reads_real_kernel_flags_without_changing_identity(self):
        from scripts.controller.processes import process_identity, process_lifecycle
        proof = process_identity(os.getpid())
        life = process_lifecycle(os.getpid())
        self.assertEqual({k: life[k] for k in ('pid', 'uid', 'start_ticks')},
                         {k: proof[k] for k in ('pid', 'uid', 'start_ticks')})
        self.assertIs(type(life['kernel_flags']), int)
        self.assertNotIn('kernel_flags', proof)


if __name__ == '__main__':
    unittest.main()
