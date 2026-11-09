import copy
import unittest

from scripts.controller.approval import ApprovalError
from scripts.reconcile_unstarted_prepared import MARKERS, terminal_from_witnesses


class PreparedNoStartTests(unittest.TestCase):
    def setUp(self):
        self.ticket = {'request': {'run_id': 'test-run'}, 'reservation_id': 'r', 'request_sha256': 's'}
        first = dict(schema='kvra-prepared-no-start-witness-v1', ok=True, remote_writes=0,
                     run_id='test-run', reservation_id='r', request_sha256='s',
                     markers=dict.fromkeys(MARKERS, False), remote_rpc_processes=[],
                     run_processes=[], all_run_containers=[], boot_id='boot', monotonic_ns=1,
                     prepared_sha256='a'*64, inventory_sha256='b'*64, new_output_bytes=123)
        self.witnesses = [first, dict(first, monotonic_ns=30_000_000_001)]

    def test_never_allocated_preserves_remote_disk_and_consumed_identity(self):
        result = terminal_from_witnesses(self.ticket, self.witnesses)
        self.assertEqual(result['status'], 'NOT_LAUNCHED')
        self.assertEqual(result['new_output_bytes'], 123)
        self.assertIsNone(result['allocation_started_monotonic_ns'])
        self.assertFalse(result['relaunch_same_id'])
        self.assertFalse(result['remote_result_file_created'])

    def test_every_launch_marker_and_live_actor_blocks_settlement(self):
        for marker in MARKERS:
            witnesses = copy.deepcopy(self.witnesses)
            witnesses[1]['markers'][marker] = True
            with self.subTest(marker=marker), self.assertRaises(ApprovalError):
                terminal_from_witnesses(self.ticket, witnesses)
        for key in ('remote_rpc_processes', 'run_processes', 'all_run_containers'):
            witnesses = copy.deepcopy(self.witnesses)
            witnesses[0][key] = ['possible actor']
            with self.subTest(actor=key), self.assertRaises(ApprovalError):
                terminal_from_witnesses(self.ticket, witnesses)

    def test_identity_missing_evidence_reboot_mutation_and_short_window_rejected(self):
        changes = {'run_id': 'other', 'reservation_id': 'other', 'request_sha256': 'other',
                   'markers': {}, 'new_output_bytes': -1, 'boot_id': 'new-boot',
                   'inventory_sha256': 'c'*64, 'prepared_sha256': 'd'*64,
                   'monotonic_ns': 30_000_000_000, 'remote_writes': 1}
        for key, value in changes.items():
            witnesses = copy.deepcopy(self.witnesses)
            witnesses[1][key] = value
            with self.subTest(key=key), self.assertRaises(ApprovalError):
                terminal_from_witnesses(self.ticket, witnesses)


if __name__ == '__main__':
    unittest.main()
