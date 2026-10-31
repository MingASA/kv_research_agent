# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Absence recovery must not retire a delayed dispatch or disguise a GPU run."""

import copy
import unittest

from scripts.controller.approval import ApprovalError
from scripts.reconcile_absent_dispatch import AUDIT_SOURCE, terminal_from_witnesses


class AbsentDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ticket = {'request':{'run_id':'never-prepared'},'reservation_id':'nonce','request_sha256':'sha'}
        value = {'schema':'kvra-absent-dispatch-readonly-witness-v1','ok':True,
                 'run_id':'never-prepared','reservation_id':'nonce','request_sha256':'sha',
                 'root_absent':True,'remote_writes':0,'boot_id':'same-boot','monotonic_ns':1,
                 'remote_rpc_processes':[],'run_processes':[],'all_run_containers':[],'selected_compute_apps':[]}
        self.witnesses = [value,dict(value,monotonic_ns=30_000_000_001)]

    def test_stable_absence_has_zero_gpu_cost_and_no_remote_success(self) -> None:
        value = terminal_from_witnesses(self.ticket,self.witnesses)
        self.assertEqual(value['status'],'NOT_LAUNCHED')
        self.assertIsNone(value['allocation_started_monotonic_ns'])
        self.assertFalse(value['remote_result_file_created'])
        self.assertEqual(value['readonly_witnesses'],self.witnesses)

    def test_any_old_writer_process_or_container_keeps_unknown(self) -> None:
        for field in ('remote_rpc_processes','run_processes','all_run_containers'):
            for index in (0,1):
                with self.subTest(field=field,index=index):
                    values=copy.deepcopy(self.witnesses);values[index][field]=['still-present']
                    with self.assertRaises(ApprovalError):terminal_from_witnesses(self.ticket,values)

    def test_other_compute_activity_is_preserved_and_does_not_authorize_GPU_admission(self) -> None:
        values=copy.deepcopy(self.witnesses)
        for value in values:value['selected_compute_apps']=['GPU-shared, 12345']
        result=terminal_from_witnesses(self.ticket,values)
        self.assertEqual(result['status'],'NOT_LAUNCHED')
        self.assertTrue(result['shared_selected_compute_apps_present'])
        self.assertFalse(result['new_GPU_admission_authorized_by_this_proof'])
        self.assertEqual(result['gpu_release_scope'],'this_never_prepared_run_only')
        self.assertEqual(result['readonly_witnesses'],values)
        for field in ('root_absent','remote_rpc_processes','all_run_containers'):
            invalid=copy.deepcopy(values)
            invalid[0][field]=False if field=='root_absent' else ['possibly-owned']
            with self.subTest(field=field),self.assertRaises(ApprovalError):
                terminal_from_witnesses(self.ticket,invalid)

    def test_changed_identity_boot_or_short_observation_keeps_unknown(self) -> None:
        for field,value in [('root_absent',False),('reservation_id','other'),('request_sha256','other'),
                            ('run_id','other'),('boot_id','new-boot'),('monotonic_ns',30_000_000_000),
                            ('remote_writes',1),('schema','unknown')]:
            with self.subTest(field=field):
                values=copy.deepcopy(self.witnesses);values[1][field]=value
                with self.assertRaises(ApprovalError):terminal_from_witnesses(self.ticket,values)

    def test_missing_witness_and_missing_process_evidence_keep_unknown(self) -> None:
        with self.assertRaises(ApprovalError):terminal_from_witnesses(self.ticket,self.witnesses[:1])
        values=copy.deepcopy(self.witnesses);del values[0]['all_run_containers']
        with self.assertRaises(ApprovalError):terminal_from_witnesses(self.ticket,values)

    def test_remote_readonly_program_compiles(self) -> None:
        compile(AUDIT_SOURCE,'<readonly-absence-audit>','exec')


if __name__ == '__main__':
    unittest.main()
