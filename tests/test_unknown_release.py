"""Driver release recovery cannot turn UNKNOWN into experiment success."""

import copy
import unittest

from scripts.controller.approval import ApprovalError
from scripts.reconcile_unknown_release import SOURCE,terminal_from_witnesses


class UnknownReleaseTests(unittest.TestCase):
    def setUp(self):
        self.ticket={'request':{'run_id':'run','gpu_uuids':['gpu']},'request_sha256':'sha','reservation_id':'nonce'}
        self.original={'run_id':'run','request_sha256':'sha','reservation_id':'nonce','status':'UNKNOWN',
            'gpu_release_verified':False,'live_owned_containers':['container'],'arm_results':[],
            'allocation_started_monotonic_ns':100,'allocation_ended_monotonic_ns':None,
            'allocation_intervals_ns':[{'arm':'fp8_e4m3','start_ns':100,'end_ns':None}],'new_output_bytes':50}
        witness={'schema':'kvra-unknown-release-witness-v1','ok':True,'remote_writes':False,
            'run_id':'run','reservation_id':'nonce','request_sha256':'sha','boot_id':'boot','allocation_boot_id':'boot',
            'monotonic_ns':1000,'result':self.original,'result_sha256':'rawsha','supervisor_live':False,
            'run_workers':[],'containers':[{'id':'container','running':False,'pid':0,'status':'exited'}],
            'selected':{'gpu':{'memory_mib':0,'pids':[],'util_pct':0}},'new_output_bytes':60}
        self.values=[copy.deepcopy(witness),copy.deepcopy(witness)];self.values[1]['monotonic_ns']+=30_000_000_000

    def test_released_run_is_failed_and_charged_through_witness_without_changing_original(self):
        before=copy.deepcopy(self.values);e=terminal_from_witnesses(self.ticket,self.values)
        self.assertEqual(e['status'],'FAILED');self.assertEqual(e['original_result_status'],'UNKNOWN')
        self.assertFalse(e['original_result_overwritten']);self.assertFalse(e['new_GPU_admission_authorized'])
        self.assertEqual(e['allocation_ended_monotonic_ns'],self.values[1]['monotonic_ns'])
        self.assertEqual(e['new_output_bytes'],60);self.assertEqual(self.values,before)

    def test_busy_driver_or_compute_pid_is_not_a_release_proof(self):
        for field,value in [('memory_mib',1),('pids',[123]),('util_pct',1)]:
            values=copy.deepcopy(self.values);values[1]['selected']['gpu'][field]=value
            with self.subTest(field=field),self.assertRaises(ApprovalError):terminal_from_witnesses(self.ticket,values)

    def test_live_or_missing_container_supervisor_or_worker_is_not_released(self):
        for field,value in [('supervisor_live',True),('run_workers',[2]),('containers',[]),
                            ('containers',[{'id':'container','running':True,'pid':2,'status':'running'}])]:
            values=copy.deepcopy(self.values);values[1][field]=value
            with self.subTest(field=field),self.assertRaises(ApprovalError):terminal_from_witnesses(self.ticket,values)

    def test_changed_identity_result_clock_or_boot_keeps_unknown(self):
        for field,value in [('request_sha256','other'),('boot_id','new'),('allocation_boot_id','other'),
                            ('result_sha256','changed'),('monotonic_ns',30_000_000_999),('remote_writes',True)]:
            values=copy.deepcopy(self.values);values[1][field]=value
            with self.subTest(field=field),self.assertRaises(ApprovalError):terminal_from_witnesses(self.ticket,values)

    def test_completed_arm_or_bad_allocation_is_not_recovered_by_failure_path(self):
        for field,value in [('arm_results',[{'already':'complete'}]),('allocation_started_monotonic_ns',None)]:
            values=copy.deepcopy(self.values)
            for v in values:v['result'][field]=value
            with self.subTest(field=field),self.assertRaises(ApprovalError):terminal_from_witnesses(self.ticket,values)

    def test_readonly_program_compiles(self):compile(SOURCE,'<release-readonly>','exec')


if __name__=='__main__':unittest.main()
